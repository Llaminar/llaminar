/**
 * @file NodeExpertOverlayParityGeometryTests.cpp
 * @brief Common model-free geometry regressions and the production test body.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /** @brief Test-only public view of the fixture's pure placement constructor. */
    class Qwen35MoEAdversarialPlacementProbe final
    : public Qwen35MoENodeExpertOverlayParityTest
    {
    public:
    using Qwen35MoENodeExpertOverlayParityTest::
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
 * Two experts on two apportioned participants already permit an exchange.
 * Its profitability belongs to measured service economics, not cardinality:
 * equal residency does not imply equal participant service rates.
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
        .routed_expert_tier = {0},
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
     * @brief A one-expert-per-device swap can reduce unequal-rate makespan.
     *
     * This is a device-free arithmetic counterexample, not a performance test.
     * The same topology opportunity must be recognized for CPU, CUDA and ROCm;
     * physical hardware, measured profitability and transfer cost remain the
     * production controller's responsibility.
     */
    TEST(Qwen35MoEDynamicMovementAxisGeometry,
         OneExpertPerParticipantCanBalanceUnequalServiceRates)
    {
        constexpr std::uint64_t hot_demand = 9u;
        constexpr std::uint64_t cold_demand = 2u;
        constexpr std::uint64_t fast_service = 2u;
        constexpr std::uint64_t slow_service = 3u;
        const auto before = std::max(hot_demand * slow_service,
                                     cold_demand * fast_service);
        const auto after = std::max(hot_demand * fast_service,
                                    cold_demand * slow_service);
        ASSERT_LT(after, before);

        for (const auto &participants :
             std::vector<std::vector<GlobalDeviceAddress>>{
                 {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
                 {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
                 {GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)}})
        {
            MoERoutedExpertPlacementPlan plan;
            RoutedExpertDomain domain;
            domain.name = "apportioned";
            domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
            domain.participants = participants;
            plan.domains = {domain};
            plan.routed_tiers = {{.name = "priority", .domain = domain.name, .priority = 7}};
            plan.placements = {{.layer = 0, .routed_expert_tier = {0, 0}}};
            EXPECT_EQ(dynamicMovementAxisContract(plan),
                      DynamicMovementAxisContract::PriorityMigrationAndParticipantBalance);

            // One endpoint cannot exchange ownership with itself, regardless
            // of expert count or the speed of its backend.
            plan.domains.front().participants.resize(1u);
            EXPECT_EQ(dynamicMovementAxisContract(plan),
                      DynamicMovementAxisContract::PriorityMigrationOnly);

            // Tensor-sharded participants jointly compute each expert rather
            // than owning independently exchangeable whole experts.
            plan.domains.front().participants = participants;
            plan.domains.front().routed_compute_policy =
                RoutedExpertComputePolicy::TensorSharded;
            EXPECT_EQ(dynamicMovementAxisContract(plan),
                      DynamicMovementAxisContract::PriorityMigrationOnly);
        }
    }

    /**
 * @brief Measured and cache-distinct convergence traffic have separate identities.
 *
 * The optimizer must observe exactly the finite workload later judged by the
 * convergence A/B. Movement-only cells instead need guaranteed full-prefill
 * evidence. Exact demand-bank closure has no nonce identity: it uses a public
 * archive purge so the authenticated workload remains stationary.
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
         request < kMaximumDynamicHistogramRequests +
                       kMaximumDynamicPublicationOverlapRequests;
         ++request)
    {
        EXPECT_EQ(
            convergenceMovementPromptIdentity(
                ConvergenceMovementTraffic::MovementProof,
                request),
            kConvergenceTimingCorpusRequests + request);
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
 * @brief Speed-witness training closes windows with cold production prefills.
 *
 * The request driver explicitly purges the archive before each request and
 * then proves a cold prefill followed by a same-epoch restore. Its finite
 * horizon must therefore count every prefill row plus each real decode forward
 * instead of pessimistically budgeting one restored-prefix token at a time.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     SpeedWitnessTrainingBudgetCountsGuaranteedColdPrefillRows)
    {
    constexpr ConvergenceTrainingTrafficPlan traffic =
        convergenceTrainingTrafficPlan();
    static_assert(traffic.valid());
    EXPECT_EQ(
        traffic.cold_prefill_rows,
        kQwen35MoEConvergenceTimingPromptRows);
    EXPECT_EQ(
        traffic.decode_forward_rows,
        kQwen35MoEConvergenceTimingDecodeForwards);
    EXPECT_EQ(traffic.guaranteedRoutedRows(), 19u);
    EXPECT_EQ(traffic.maximumRoutedRows(), 36u);
    EXPECT_EQ(kMaximumDynamicHistogramRequests, 28);
    EXPECT_GE(
        static_cast<std::uint64_t>(kMaximumDynamicHistogramRequests) *
            traffic.guaranteedRoutedRows(),
        static_cast<std::uint64_t>(
            kObservedSpeedupConvergenceWindows *
            kConvergenceHistogramWindowTokens));
}

    /**
 * @brief Movement proof admits only the later parity prefill's exact rows.
 *
 * This is the model-free regression for two adjacent campaign failures. An
 * autonomous decode at depth three optimized a valid sampled route absent from
 * Hugging Face parity. Replacing it with `forceDecodeToken()` still admitted an
 * MTP state-maintenance route at dynamic depth. Exact authenticated prefills
 * alone close the demand banks, so no decode or speculative row belongs in the
 * movement histogram.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     MovementProofTrafficUsesAuthenticatedPrefillOnly)
    {
    constexpr MovementProofTrafficPlan plan =
        movementProofTrafficPlan();
    static_assert(plan.authenticatedPrefillOnly());
    EXPECT_EQ(
        plan.authenticated_prefill_rows,
        kQwen35MoEParityTokenIds.size());
    EXPECT_EQ(plan.committed_decode_rows, 0u);
    EXPECT_EQ(plan.speculative_predictor_rows, 0u);
}

    /**
 * @brief Every parity purpose protects its own required prefix proof budget.
 *
 * The canonical matrix assigns exactly one matched A/B witness per Dynamic
 * topology and owner order. MTP owns its separate serial/prefix evidence
 * horizon; ordinary decode protects its own smaller seed/restore interval.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RequiresDemandHeadroomForTheDeclaredEvidencePurpose)
    {
    EXPECT_EQ(
        convergenceBoundaryProtectedRows(
            ConvergenceBoundaryPurpose::NumericalParity),
        qwen35MoEMaximumNumericalParityRoutedRows());
    EXPECT_EQ(
        convergenceBoundaryProtectedRows(
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort),
        convergenceProtectedRoutedRows());
    EXPECT_EQ(
        convergenceBoundaryProtectedRows(
            ConvergenceBoundaryPurpose::MTPNumericalParity),
        mtpNumericalParityProtectedRoutedRows());
}

    /**
     * @brief A quiescent four-row bank cannot retain an MTP cache/serial proof.
     *
     * Successful movement keeps the production observation window short. The
     * next prompt would rotate it and invalidate a freshly seeded prefix. Use
     * the existing exact-closure transition until the authority, not a guessed
     * post-movement cooldown, advertises enough headroom for the entire proof.
     */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         ProtectsMTPPrefixAndSerialOracleAfterSuccessfulMovement)
    {
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::CollectingDemand,
            .demand_window = {
                .generation = 2u,
                .collected_routed_rows = 1u,
                .capacity_routed_rows = 4u,
            },
            .published_progress_generation = 3u,
            .reconciled_progress_generation = 3u,
        };
        const auto purpose = ConvergenceBoundaryPurpose::MTPNumericalParity;
        const auto short_bank = classifyConvergenceBoundary(status, purpose);
        EXPECT_EQ(short_bank.state,
                  ConvergenceBoundaryState::NeedsDemandWindowClosure);
        ASSERT_TRUE(short_bank.closure.has_value());
        EXPECT_EQ(short_bank.closure->routed_rows, 3u);
        const auto required = mtpNumericalParityProtectedRoutedRows();
        EXPECT_GT(required, kQwen122MaximumMTPDraftDepth + 1u);
        status.demand_window.capacity_routed_rows = required + 1u;
        EXPECT_EQ(classifyConvergenceBoundary(status, purpose).state,
                  ConvergenceBoundaryState::NeedsDemandWindowClosure)
            << "Filling the bank exactly can publish another placement";
        ++status.demand_window.capacity_routed_rows;
        EXPECT_EQ(classifyConvergenceBoundary(status, purpose).state,
                  ConvergenceBoundaryState::Ready);
    }

    /**
     * @brief MTP-off parity also needs one epoch for its seed/restore pair.
     *
     * The nine-row production bank below is idle after successful movement,
     * but the next authenticated prefill would fill it and invalidate its own
     * prefix before decode restores it. Quiescence must request exact closure,
     * not admit a numerical proof solely because speculative decode is off.
     */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         ProtectsNonMTPPrefixAfterSuccessfulMovement)
    {
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::CollectingDemand,
            .demand_window = {
                .generation = 2u,
                .collected_routed_rows = 1u,
                .capacity_routed_rows = kQwen35MoEParityTokenIds.size(),
            },
            .published_progress_generation = 3u,
            .reconciled_progress_generation = 3u,
        };
        const auto purpose = ConvergenceBoundaryPurpose::NumericalParity;
        ASSERT_TRUE(status.quiescentBetweenWaves());
        const auto short_bank = classifyConvergenceBoundary(status, purpose);
        ASSERT_EQ(short_bank.state,
                  ConvergenceBoundaryState::NeedsDemandWindowClosure);
        ASSERT_TRUE(short_bank.closure.has_value());
        EXPECT_EQ(short_bank.closure->routed_rows, 8u);
        const auto required = qwen35MoEMaximumNumericalParityRoutedRows();
        status.demand_window.capacity_routed_rows = required + 1u;
        EXPECT_EQ(classifyConvergenceBoundary(status, purpose).state,
                  ConvergenceBoundaryState::NeedsDemandWindowClosure);
        ++status.demand_window.capacity_routed_rows;
        EXPECT_EQ(classifyConvergenceBoundary(status, purpose).state,
                  ConvergenceBoundaryState::Ready);
    }

    /** @brief Closing a demand bank cannot secretly train a different prompt. */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         DemandWindowClosureKeepsAuthenticatedTokensStationary)
    {
        constexpr std::array<int, 3> tokens = {13, 271, 760};
        EXPECT_EQ(stationaryDemandWindowPrompt(tokens, 2u),
                  (std::vector<int32_t>{13, 271}));
        EXPECT_EQ(stationaryDemandWindowPrompt(tokens, 5u),
                  (std::vector<int32_t>{13, 271, 760, 13, 271}));
        EXPECT_EQ(stationaryDemandWindowPrompt(tokens, 5u),
                  stationaryDemandWindowPrompt(tokens, 5u));
        EXPECT_THROW(stationaryDemandWindowPrompt({}, 1u), std::invalid_argument);
        EXPECT_THROW(stationaryDemandWindowPrompt(tokens, 0u), std::invalid_argument);
    }

    /**
     * @brief Evidence traffic uses every already-owned parallel transfer lane.
     *
     * Both evidence policies inherit the same topology-derived physical slots.
     * A hidden two-cycle cap would serialize a large model's convergence before
     * MTP admission, despite paying for the full transfer fabric at setup.
     */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         UsesAllocatedTransferConcurrencyForEvery122BTopology)
    {
        for (const auto &spec : qwen122OverlayTopologySpecs())
        {
            SCOPED_TRACE(spec.test_id);
            const auto policies = qwen122DynamicRuntimePolicies(spec, 48, 4096);
            EXPECT_GT(static_cast<std::uint64_t>(policies.economic_movement.max_window_size),
                      qwen35MoEMTPNumericalParityProtectedRoutedRows());
            for (const auto *policy : {
                     &policies.economic_movement,
                     &policies.economic_movement_and_observed_speedup})
            {
                EXPECT_FALSE(policy->migration_cycles_per_wave.has_value());
                EXPECT_EQ(policy->resolvedMigrationCyclesPerWave(),
                          policy->migration_transfer_slots);
                EXPECT_GE(policy->resolvedMigrationCyclesPerWave(), 48u);
            }
        }
    }

    /**
 * @brief Settlement protects prefix parity from a post-cohort fifth wave.
 *
 * The former 80-row bank admitted a 19-row publication-overlap request and a
 * 57-row timing cohort, then crossed its threshold when the canonical 9-row
 * parity prefill seeded the prefix cache. The authority correctly published a
 * fifth profitable wave, but that publication invalidated the seed before its
 * required restore. The typed boundary must reject that bank and admit the
 * definition-owned successor, including the complete partial-prefix proof.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     ProtectsNumericalParityTailFromFifthWave)
    {
    const auto make_status = [](std::uint64_t capacity)
    {
        return MoEOptimizationStatus{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::CollectingDemand,
            .demand_window = {
                .generation = 5u,
                .collected_routed_rows =
                    qwen35MoEConvergenceTimingRequestRoutedRows(),
                .capacity_routed_rows = capacity,
            },
            .published_progress_generation = 9u,
            .reconciled_progress_generation = 9u,
        };
    };

    const MoEOptimizationStatus undersized = make_status(80u);
    const ConvergenceBoundaryDecision rejected =
        classifyConvergenceBoundary(
            undersized,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort);
    EXPECT_EQ(
        rejected.state,
        ConvergenceBoundaryState::NeedsDemandWindowClosure);
    ASSERT_TRUE(rejected.closure.has_value());
    EXPECT_EQ(rejected.closure->routed_rows, 61u);

    const MoEOptimizationStatus protected_bank = make_status(
        static_cast<std::uint64_t>(
            kQwen35MoEConvergenceHistogramWindowRows));
    EXPECT_EQ(
        classifyConvergenceBoundary(
            protected_bank,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
            .state,
        ConvergenceBoundaryState::Ready);
    EXPECT_GT(
        protected_bank.demand_window.remainingRoutedRows(),
        convergenceProtectedRoutedRows());
}

    /**
     * @brief An admitted proof must include the mandatory reseed and suffix.
     *
     * The earlier 96-row bank appeared safe with 19 overlap + 57 timing + 14
     * numerical rows. It actually reached its threshold during the omitted
     * nine-row reseed, allowing invalidation before the partial hit. Admission
     * must reject that bank, reject exact equality at the wave threshold, and
     * retain the complete proof even after a movement-invalidated training
     * request has consumed its maximum possible rows.
     */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         PartialPrefixReseedCannotCrossTheNextMovementWindow)
    {
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::CollectingDemand,
            .demand_window = {
                .generation = 5u,
                .collected_routed_rows = 19u,
                .capacity_routed_rows = 96u,
            },
            .published_progress_generation = 9u,
            .reconciled_progress_generation = 9u,
        };
        const auto classify = [&status] {
            return classifyConvergenceBoundary(
                status, ConvergenceBoundaryPurpose::ObservedSpeedupCohort);
        };
        EXPECT_EQ(classify().state, ConvergenceBoundaryState::NeedsDemandWindowClosure);

        constexpr ProductionParityPrefixRestorePlan prefix(kQwen35MoEParityTokenIds.size());
        const std::array<std::uint64_t, 4> proof_rows{
            qwen35MoEConvergenceTimingCohortRoutedRows(),
            kQwen35MoEParityTokenIds.size() + kQwen35MoEMaximumParityDecodeForwards,
            static_cast<std::uint64_t>(prefix.seedTokens()),
            static_cast<std::uint64_t>(prefix.suffixTokens()),
        };
        status.demand_window.collected_routed_rows =
            convergenceTrainingTrafficPlan().maximumRoutedRows();
        status.demand_window.capacity_routed_rows =
            status.demand_window.collected_routed_rows + convergenceProtectedRoutedRows();
        EXPECT_EQ(classify().state, ConvergenceBoundaryState::NeedsDemandWindowClosure);
        status.demand_window.capacity_routed_rows = kQwen35MoEConvergenceHistogramWindowRows;
        ASSERT_EQ(classify().state, ConvergenceBoundaryState::Ready);
        for (const auto rows : proof_rows)
        {
            status.demand_window.collected_routed_rows += rows;
            EXPECT_LT(status.demand_window.collected_routed_rows,
                      status.demand_window.capacity_routed_rows);
        }
        EXPECT_EQ(status.demand_window.remainingRoutedRows(), 1u);
    }

    /**
 * @brief A partial timing bank closes exactly before its successor is measured.
 *
 * A movement wave can rotate its successor bank after an inference request is
 * admitted but before that request returns to the test-owned admission loop.
 * The typed window retains enough headroom for that single unavoidable
 * overlap, the complete matched cohort, and the canonical parity tail. If
 * unexpected extra occupancy is nevertheless observed, the passive classifier
 * must name the exact remaining rows, refuse the full bank while its wake is
 * unreconciled, and admit the complete protected interval only after
 * production rotates to an empty successor.
 */
    /** Device cadence names its closing phase and never waits for an unsent command. */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
         DeviceDemandAdmissionRetainsPhaseAndPendingCommandEdges)
    {
        MoEOptimizationStatus status{
            .authority = MoEOptimizationAuthority::Device,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::CollectingDemand,
            .demand_window = {
                .generation = 2u,
                .collected_routed_rows = 7u,
                .capacity_routed_rows = 9u,
                .scope = MoEOptimizationDemandScope::DecodeCadence,
            },
        };
        const auto closure = classifyConvergenceBoundary(
            status, ConvergenceBoundaryPurpose::NumericalParity);
        ASSERT_TRUE(closure.closure);
        EXPECT_EQ(closure.closure->routed_rows, 2u);
        EXPECT_EQ(closure.closure->scope, MoEOptimizationDemandScope::DecodeCadence);
        EXPECT_EQ(closure.closure->kind, DemandWindowClosureKind::CompleteWindow);
        status.demand_window.collected_routed_rows = 9u;
        EXPECT_EQ(classifyConvergenceBoundary(status,
            ConvergenceBoundaryPurpose::NumericalParity).state,
            ConvergenceBoundaryState::AwaitingQuiescence);
        status.demand_window.pending_submission_rows = 2u;
        const auto delivery = classifyConvergenceBoundary(
            status, ConvergenceBoundaryPurpose::NumericalParity);
        ASSERT_TRUE(delivery.closure);
        EXPECT_TRUE(delivery.closure->valid());
        EXPECT_EQ(delivery.closure->kind, DemandWindowClosureKind::DeliverPendingProgress);
        EXPECT_EQ(delivery.closure->routed_rows, 0u);
        const auto movement = classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication, status,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted);
        ASSERT_TRUE(movement.closure);
        EXPECT_EQ(movement.state,
            DynamicConvergenceSettlementState::NeedsMovementDemandWindowClosure);
        EXPECT_EQ(movement.closure->kind, DemandWindowClosureKind::DeliverPendingProgress);
    }

    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     OneOverlapFitsAndUnexpectedOccupancyClosesExactly)
    {
    constexpr std::uint64_t kOverlappingRequestRows =
        qwen35MoEConvergenceTrainingMaximumRoutedRows();
    const MoEOptimizationStatus overlap_bank{
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

    ASSERT_TRUE(overlap_bank.quiescentBetweenWaves());
    EXPECT_GT(
        overlap_bank.demand_window.remainingRoutedRows(),
        convergenceProtectedRoutedRows());
    EXPECT_EQ(
        classifyConvergenceBoundary(
            overlap_bank,
            ConvergenceBoundaryPurpose::ObservedSpeedupCohort)
            .state,
        ConvergenceBoundaryState::Ready)
        << "One publication-overlap request must not force a fifth wave";

    auto partial_bank = overlap_bank;
    partial_bank.demand_window.collected_routed_rows +=
        kConvergenceTimingPromptRows;
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
            partial_bank.demand_window.collected_routed_rows);

    EXPECT_EQ(
        classifyConvergenceBoundary(
            partial_bank,
            ConvergenceBoundaryPurpose::NumericalParity)
            .state,
        ConvergenceBoundaryState::Ready)
        << "Ordinary parity needs its smaller seed/decode bank, not timing rows";

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
 * @brief The finite horizon closes a partial bank before the final publication.
 *
 * This reproduces the segmented-prefill failure in which three required waves
 * completed and the fourth authenticated bank stopped at 244/256 routed rows.
 * While certified traffic remains available the bank must accept the normal
 * workload. Once that finite horizon closes, the typed settlement must request
 * exactly 12 ordinary rows without claiming that movement already converged.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     FiniteHorizonClosesPartialBankBeforeFinalPublication)
    {
    const MoEOptimizationStatus partial_bank{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .activity = MoEOptimizationActivityState::CollectingDemand,
        .published_movement_waves = 3u,
        .completed_movement = {.transactions = 3u},
        .demand_window = {
            .generation = 4u,
            .collected_routed_rows = 244u,
            .capacity_routed_rows = 256u,
        },
        .published_progress_generation = 1125u,
        .reconciled_progress_generation = 1125u,
    };
    ASSERT_TRUE(partial_bank.quiescentBetweenWaves());

    const DynamicConvergenceSettlementDecision open_horizon =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            partial_bank,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Open);
    EXPECT_EQ(
        open_horizon.state,
        DynamicConvergenceSettlementState::AwaitingMovement);
    EXPECT_FALSE(open_horizon.closure.has_value());
    EXPECT_FALSE(open_horizon.movementTargetSatisfied());

    const DynamicConvergenceSettlementDecision exhausted_horizon =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            partial_bank,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted);
    ASSERT_EQ(
        exhausted_horizon.state,
        DynamicConvergenceSettlementState::
            NeedsMovementDemandWindowClosure);
    ASSERT_TRUE(exhausted_horizon.closure.has_value());
    EXPECT_EQ(exhausted_horizon.closure->generation, 4u);
    EXPECT_EQ(
        exhausted_horizon.closure->observed_routed_rows,
        244u);
    EXPECT_EQ(exhausted_horizon.closure->routed_rows, 12u);
    EXPECT_FALSE(exhausted_horizon.movementTargetSatisfied());

    auto unreconciled_bank = partial_bank;
    ++unreconciled_bank.published_progress_generation;
    const DynamicConvergenceSettlementDecision unreconciled =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            unreconciled_bank,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted);
    EXPECT_EQ(
        unreconciled.state,
        DynamicConvergenceSettlementState::AwaitingMovement)
        << "An exact closure cannot race an unreconciled inference wake";
    EXPECT_FALSE(unreconciled.closure.has_value());

    auto malformed_bank = partial_bank;
    malformed_bank.demand_window.collected_routed_rows = 257u;
    EXPECT_EQ(
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            malformed_bank,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted)
            .state,
        DynamicConvergenceSettlementState::InvalidAuthorityEvidence);
}

    /**
 * @brief A slow wave's empty successor explicitly requests fresh demand.
 *
 * This reproduces the Dynamic/Ordinal production failure in which the first
 * 82-migration transaction completed after the global traffic horizon had
 * already been spent.  Publication validly rotated to an empty generation-two
 * bank.  That bank needs authenticated inference, not a passive movement wait.
 * The submitted receipt must then suppress duplicate traffic until occupancy
 * or generation advances.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     EmptySuccessorBankRequestsFreshDemandAfterSlowAsyncWave)
    {
    const MoEOptimizationStatus empty_successor{
        .authority = MoEOptimizationAuthority::Host,
        .state = MoEOptimizationLifecycleState::Active,
        .activity = MoEOptimizationActivityState::CollectingDemand,
        .published_movement_waves = 1u,
        .completed_movement = {.transactions = 1u},
        .demand_window = {
            .generation = 2u,
            .collected_routed_rows = 0u,
            .capacity_routed_rows = 256u,
        },
        .published_progress_generation = 33u,
        .reconciled_progress_generation = 33u,
    };
    ASSERT_TRUE(empty_successor.quiescentBetweenWaves());

    const DynamicConvergenceSettlementDecision exhausted =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            empty_successor,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted);
    ASSERT_EQ(
        exhausted.state,
        DynamicConvergenceSettlementState::
            NeedsMovementDemandWindowSeed);
    ASSERT_TRUE(exhausted.seed.has_value());
    EXPECT_EQ(exhausted.seed->generation, 2u);
    EXPECT_FALSE(exhausted.closure.has_value());
    EXPECT_FALSE(exhausted.movementTargetSatisfied());

    const DynamicConvergenceSettlementDecision open =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            empty_successor,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Open);
    EXPECT_EQ(
        open.state,
        DynamicConvergenceSettlementState::AwaitingMovement);
    EXPECT_FALSE(open.seed.has_value());

    const SubmittedDemandWindowAdmission submitted{
        .generation = 2u,
        .observed_routed_rows = 0u,
    };
    EXPECT_EQ(
        observeDemandWindowAdmission(submitted, empty_successor),
        DemandWindowAdmissionObservation::AwaitingPublication)
        << "An unchanged empty bank must not authorize duplicate demand";

    auto populated = empty_successor;
    populated.demand_window.collected_routed_rows = 9u;
    EXPECT_EQ(
        observeDemandWindowAdmission(submitted, populated),
        DemandWindowAdmissionObservation::Published);

    auto rotated = empty_successor;
    ++rotated.demand_window.generation;
    EXPECT_EQ(
        observeDemandWindowAdmission(submitted, rotated),
        DemandWindowAdmissionObservation::Published);

    auto regressed = empty_successor;
    --regressed.demand_window.generation;
    EXPECT_EQ(
        observeDemandWindowAdmission(submitted, regressed),
        DemandWindowAdmissionObservation::InvalidAuthorityEvidence);

    auto full = empty_successor;
    full.demand_window.collected_routed_rows = 256u;
    const DynamicConvergenceSettlementDecision full_bank =
        classifyDynamicConvergenceSettlement(
            DynamicResidencyConvergenceState::AwaitingPublication,
            full,
            ConvergenceBoundaryPurpose::NumericalParity,
            ConvergenceTrafficHorizon::Exhausted);
    EXPECT_EQ(
        full_bank.state,
        DynamicConvergenceSettlementState::AwaitingMovement);
    EXPECT_FALSE(full_bank.seed.has_value());
    EXPECT_FALSE(full_bank.closure.has_value());
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
 * @brief Topology-valid service movement cannot replace a parity witness.
 *
 * The first promotion models broad service-economy traffic and is absent from
 * the immutable Hugging Face prefill. The second promotion is selected by that
 * exact reference workload. Only the latter may close the mathematical
 * movement proof, while both remain valid production ledger entries.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RequiresAuthenticatedWorkloadPromotion)
    {
    const DynamicResidencyConvergenceTarget target{
        .minimum_published_waves = 1u,
        .minimum_authenticated_promotions = 1u,
        .axis_contract =
            DynamicMovementAxisContract::PriorityMigrationOnly,
    };
    const DynamicResidencyConvergenceOrigin origin{};
    const MoEOptimizationStatus status{
        .authority = MoEOptimizationAuthority::Device,
        .state = MoEOptimizationLifecycleState::Active,
        .published_movement_waves = 1u,
        .completed_movement = {
            .transactions = 1u,
            .commands = 2u,
            .promotions = 1u,
            .demotions = 1u,
        },
    };
    std::vector<std::vector<std::uint64_t>> routes(
        2u, std::vector<std::uint64_t>(4u, 0u));
    routes[1][3] = 2u;

    const auto movement = [](
                              int layer,
                              int expert,
                              std::size_t cycle_index)
    {
        return MoEOptimizationMovementEdge{
            .authority = MoEOptimizationAuthority::Device,
            .transaction = 1u,
            .candidate_epoch = 2u,
            .layer = layer,
            .expert = expert,
            .cycle_index = cycle_index,
            .cycle_size = 2u,
            .direction =
                cycle_index == 0u
                    ? MoEOptimizationMovementDirection::Promotion
                    : MoEOptimizationMovementDirection::Demotion,
            .axis = MoEOptimizationMovementAxis::TierResidency,
            .source_participant = cycle_index == 0u ? 1 : 0,
            .destination_participant = cycle_index == 0u ? 0 : 1,
            .source_priority = cycle_index == 0u ? 1 : 0,
            .destination_priority = cycle_index == 0u ? 0 : 1,
            .source_device = cycle_index == 0u
                                 ? DeviceId::rocm(0)
                                 : DeviceId::cuda(0),
            .destination_device = cycle_index == 0u
                                      ? DeviceId::cuda(0)
                                      : DeviceId::rocm(0),
        };
    };

    MoEOptimizationMovementLedger service_only{
        .edges = {movement(0, 1, 0u), movement(0, 2, 1u)},
    };
    const auto topology = classifyDynamicResidencyConvergence(
        target, origin, status, service_only);
    EXPECT_EQ(topology, DynamicResidencyConvergenceState::Satisfied);
    const auto service_promotion =
        classifyAuthenticatedPromotionConvergence(
            target, origin, service_only, routes);
    EXPECT_EQ(
        service_promotion,
        AuthenticatedPromotionConvergenceState::AwaitingPromotion);
    EXPECT_EQ(
        requireAuthenticatedPromotion(topology, service_promotion),
        DynamicResidencyConvergenceState::
            AwaitingAuthenticatedPromotion);

    auto authenticated = service_only;
    authenticated.edges.push_back(movement(1, 3, 0u));
    EXPECT_EQ(
        classifyAuthenticatedPromotionConvergence(
            target, origin, authenticated, routes),
        AuthenticatedPromotionConvergenceState::Satisfied);
    EXPECT_EQ(
        requireAuthenticatedPromotion(
            topology,
            classifyAuthenticatedPromotionConvergence(
                target, origin, authenticated, routes)),
        DynamicResidencyConvergenceState::Satisfied);

    auto malformed = authenticated;
    malformed.edges.back().expert = 4;
    EXPECT_EQ(
        classifyAuthenticatedPromotionConvergence(
            target, origin, malformed, routes),
        AuthenticatedPromotionConvergenceState::
            InvalidAuthorityEvidence);
}

    /**
 * @brief Numerical parity is admitted only after proof-prefix isolation.
 *
 * Both movement-only and measured-speedup paths can leave a valid prefix in
 * the final placement epoch. This device-free state-machine regression makes
 * it impossible for either path to enter the mathematical prefill before the
 * coordinated production purge has completed.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RequiresProofPrefixIsolationBeforeNumericalParity)
    {
    const DynamicResidencyConvergenceOrigin origin{
        .published_waves = 0u,
        .completed_transactions = 0u,
        .ledger_edges = 0u,
        .host_admissions = 0u,
    };

    DynamicResidencyProofLifecycle movement_only;
    movement_only.beginEconomyCertification(origin);
    movement_only.completeEconomyCertification();
    EXPECT_THROW(
        movement_only.recordNumericalParityIsolation(),
        std::logic_error);
    movement_only.recordMovementTarget();
    movement_only.recordMovementBoundarySettled();
    movement_only.recordNumericalParityIsolation();
    EXPECT_EQ(
        movement_only.phase(),
        DynamicResidencyProofPhase::NumericalParityReady);

    DynamicResidencyProofLifecycle measured_speedup;
    measured_speedup.beginEconomyCertification(origin);
    measured_speedup.completeEconomyCertification();
    measured_speedup.recordInitialCohort();
    measured_speedup.recordMovementTarget();
    measured_speedup.recordMovementBoundarySettled();
    measured_speedup.recordConvergedCohort();
    measured_speedup.recordNumericalParityIsolation();
    EXPECT_EQ(
        measured_speedup.phase(),
        DynamicResidencyProofPhase::NumericalParityReady);
}

    /**
 * @brief MTP promotions cannot be judged before their checkpoint producer.
 *
 * Main-model prefill/decode and the primary sidecar checkpoint publish
 * different routed-expert namespaces. This regression rejects the historical
 * call order in which the promotion epilogue ran after main parity but before
 * the MTP observer had a chance to contribute a layer-48 witness.
 */
    TEST(Qwen35MoEDynamicConvergenceLifecycle,
     RequiresPrimaryMTPCheckpointBeforePromotedExpertEvidence)
    {
    const DynamicResidencyConvergenceOrigin origin{};
    const auto ready_for_parity = [&]()
    {
        DynamicResidencyProofLifecycle lifecycle;
        lifecycle.beginEconomyCertification(origin);
        lifecycle.completeEconomyCertification();
        lifecycle.recordMovementTarget();
        lifecycle.recordMovementBoundarySettled();
        lifecycle.recordNumericalParityIsolation();
        return lifecycle;
    };

    auto mtp = ready_for_parity();
    EXPECT_THROW(mtp.recordMTPParityComparison(), std::logic_error);
    mtp.recordMainParityComparison(true);
    EXPECT_EQ(
        mtp.phase(), DynamicResidencyProofPhase::AwaitingMTPParity);
    EXPECT_FALSE(mtp.numericalEvidenceComplete());
    EXPECT_THROW(mtp.recordMainParityComparison(true), std::logic_error);
    mtp.recordMTPParityComparison();
    EXPECT_EQ(
        mtp.phase(), DynamicResidencyProofPhase::NumericalEvidenceComplete);
    EXPECT_TRUE(mtp.numericalEvidenceComplete());

    auto no_mtp = ready_for_parity();
    no_mtp.recordMainParityComparison(false);
    EXPECT_EQ(
        no_mtp.phase(),
        DynamicResidencyProofPhase::NumericalEvidenceComplete);
    EXPECT_TRUE(no_mtp.numericalEvidenceComplete());
    EXPECT_THROW(no_mtp.recordMTPParityComparison(), std::logic_error);
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

    /**
 * @brief Execute one generated production 35B or 122B policy cell.
 *
 * Every parameter owns a fresh request runner but shares one bounded
 * process-resident model/prepared-weight authority where supported. Runtime
 * policy comes from `GetParam()`; the stable parameter name is diagnostic
 * output only and is never parsed to recover execution intent.
 */
    TEST_P(Qwen35MoENodeExpertOverlayParityTest, ProductionParity)
    {
    runGraphNativeProductionParityBody();
}


}
