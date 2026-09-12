/**
 * @file NodeExpertOverlayParityMovement.cpp
 * @brief Movement implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 * Keep convergence-driver edits local to this shard so they do not rebuild
 * the independent MTP, routing, or reference implementations.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Feed bounded real requests until distributed residency actually moves.
     *
     * Physical topology preparation, measured service certification, and the
     * initial stationary timing cohort are complete before this method begins.
     * This driver replays those exact prompt identities while optimizing, so
     * phase-weighted placement sees the same prefill and decode trajectories
     * that the A/B gate will judge. Typed demand-window closure keeps the
     * authenticated prompt stationary and purges the reusable prefix archive
     * to guarantee actual routed rows. The background worker owns interval
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
    auto Qwen35MoENodeExpertOverlayParityTest::driveDynamicResidencyToDistributedMigration() -> bool
    {
        auto profile_scope = profileParityScope(
            "qwen122.dynamic.drive_migration");
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
        if (convergence_target.minimum_published_waves >
            std::numeric_limits<std::uint64_t>::max() -
                convergence_target.minimum_authenticated_promotions)
        {
            throw std::overflow_error(
                "Dynamic movement-proof publication budget overflowed");
        }
        const std::uint64_t movement_proof_publication_budget =
            convergence_target.minimum_published_waves +
            convergence_target.minimum_authenticated_promotions;
        const int maximum_ordinary_requests =
            (requiresObservedConvergenceSpeedup()
                 ? kMaximumDynamicHistogramRequests
                 : movementProofHistogramRequestBudget(
                       activeModelParityCaseOrThrow()
                           .dynamic_rebalance.window_size,
                       std::min(
                           activeModelParityCaseOrThrow()
                               .dynamic_rebalance.window_size,
                           static_cast<int>(config_.token_ids.size())),
                       movement_proof_publication_budget)) +
            kMaximumDynamicPublicationOverlapRequests;
        const int vocabulary_size = orch_runner_->vocabSize();
        if (config_.token_ids.empty() || vocabulary_size <= 4'096)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Economy service coverage requires a non-empty authenticated prompt and a valid vocabulary");
            return false;
        }
        int ordinary_requests = 0;
        int admitted_requests = 0;
        const MoEOptimizationStatus initial_optimization =
            optimizationStatus();
        if (!initial_optimization.active() ||
            !initial_optimization.demand_window.valid())
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Dynamic movement began without an active authority and valid demand bank");
            return false;
        }
        const std::uint64_t initial_demand_generation =
            initial_optimization.demand_window.generation;
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
            NeedsDemandWindowSeed,
            NeedsMovementDemandWindowClosure,
            NeedsBoundaryDemandWindowClosure,
            Failed,
        };
        struct ConvergenceBoundarySettlementDecision
        {
            ConvergenceBoundarySettlement state =
                ConvergenceBoundarySettlement::Failed;
            std::optional<DemandWindowSeed> seed;
            std::optional<DemandWindowClosure> closure;
        };
        const ConvergenceBoundaryPurpose boundary_purpose =
            requiresObservedConvergenceSpeedup()
                ? ConvergenceBoundaryPurpose::ObservedSpeedupCohort
                : activeMTPEnabled()
                ? ConvergenceBoundaryPurpose::MTPNumericalParity
                : ConvergenceBoundaryPurpose::NumericalParity;
        std::optional<SubmittedDemandWindowAdmission>
            submitted_demand_admission;
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
                classifyDynamicResidencyProofConvergence(
                    convergence_target,
                    convergence_origin,
                    current,
                    ledger);
            const DynamicConvergenceSettlementDecision settlement =
                classifyDynamicConvergenceSettlement(
                    convergence,
                    current,
                    boundary_purpose,
                    ConvergenceTrafficHorizon::Open);
            if (settlement.state ==
                DynamicConvergenceSettlementState::
                    InvalidAuthorityEvidence)
            {
                throw std::logic_error(
                    "ExpertOverlay convergence observer received regressed, truncated, or malformed authority evidence");
            }
            if (!settlement.movementTargetSatisfied())
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
         * Quiescence alone does not reserve the next request's epoch. Every
         * numerical proof needs room for its prefix seed and complete restore.
         * MTP numerical parity additionally needs room for its serial oracle,
         * grouped transaction and prefix-restore checks; observed-speedup
         * cells need their timing cohort and numerical tail. The authority
         * publishes exact active-bank occupancy for that purpose. A newly
         * rotated empty bank requests one authenticated seed; an insufficient
         * partial bank returns one typed exact closure. A submitted-admission
         * receipt then waits for authoritative occupancy or generation
         * progress before any further traffic can be issued. No demand is
         * discarded, movement is not paused, and the test never mutates
         * controller policy.
         */
        const auto settleConvergenceBoundary = [&]()
            -> ConvergenceBoundarySettlementDecision
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
                    return {
                        .state = ConvergenceBoundarySettlement::Failed,
                    };
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

                if (submitted_demand_admission)
                {
                    const DemandWindowAdmissionObservation observation =
                        observeDemandWindowAdmission(
                            *submitted_demand_admission,
                            last_status);
                    if (observation ==
                        DemandWindowAdmissionObservation::
                            InvalidAuthorityEvidence)
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Demand admission observed regressed or malformed authority evidence: submitted_generation="
                            << submitted_demand_admission->generation
                            << " submitted_rows="
                            << submitted_demand_admission
                                   ->observed_routed_rows
                            << " current_generation="
                            << last_status.demand_window.generation
                            << " current_rows="
                            << last_status.demand_window
                                   .collected_routed_rows);
                        return {
                            .state =
                                ConvergenceBoundarySettlement::Failed,
                        };
                    }
                    if (observation ==
                        DemandWindowAdmissionObservation::
                            AwaitingPublication)
                    {
                        /* A submitted prefill owns the next progress edge.
                         * Reclassifying the unchanged empty/partial bank would
                         * duplicate that traffic and could overfill it. */
                        if (observed_at >= no_progress_deadline)
                            break;
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(1));
                        continue;
                    }
                    submitted_demand_admission.reset();
                }
                const MoEOptimizationMovementLedger ledger =
                    orch_runner_->moeOptimizationMovementLedger();
                const DynamicResidencyConvergenceState convergence =
                    classifyDynamicResidencyProofConvergence(
                        convergence_target,
                        convergence_origin,
                        last_status,
                        ledger);
                const DynamicConvergenceSettlementDecision settlement =
                    classifyDynamicConvergenceSettlement(
                        convergence,
                        last_status,
                        boundary_purpose,
                        ConvergenceTrafficHorizon::Exhausted);
                if (settlement.state ==
                    DynamicConvergenceSettlementState::
                        InvalidAuthorityEvidence)
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Between-wave settlement observed invalid authoritative movement evidence");
                    return {
                        .state = ConvergenceBoundarySettlement::Failed,
                    };
                }
                if (settlement.movementTargetSatisfied())
                {
                    /* The request that closed the final demand bank can leave
                     * its profitable wave in the background authority after
                     * the finite traffic horizon. Settlement owns that same
                     * lifecycle edge, so publish the typed target transition
                     * here as well as in the request-boundary fast path. */
                    dynamic_residency_proof_lifecycle_
                        .recordMovementTarget();
                }

                switch (settlement.state)
                {
                case DynamicConvergenceSettlementState::AwaitingMovement:
                case DynamicConvergenceSettlementState::
                    AwaitingBoundaryQuiescence:
                    break;
                case DynamicConvergenceSettlementState::Ready:
                    dynamic_residency_proof_lifecycle_
                        .recordMovementBoundarySettled();
                    return {
                        .state = ConvergenceBoundarySettlement::Settled,
                    };
                case DynamicConvergenceSettlementState::
                    InvalidAuthorityEvidence:
                    throw std::logic_error(
                        "Invalid Dynamic settlement escaped the fatal classifier branch");
                case DynamicConvergenceSettlementState::
                    NeedsMovementDemandWindowSeed:
                    if (!settlement.seed || settlement.closure)
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Dynamic settlement classified demand seeding without one valid typed seed");
                        return {
                            .state =
                                ConvergenceBoundarySettlement::Failed,
                        };
                    }
                    LOG_INFO(
                        "[Qwen3.5 MoE GraphNative] Quiescent Dynamic authority requires authenticated demand in a newly rotated bank before the next movement publication: generation="
                        << settlement.seed->generation
                        << " capacity_rows="
                        << last_status.demand_window
                               .capacity_routed_rows);
                    return {
                        .state = ConvergenceBoundarySettlement::
                            NeedsDemandWindowSeed,
                        .seed = settlement.seed,
                    };
                case DynamicConvergenceSettlementState::
                    NeedsMovementDemandWindowClosure:
                case DynamicConvergenceSettlementState::
                    NeedsBoundaryDemandWindowClosure:
                    if (!settlement.closure ||
                        !settlement.closure->valid())
                    {
                        LOG_ERROR(
                            "[Qwen3.5 MoE GraphNative] Dynamic settlement classified closure work without a valid typed closure");
                        return {
                            .state =
                                ConvergenceBoundarySettlement::Failed,
                        };
                    }
                    LOG_INFO(
                        "[Qwen3.5 MoE GraphNative] Quiescent Dynamic authority requires exact demand-window closure "
                        << (settlement.movementTargetSatisfied()
                                ? "before the post-movement evidence cohort"
                                : "before the next required movement publication")
                        << ": generation="
                        << settlement.closure->generation
                        << " collected_rows="
                        << last_status.demand_window
                               .collected_routed_rows
                        << " capacity_rows="
                        << last_status.demand_window
                               .capacity_routed_rows
                        << " closure_rows="
                        << settlement.closure->routed_rows);
                    return {
                        .state = settlement.movementTargetSatisfied()
                                     ? ConvergenceBoundarySettlement::
                                           NeedsBoundaryDemandWindowClosure
                                     : ConvergenceBoundarySettlement::
                                           NeedsMovementDemandWindowClosure,
                        .closure = settlement.closure,
                    };
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
            return {
                .state = ConvergenceBoundarySettlement::Failed,
            };
        };

        /** Preserve complete production diagnostics before coordinated failure. */
        const auto reportConvergenceFailure = [&]() -> bool
        {
            if (isRootParityRank())
            {
                writeResidencyDiagnosticsCsv();
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Dynamic residency did not produce the required profitable publication epoch(s) and topology-valid movement after "
                    << admitted_requests << " admitted requests ("
                    << ordinary_requests << "/"
                    << maximum_ordinary_requests
                    << " ordinary-horizon requests)\n"
                    << PerfStatsCollector::summaryString(
                           {"moe_overlay_residency",
                            "moe_overlay_controller"}));
            }
            return false;
        };

        /**
         * @brief Bound exact settlement work by authoritative bank generation.
         *
         * The ordinary request count cannot account for traffic served while a
         * bank is frozen by asynchronous movement.  Demand generations can:
         * each completed bank advances exactly once.  The convergence target
         * admits one bank per required publication plus one bank in which an
         * authenticated promotion may become numerically witnessable.
         */
        const auto movementDemandGenerationAdmissible =
            [&](std::uint64_t generation) noexcept
        {
            if (generation < initial_demand_generation)
                return false;
            return generation - initial_demand_generation <
                   movement_proof_publication_budget;
        };

        for (;;)
        {
            /* One observation at the request boundary determines whether the
             * driver admits ordinary traffic or enters typed settlement. */
            const MovementObservation movement = observeMovement();
            if (movement == MovementObservation::Continue &&
                ordinary_requests < maximum_ordinary_requests)
            {
                const int corpus_request =
                    convergenceMovementPromptIdentity(
                        requiresObservedConvergenceSpeedup()
                            ? ConvergenceMovementTraffic::MeasuredWorkload
                            : ConvergenceMovementTraffic::MovementProof,
                        admitted_requests);
                const bool completed =
                    requiresObservedConvergenceSpeedup()
                        ? replayStationaryConvergenceRequest(
                              corpus_request)
                        : replayStationaryMovementProofRequest(
                              corpus_request);
                if (!completed)
                    return false;
                ++ordinary_requests;
                ++admitted_requests;
                continue;
            }

            const ConvergenceBoundarySettlementDecision settlement =
                settleConvergenceBoundary();
            switch (settlement.state)
            {
            case ConvergenceBoundarySettlement::Settled:
                return true;
            case ConvergenceBoundarySettlement::Failed:
                return reportConvergenceFailure();
            case ConvergenceBoundarySettlement::NeedsDemandWindowSeed:
            {
                if (!settlement.seed || settlement.closure)
                {
                    throw std::logic_error(
                        "Dynamic settlement returned malformed demand-seed work");
                }
                if (!movementDemandGenerationAdmissible(
                        settlement.seed->generation))
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Dynamic movement exhausted its authoritative demand-generation budget without satisfying convergence: initial_generation="
                        << initial_demand_generation
                        << " requested_generation="
                        << settlement.seed->generation
                        << " generation_budget="
                        << movement_proof_publication_budget);
                    return reportConvergenceFailure();
                }

                const int corpus_request =
                    convergenceMovementPromptIdentity(
                        requiresObservedConvergenceSpeedup()
                            ? ConvergenceMovementTraffic::MeasuredWorkload
                            : ConvergenceMovementTraffic::MovementProof,
                        admitted_requests);
                const bool seeded =
                    requiresObservedConvergenceSpeedup()
                        ? replayStationaryConvergenceRequest(
                              corpus_request)
                        : replayStationaryMovementProofRequest(
                              corpus_request);
                if (!seeded)
                    return false;
                if (!requiresObservedConvergenceSpeedup() &&
                    !runDynamicEconomyDecode(
                         1, "demand-window-seed-boundary")
                         .has_value())
                {
                    return false;
                }
                submitted_demand_admission =
                    SubmittedDemandWindowAdmission{
                        .generation = settlement.seed->generation,
                        .observed_routed_rows = 0u,
                    };
                ++admitted_requests;
                continue;
            }
            case ConvergenceBoundarySettlement::
                NeedsMovementDemandWindowClosure:
            case ConvergenceBoundarySettlement::
                NeedsBoundaryDemandWindowClosure:
            {
                if (!settlement.closure || settlement.seed ||
                    !settlement.closure->valid())
                {
                    throw std::logic_error(
                        "Dynamic settlement returned malformed demand-closure work");
                }
                const bool movement_closure =
                    settlement.state ==
                    ConvergenceBoundarySettlement::
                        NeedsMovementDemandWindowClosure;
                if (movement_closure &&
                    !movementDemandGenerationAdmissible(
                        settlement.closure->generation))
                {
                    LOG_ERROR(
                        "[Qwen3.5 MoE GraphNative] Dynamic movement exhausted its authoritative demand-generation budget before exact closure: initial_generation="
                        << initial_demand_generation
                        << " requested_generation="
                        << settlement.closure->generation
                        << " generation_budget="
                        << movement_proof_publication_budget);
                    return reportConvergenceFailure();
                }

                if (settlement.closure->kind ==
                    DemandWindowClosureKind::DeliverPendingProgress)
                {
                    // An ordinary captured command delivers the already-retired
                    // sideband to every participant. Polling alone cannot do it.
                    if (!runDynamicEconomyDecode(1, "pending-progress-delivery"))
                        return false;
                    submitted_demand_admission = SubmittedDemandWindowAdmission{
                        .generation = settlement.closure->generation,
                        .observed_routed_rows = settlement.closure->observed_routed_rows,
                    };
                    ++admitted_requests;
                    continue;
                }
                if (settlement.closure->scope ==
                    MoEOptimizationDemandScope::DecodeCadence)
                {
                    // Cadence counts committed tokens, not router activations.
                    // Budget one retains the real captured serial transaction
                    // even in an MTP cell and advances exactly one notification.
                    for (std::uint64_t row = 0u;
                         row < settlement.closure->routed_rows; ++row)
                    {
                        const auto complete = runDynamicEconomyDecode(
                            1, "decode-cadence-closure");
                        if (!complete)
                            return false;
                        if (*complete && !runDynamicEconomyPrefill(
                                config_.token_ids, "decode-cadence-next-request"))
                            return false;
                    }
                    // Deliver the final decode sideband through the next real
                    // request, including any naturally occurring prefix hit.
                    if (!runDynamicEconomyPrefill(
                            config_.token_ids, "decode-cadence-publication"))
                        return false;
                    submitted_demand_admission = SubmittedDemandWindowAdmission{
                        .generation = settlement.closure->generation,
                        .observed_routed_rows = settlement.closure->observed_routed_rows,
                    };
                    ++admitted_requests;
                    continue;
                }

                // End the preceding request's cache lease before eviction.
                // Keep the workload stationary: a nonce in the leading token
                // changes router demand, it is not merely a cache-busting key.
                activeClearCache();
                if (!orch_runner_->purgePrefixCache())
                {
                    LOG_ERROR("[Qwen3.5 MoE GraphNative] Demand closure could not purge reusable prefixes: "
                              << orch_runner_->lastError());
                    return false;
                }
                if (!runDynamicEconomyPrefill(
                        makeDemandWindowClosurePrompt(
                            settlement.closure->routed_rows),
                        "demand-window-closure"))
                {
                    return false;
                }
                /* This consumes already-produced prefill logits and publishes
                 * the request-progress wake without adding a routed row. */
                if (!runDynamicEconomyDecode(
                         1, "demand-window-closure-boundary")
                         .has_value())
                {
                    return false;
                }
                submitted_demand_admission =
                    SubmittedDemandWindowAdmission{
                        .generation = settlement.closure->generation,
                        .observed_routed_rows = settlement.closure
                                                    ->observed_routed_rows,
                    };
                ++admitted_requests;
                continue;
            }
            }
            throw std::logic_error(
                "Unhandled Dynamic convergence driver directive");
        }
    }

}
