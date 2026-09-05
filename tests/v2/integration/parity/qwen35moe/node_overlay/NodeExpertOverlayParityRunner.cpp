/**
 * @file NodeExpertOverlayParityRunner.cpp
 * @brief Runner implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Return the inventory-resolved dense continuation authority.
     *
     * Before runner setup, rank zero retains reference-pack preparation. Once
     * production has bound the topology, comparisons and CSV output move to
     * the same rank that owns logits and stage snapshots.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::parityArtifactAuthorityRank() const -> int
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
    auto Qwen35MoENodeExpertOverlayParityTest::parityReferenceGenerationCoordination() const -> ParityReferenceGenerationCoordination
    {
        return ParityReferenceGenerationCoordination::ArtifactAuthorityOnly;
    }

    /** @return Whether this process owns the inventory-resolved continuation. */
    auto Qwen35MoENodeExpertOverlayParityTest::isRootParityRank() const -> bool
    {
        return mpi_ctx_
                   ? mpi_ctx_->rank() == parityArtifactAuthorityRank()
                   : isRank0();
    }


    auto Qwen35MoENodeExpertOverlayParityTest::synchronizedDecodeWorkAvailable() -> bool
    {
        bool available = true;
        if (isRootParityRank())
        {
            available = !loadPyTorchSnapshot("decode_step0_LM_HEAD").empty() &&
                        !readDecodeTokensFromMetadata().empty();
        }
        return broadcastRootFlag(available);
    }


    auto Qwen35MoENodeExpertOverlayParityTest::producedPrefillSummary(const ParityTestSummary &summary) const -> bool
    {
        return summary.embedding_passed ||
               !summary.layer_stats.empty() ||
               summary.lm_head_passed ||
               summary.lm_head_cosine != 0.0f ||
               summary.total_layers_passed > 0;
    }


    auto Qwen35MoENodeExpertOverlayParityTest::producedDecodeSummary(const DecodeParitySummary &summary) const -> bool
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
    auto Qwen35MoENodeExpertOverlayParityTest::hasHuggingFaceMTPBranchReference(
        int reference_step,
        const std::vector<int32_t> &condition_tokens) const -> bool
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
    auto Qwen35MoENodeExpertOverlayParityTest::deferHuggingFaceMTPBranchReference(
        int call,
        int reference_step,
        int reference_depth,
        const std::vector<int32_t> &condition_tokens,
        const std::string &production_prefix,
        std::span<const std::string_view> required_stages,
        const std::filesystem::path &snapshot_csv_path) -> bool
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
            .mtp_recursive_aggregate_cosine_floor =
                config_.mtp_recursive_aggregate_cosine_floor,
            .kl_threshold = config_.mtp_kl_threshold.value_or(
                config_.kl_threshold),
            .pytorch_top1_in_topk = config_.pytorch_top1_in_topk,
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


    [[noreturn]] auto Qwen35MoENodeExpertOverlayParityTest::abortGraphNativeWorld(const std::string &reason) const -> void
    {
        const std::string message =
            "[Qwen3.5 MoE GraphNative] " + reason +
            "; aborting MPI world to avoid stranding rocm_warm/cpu_cold participants";
        LOG_ERROR(message);
        std::cerr << message << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }


    [[noreturn]] auto Qwen35MoENodeExpertOverlayParityTest::abortAfterRootThrow(const char *phase, const std::string &what) const -> void
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
    auto Qwen35MoENodeExpertOverlayParityTest::writeSparseEndpointEvidenceCsv() const -> void
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
    auto Qwen35MoENodeExpertOverlayParityTest::assertMTPTerminalHiddenMailboxEvidence() const -> void
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
     * @brief Prove TP>2 MTP executed the captured canonical reduction route.
     *
     * The typed graph policy and byte-exact grouped-vs-serial CSV comparisons
     * remain the arithmetic authorities. PerfStats is used only as route
     * evidence: it proves the live retained graph actually launched the native
     * allgather plus ascending-rank device fold selected by that policy.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertCanonicalTPAllreduceRouteEvidence() const -> void
    {
        if (!isRootParityRank() ||
            !activeModelParityCaseOrThrow()
                 .requiresCanonicalTPAllreduceMTPDiagnostics())
        {
            return;
        }

        const auto &plan = resolvedOverlayPlan();
        const auto continuation = std::find_if(
            plan.domains.begin(),
            plan.domains.end(),
            [&](const RoutedExpertDomain &domain)
            { return domain.name == plan.continuation_domain; });
        ASSERT_NE(continuation, plan.domains.end())
            << "ExpertOverlay plan has no continuation domain";
        ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("tp_allreduce"))
            << "TP>2 MTP parity requires canonical allreduce route evidence";
        std::uint64_t canonical_reductions = 0u;
        for (const auto &record :
             PerfStatsCollector::snapshot({"tp_allreduce"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "tp_allreduce" ||
                record.name != "canonical_rank_order_reductions")
            {
                continue;
            }

            const auto degree = record.tags.find("degree");
            const auto transport = record.tags.find("transport");
            const auto arithmetic = record.tags.find("arithmetic");
            const auto capture_mode = record.tags.find("capture_mode");
            EXPECT_EQ(record.phase, "collective");
            EXPECT_NE(degree, record.tags.end());
            EXPECT_NE(transport, record.tags.end());
            EXPECT_NE(arithmetic, record.tags.end());
            EXPECT_NE(capture_mode, record.tags.end());
            if (degree != record.tags.end())
            {
                EXPECT_EQ(
                    degree->second,
                    std::to_string(continuation->participants.size()));
            }
            if (transport != record.tags.end())
                EXPECT_EQ(transport->second, "native_allgather");
            if (arithmetic != record.tags.end())
                EXPECT_EQ(arithmetic->second, "device_rank_order");
            if (capture_mode != record.tags.end())
                EXPECT_EQ(capture_mode->second, "graph_capture");
            EXPECT_GT(record.value, 0.0);
            canonical_reductions +=
                static_cast<std::uint64_t>(record.value);
        }
        EXPECT_GT(canonical_reductions, 0u)
            << "TP>2 MTP retained graphs published no canonical rank-order "
               "allreduce route evidence";
    }

    /**
     * @brief Fold the identical post-loop evidence sequence on every MPI rank.
     *
     * Worker ranks enter this sequence immediately after receiving either the
     * typed terminal SHUTDOWN or nonterminal retained-runner YIELD command.
     * The continuation authority calls it in the same order after closing the
     * loop so no test-only collective can race the production command channel.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertEvidenceAfterWorkerLoopExit() -> void
    {
        writeSparseEndpointEvidenceCsv();
        writeResidencyDiagnosticsCsv();
        assertMTPTerminalHiddenMailboxEvidence();
        assertCanonicalTPAllreduceRouteEvidence();
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
    auto Qwen35MoENodeExpertOverlayParityTest::runGraphNativeProductionParityBody() -> void
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
            campaign_runner_yielded_for_reuse_ =
                mayRetainQwen122OverlayRunner();
            assertEvidenceAfterWorkerLoopExit();
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
            assertEvidenceAfterWorkerLoopExit();
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
            assertEvidenceAfterWorkerLoopExit();
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
            assertEvidenceAfterWorkerLoopExit();
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

            if (!prepareDynamicNumericalParityBoundary())
            {
                orch_runner_->shutdownMPIWorkers();
                orch_runner_->setMPICoordinatedMode(false);
                worker_shutdown.runner = nullptr;
                assertEvidenceAfterWorkerLoopExit();
                ADD_FAILURE()
                    << "Could not isolate Dynamic proof traffic from numerical parity";
                return;
            }

            if (paritySnapshotSetupMode() ==
                ParitySnapshotSetupMode::PreparedInactive)
            {
                ASSERT_TRUE(orch_runner_->activatePreparedSnapshotCapture())
                    << orch_runner_->lastError();
            }
            if (dynamic_residency_proof_lifecycle_.phase() !=
                DynamicResidencyProofPhase::NumericalParityReady)
            {
                throw std::logic_error(
                    "Dynamic numerical parity began without its typed traffic-isolation transition");
            }
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
            if (isDynamicResidencyProductionTest())
            {
                dynamic_residency_proof_lifecycle_
                    .recordMainParityComparison(activeMTPEnabled());
            }
        }

        try
        {
            writeReusedMTPHuggingFaceCheckpointEvidence();
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

        if (isRootParityRank() && isDynamicResidencyProductionTest())
        {
            if (activeMTPEnabled())
            {
                dynamic_residency_proof_lifecycle_
                    .recordMTPParityComparison();
            }
            assertParityExecutionExercisesPromotedExpert();
        }

        /*
         * End the production worker protocol before entering the evidence
         * allreduce.  This keeps the command loop and the test-only collective
         * from competing for the same MPI messages while retaining the real
         * production setup and forward path above.
         */
        if (mayRetainQwen122OverlayRunner())
        {
            /* Prefix records and request state are cell-owned even though
             * weights, arenas, streams, and captured executables survive. The
             * ordered PURGE/CLEAR/YIELD commands make every follower cross the
             * same reset frontier before it leaves the command loop. */
            const bool prefix_purged = orch_runner_->purgePrefixCache();
            if (prefix_purged)
            {
                activeClearSnapshots();
                activeClearCache();
            }
            const bool yielded =
                prefix_purged &&
                orch_runner_->yieldMPIWorkersForRetainedRunner();
            if (!yielded)
            {
                orch_runner_->shutdownMPIWorkers();
                ADD_FAILURE()
                    << "Could not publish an idle retained-runner boundary: "
                    << orch_runner_->lastError();
            }
            else
            {
                campaign_runner_yielded_for_reuse_ = true;
            }
        }
        else
        {
            orch_runner_->shutdownMPIWorkers();
        }
        orch_runner_->setMPICoordinatedMode(false);
        worker_shutdown.runner = nullptr;
        assertEvidenceAfterWorkerLoopExit();
    }

}
