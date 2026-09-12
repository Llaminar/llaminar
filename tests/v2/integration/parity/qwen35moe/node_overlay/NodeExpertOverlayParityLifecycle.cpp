/**
 * @file NodeExpertOverlayParityLifecycle.cpp
 * @brief Lifecycle implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

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
    auto Qwen35MoENodeExpertOverlayParityTest::TearDownTestSuite() -> void
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
    auto Qwen35MoENodeExpertOverlayParityTest::generatedConfig() -> const TestConfig &
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
    auto Qwen35MoENodeExpertOverlayParityTest::getTestConfig() const -> const TestConfig &
    {
        return generatedConfig();
    }

    /** @return Capacity-resolved immutable placement installed in production. */
    auto Qwen35MoENodeExpertOverlayParityTest::resolvedOverlayPlan() const -> const MoERoutedExpertPlacementPlan &
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
    auto Qwen35MoENodeExpertOverlayParityTest::parityGraphSnapshotPolicy(
        ParityForwardPhase phase) const -> ParityGraphSnapshotPolicy
    {
        auto policy = Base::parityGraphSnapshotPolicy(phase);
        auto &required =
            phase == ParityForwardPhase::Prefill
                ? policy.required_prefill_snapshot_keys
                : policy.required_decode_snapshot_keys;
        const auto append_required = [&](std::string key)
        {
            if (std::find(required.begin(), required.end(), key) ==
                required.end())
            {
                required.push_back(std::move(key));
            }
        };
        const bool require_canonical_tp_diagnostics =
            phase == ParityForwardPhase::Decode &&
            activeModelParityCaseOrThrow()
                .requiresCanonicalTPAllreduceMTPDiagnostics();
        /*
         * This policy is consumed after the runner object exists but before
         * initialize() loads its ModelContext. Consequently parityLayerCount()
         * is not yet authoritative here. The central typed model definition is
         * the immutable pre-load graph identity; setup cross-checks it against
         * the loaded GGUF before admitting inference.
         */
        const int declared_main_layer_count =
            activeModelParityCaseOrThrow().model.transformer_layers;
        const auto route_snapshot_inventory =
            modelParityExpertOverlayRouteSnapshotInventory(
                declared_main_layer_count,
                activeModelParityCaseOrThrow().mtp);
        for (const std::string &key : route_snapshot_inventory.main_model)
            append_required(key);

        /*
         * A sidecar route ledger is a live value only during a speculative
         * transaction. Select its graph outputs during immutable setup, but do
         * not demand them from ordinary prefill or teacher-forced decode. The
         * grouped-MTP proof below validates every selected sidecar ledger in
         * its exact context-qualified transaction namespace.
         */
        if (phase == ParityForwardPhase::Decode)
        {
            auto &capture_filter = policy.decode_snapshot_capture_filter;
            for (const std::string &key :
                 route_snapshot_inventory.mtp_sidecar)
            {
                if (std::find(
                        capture_filter.begin(),
                        capture_filter.end(),
                        key) == capture_filter.end())
                {
                    capture_filter.push_back(key);
                }
            }
        }

        for (int layer = 0; layer < declared_main_layer_count; ++layer)
        {
            const std::string prefix =
                "layer" + std::to_string(layer);
            /*
             * These publications exist only in the homogeneous TP>2 decode
             * graph selected for MTP batch-invariance.  Requiring them from
             * prefill or a one-device continuation domain rejects a valid
             * production graph after execution merely because that graph has
             * no allreduce/fused-residual edge to publish.
             */
            if (require_canonical_tp_diagnostics)
            {
                append_required(prefix +
                                "_ATTENTION_OUTPUT_ALLREDUCED");
                append_required(prefix +
                                "_FFN_NORM_RESIDUAL_OUT");
            }
        }
        return policy;
    }

    /** @brief The 122B matrix mathematically compares recursive MTP sidecars. */
    auto Qwen35MoENodeExpertOverlayParityTest::requiresMTPSidecarReferenceSnapshots() const -> bool
    {
        return isQwen122ProductionTest() && activeMTPEnabled();
    }

    /** @return Maximum recurrent sidecar depth admitted by the 122B matrix. */
    auto Qwen35MoENodeExpertOverlayParityTest::requiredMTPSidecarReferenceDraftDepth() const -> int
    {
        return isQwen122ProductionTest() && activeMTPEnabled()
                   ? kQwen122MaximumMTPDraftDepth
                   : 0;
    }

    /** @return Whether this process intentionally runs consecutive 122B cells. */
    auto Qwen35MoENodeExpertOverlayParityTest::mayReuseQwen122OverlayModelContext() const -> bool
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
    auto Qwen35MoENodeExpertOverlayParityTest::qwen122ModelContextPhysicalIdentity() const -> Qwen122OverlayPhysicalIdentity
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

    /** @return Whether this cell may preserve immutable runner topology. */
    auto Qwen35MoENodeExpertOverlayParityTest::mayRetainQwen122OverlayRunner() const -> bool
    {
        return mayReuseQwen122OverlayModelContext() &&
               !isDynamicResidencyProductionTest() &&
               activeMTPEnabled();
    }

    /** @return Exact immutable identity required by the active runner cell. */
    auto Qwen35MoENodeExpertOverlayParityTest::qwen122RunnerIdentity() const -> Qwen122OverlayRunnerIdentity
    {
        const auto &test_case = activeModelParityCaseOrThrow();
        return {
            .physical = qwen122ModelContextPhysicalIdentity(),
            .activation_precision = test_case.activation_precision,
            .kv_cache_precision = test_case.kv_cache_precision,
            .max_seq_len = test_case.model.max_seq_len,
            .retained_mtp_draft_capacity =
                test_case.retained_mtp_draft_capacity,
            .prefill_capture_rows =
                test_case.prefill_graph.captured_rows,
            .snapshot_mode = static_cast<std::uint8_t>(
                paritySnapshotSetupMode()),
        };
    }

    /**
     * @brief Probe the bounded cache without transferring runner ownership.
     * @param error Receives an internally inconsistent cache diagnostic.
     * @return True only for one complete exact-identity retained runner.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::hasCompatibleQwen122OverlayRunner(std::string *error) const -> bool
    {
        if (error)
            error->clear();
        if (!mayRetainQwen122OverlayRunner())
            return false;
        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (static_cast<bool>(cache.runner) !=
            cache.runner_identity.has_value())
        {
            if (error)
                *error = "122B retained-runner cache has split ownership identity";
            return false;
        }
        return cache.runner &&
               *cache.runner_identity == qwen122RunnerIdentity();
    }

    /**
     * @brief Bind evidence retention to the same probe that admits runner reuse.
     * @return Fresh on a miss; retained only for complete matching ownership.
     * @throws std::logic_error If the cache splits identity from ownership.
     *
     * Called before adoption and before the new cell clears its measurements.
     * Incompatible runners have already been retired before Base::SetUp; their
     * construction records must not certify this cell's new topology.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::productionParityRunnerEvidenceLifetime() const -> ParityRunnerEvidenceLifetime
    {
        std::string error;
        const bool retained = hasCompatibleQwen122OverlayRunner(&error);
        if (!error.empty())
            throw std::logic_error(error);
        return retained ? ParityRunnerEvidenceLifetime::RetainedRunner
                        : ParityRunnerEvidenceLifetime::FreshRunner;
    }

    /**
     * @brief Transfer an exact yielded runner into this fixture.
     * @return Sole runner ownership, or null on an exact cache miss.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::takeCompatibleQwen122OverlayRunner() -> std::unique_ptr<IOrchestrationRunner>
    {
        if (!mayRetainQwen122OverlayRunner())
            return nullptr;
        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (!cache.runner || !cache.runner_identity ||
            *cache.runner_identity != qwen122RunnerIdentity())
        {
            return nullptr;
        }
        cache.runner_identity.reset();
        return std::move(cache.runner);
    }

    /**
     * @brief Destroy an incompatible yielded runner before common setup.
     *
     * Every rank performs this transition before Base::SetUp can clear the
     * process kernel registry. The two barriers ensure no rank constructs the
     * next topology while a peer still owns streams or graph executables from
     * the prior one.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::retireIncompatibleQwen122OverlayRunner() -> void
    {
        std::unique_ptr<IOrchestrationRunner> stale;
        {
            auto &cache = qwen122OverlayModelContextCampaignCache();
            std::lock_guard<std::mutex> lock(cache.mutex);
            const bool compatible =
                mayRetainQwen122OverlayRunner() && cache.runner &&
                cache.runner_identity &&
                *cache.runner_identity == qwen122RunnerIdentity();
            if (!compatible && cache.runner)
            {
                stale = std::move(cache.runner);
                cache.runner_identity.reset();
            }
        }

        int local_stale = stale ? 1 : 0;
        int minimum_stale = 0;
        int maximum_stale = 0;
        MPI_Allreduce(
            &local_stale,
            &minimum_stale,
            1,
            MPI_INT,
            MPI_MIN,
            MPI_COMM_WORLD);
        MPI_Allreduce(
            &local_stale,
            &maximum_stale,
            1,
            MPI_INT,
            MPI_MAX,
            MPI_COMM_WORLD);
        if (minimum_stale != maximum_stale)
        {
            throw std::logic_error(
                "122B retained-runner retirement disagreed across MPI ranks");
        }
        if (maximum_stale == 0)
            return;

        MPI_Barrier(MPI_COMM_WORLD);
        stale.reset();
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Find the prior runner's rank-local prepared-weight certificate.
     *
     * A miss does not synthesize ModelContext configuration from test metadata;
     * only a previously initialized production runner may populate the slot.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::findQwen122OverlayModelContext(
        bool *cache_hit,
        std::string *error) const -> std::optional<ModelContextReuseContract>
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
    auto Qwen35MoENodeExpertOverlayParityTest::publishQwen122OverlayModelContext(
        const ModelContextReuseContract &contract,
        std::string *error) const -> bool
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


    auto Qwen35MoENodeExpertOverlayParityTest::SetUp() -> void
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

        try
        {
            if (mayReuseQwen122OverlayModelContext())
                retireIncompatibleQwen122OverlayRunner();
        }
        catch (const std::exception &exception)
        {
            FAIL() << "Failed to retire an incompatible retained runner: "
                   << exception.what();
            return;
        }

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
    auto Qwen35MoENodeExpertOverlayParityTest::TearDown() -> void
    {
        Base::TearDown();
        g_active_model_parity_case = nullptr;
        mpi_ctx_.reset();
    }

    /**
     * @brief Preserve process caches only while a static MTP runner may live.
     * @return True for exact process-campaign cells eligible for typed reuse.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::preserveParityPipelineCachesBetweenTests() const -> bool
    {
        return mayRetainQwen122OverlayRunner();
    }

    /**
     * @brief Move one successfully yielded runner into the bounded cache.
     *
     * A red cell never publishes reusable execution state. Its runner is
     * destroyed and the kernel registry is cleared here because the common
     * teardown deliberately skipped that operation for an otherwise eligible
     * retained cell.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::retireOwnedParityRunners() -> void
    {
        runner_.reset();
        if (!orch_runner_)
            return;

        if (!campaign_runner_yielded_for_reuse_ || HasFailure())
        {
            orch_runner_.reset();
            if (mayRetainQwen122OverlayRunner())
                llaminar::v2::kernels::KernelFactory::clearCache();
            return;
        }

        auto &cache = qwen122OverlayModelContextCampaignCache();
        std::lock_guard<std::mutex> lock(cache.mutex);
        if (cache.runner || cache.runner_identity || !cache.contract)
        {
            throw std::logic_error(
                "122B campaign cannot publish a yielded runner into a nonempty or uncertified cache");
        }
        cache.runner_identity = qwen122RunnerIdentity();
        cache.runner = std::move(orch_runner_);
    }


    auto Qwen35MoENodeExpertOverlayParityTest::applyModelOverrides() -> void
    {
        // Preserve the complete declarative TestConfig contract, including
        // decode depth. Reimplementing only model/prompt fields here left the
        // 122B MTP corpus at ParityConfig's historical five-step default.
        // The shared parity base is also the sole reference-pack lifecycle
        // authority: it validates model identity and sidecar schema, performs
        // one rank-zero regeneration, then publishes readiness to every rank.
        Base::applyModelOverrides();
    }


    auto Qwen35MoENodeExpertOverlayParityTest::broadcastRootFlag(bool root_value) const -> bool
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


    auto Qwen35MoENodeExpertOverlayParityTest::setupPipeline() -> bool
    {
        auto profile_scope = profileParityScope("qwen122.setup_pipeline");
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
            if (isDynamicResidencyProductionTest())
            {
                const auto metadata = topologyOnlyMetadata();
                authenticated_movement_routes_ =
                    loadAuthenticatedPrefillRouteCounts(
                        activeModelParityCaseOrThrow()
                            .model.transformer_layers,
                        metadata.num_experts);
            }
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
             * Derive the complete protected interval from the same constants
             * used by the driver. This admission check makes an epoch-crossing
             * timing/parity sequence unrepresentable if a future prompt or
             * decode horizon changes without updating the typed proof window.
             */
            const std::uint64_t protected_routed_rows =
                convergenceProtectedRoutedRows() +
                qwen35MoEConvergenceTrainingMaximumRoutedRows();
            const int configured_window =
                orchestration.moe_rebalance.window_size;
            if (protected_routed_rows >=
                    static_cast<std::uint64_t>(configured_window) ||
                configured_window !=
                    orchestration.moe_rebalance.max_window_size ||
                orchestration.moe_rebalance.window_growth_factor != 1.0F)
            {
                throw std::logic_error(
                    "Observed convergence policy does not retain its publication overlap, timing cohort, and numerical parity inside one immutable histogram epoch: routed_rows=" +
                    std::to_string(protected_routed_rows) +
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
            convergence_migration_execution_streams_ =
                orchestration.moe_rebalance
                    .resolvedMigrationExecutionStreams();
            convergence_migration_cycles_per_wave_ =
                orchestration.moe_rebalance
                    .resolvedMigrationCyclesPerWave();
            if (convergence_migration_transfer_slots_ == 0u ||
                convergence_migration_execution_streams_ == 0u ||
                convergence_migration_execution_streams_ >
                    convergence_migration_transfer_slots_ ||
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

        const auto snapshot_setup_mode = paritySnapshotSetupMode();
        bool runner_cache_hit = false;
        std::string runner_cache_error;
        const bool local_runner_cache_hit =
            hasCompatibleQwen122OverlayRunner(&runner_cache_error);
        const auto runner_admission =
            mayReuseQwen122OverlayModelContext()
                ? reachQwen122CampaignRunnerAdmission(
                      parityCoordinationCommunicator(),
                      local_runner_cache_hit,
                      runner_cache_error)
                : Qwen122CampaignRunnerAdmissionResult{
                      .admission = Qwen122CampaignRunnerAdmission::Fresh,
                      .succeeded = runner_cache_error.empty(),
                      .diagnostic = runner_cache_error,
                  };
        if (!runner_admission.succeeded)
        {
            LOG_ERROR("[Qwen3.5 MoE GraphNative] "
                      << runner_admission.diagnostic);
            return false;
        }
        if (runner_admission.admission ==
            Qwen122CampaignRunnerAdmission::Retained)
        {
            auto retained_runner =
                takeCompatibleQwen122OverlayRunner();
            if (!retained_runner ||
                !adoptRetainedOrchestrationRunner(
                    std::move(retained_runner)))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Rank-unanimous retained-runner admission had no initialized local owner");
                return false;
            }
            runner_cache_hit = true;
        }

        bool model_context_cache_hit = false;
        std::string model_context_cache_error;
        if (!runner_cache_hit)
        {
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
        }

        if (!orch_runner_->configureMTPRequestPolicy(
                makeMTPRequestPolicy(orchestration.mtp)))
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Production runner rejected the typed cell MTP policy: "
                << orch_runner_->lastError());
            return false;
        }
        const int declared_main_layer_count =
            activeModelParityCaseOrThrow().model.transformer_layers;
        const int loaded_main_layer_count = parityLayerCount();
        if (loaded_main_layer_count != declared_main_layer_count)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Typed model definition declares "
                << declared_main_layer_count << " transformer layers, but the loaded GGUF exposes "
                << loaded_main_layer_count);
            return false;
        }
        if (!certifyInstalledReferenceAdversarialPlacement())
            return false;

        /*
         * The retained contract is the exact model-owned authority consumed
         * by setupOrchestrationRunner above. Publish that typed cache outcome
         * through the shared campaign evidence field; inferring reuse later
         * from elapsed time or an incidental loader counter would make the
         * production_path.csv claim weaker than the lifecycle we just proved.
         */
        production_parity_model_context_reused_ =
            model_context_cache_hit || runner_cache_hit;

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
                runner_cache_hit
                    ? "parity_campaign_runner_cache_hits"
                    : model_context_cache_hit
                          ? "parity_campaign_model_context_cache_hits"
                          : "parity_campaign_model_context_cache_misses",
                1.0,
                "setup",
                orch_runner_->primaryDeviceId().toString(),
                {{"owner_order",
                  isRandomOwnerProductionTest() ? "random" : "ordinal"},
                 {"mtp_depth", std::to_string(activeMTPDraftDepth())},
                 {"retained_runner",
                  runner_cache_hit ? "true" : "false"}});
        }

        return true;
    }

}
