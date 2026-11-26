/**
 * @file OrchestrationRunner.cpp
 * @brief Coordinates request lifecycle, graph execution, and rank-local inference.
 *
 * This implementation turns validated orchestration configuration into a
 * rank-local production execution plan, owns request lifecycle state, and
 * drives prefill, decode, MTP, prefix-cache, and dynamic MoE maintenance.
 * The inventory-resolved continuation rank publishes coordinated commands
 * while follower ranks execute the same graph/collective schedule. Failure
 * handling is deliberately strict:
 * once a worker can no longer participate in that schedule, it aborts the MPI
 * job rather than returning to a command loop that would leave a peer blocked
 * in an unmatched collective.
 * Seeded CPU sampling uses the same logical-position draw identity for ordinary
 * outputs, batched prefill and MTP verification; RNG call order is not authority.
 * Optional finite benchmark traces export every actual host controller observation;
 * they neither mirror GPU control nor retain unbounded serving chronology.
 * CPU stochastic sampling owns penalties inside its host distribution builder;
 * it never pre-mutates the model row through a backend sampling operation.
 * Planning metadata is read once by the admission root and published through
 * the common rank-initialization lifecycle before participant placement. A
 * missing model cannot manufacture a synthetic geometry in production admission.
 * Admission-only completion remains distinct from inference readiness, so dry
 * runs neither enter inference shutdown collectives nor seal unprepared weights.
 * Model-aware topology is resolved by ResolvedRankOrchestration; this runner
 * installs that result and its requested weight identity together, before any
 * capacity-frozen placement replaces the declarative request. There is no second
 * overlay-binding or role-selection path.
 * Overlay candidates use the shared BOM adapter inside the common typed rank
 * initialization lifecycle. Capture-policy errors reach consensus before any
 * peer may enter the physical-budget collective.
 * Prompt tokens and prefix/KV admission share one frozen continuation group;
 * expert-only followers receive command metadata and sparse tickets, not tokens.
 * Admission publishes the already resolved stochastic request seeds. Sampling
 * and verification share that entropy across response windows and graph reuse.
 * Forced MTP output preserves an already-returned resident carry before
 * replacing the next condition; unreturned bonus replacement stays distinct.
 *
 * @author David Sanftenberg
 * @date January 2026
 *
 * Physical TP validation borrows the already-resolved LocalTP endpoint order;
 * early structural validation never manufactures physical participant IDs.
 * CPU serial storage declarations survive admission and runtime projection.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 * Global pipeline graph completion belongs to the original rank's parent:
 * nested TP children retain shared sources until its complete serving family
 * and every stage ownership proof succeed, including terminal MTP operands.
 */


#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{

    std::shared_ptr<MoERoutedExpertPlacementPlan> freezeMoEExpertOverlayPlanForModel(
        IModelContext &model_ctx,
        const std::shared_ptr<MoERoutedExpertPlacementPlan> &plan,
        const MTPRuntimeConfig &mtp)
    {
        if (!plan)
            return nullptr;

        InferenceRunnerConfig runner_config;
        runner_config.moe_routed_expert_plan = plan;
        runner_config.mtp = mtp;
        return resolveMoERoutedExpertPlacementPlanForModel(model_ctx, runner_config);
    }

    /**
     * @brief Size a topology-wide command page for every representable move.
     *
     * One command can name one `(participant, layer, expert)` destination.
     * Reserving that complete Cartesian product keeps Static, Dynamic, and
     * current-batch LLEP on the same immutable captured address family. Runtime
     * policy may publish far fewer entries, but increasing cycles per wave must
     * never require graph recapture or silently truncate a transaction.
     */
    std::uint32_t completeMoEOverlayDeviceCommandCapacity(
        int num_layers,
        int num_experts,
        std::size_t participant_count)
    {
        if (num_layers <= 0 || num_experts <= 0 || participant_count == 0u)
        {
            throw std::invalid_argument(
                "Device-resident ExpertOverlay command capacity requires positive model and participant geometry");
        }
        const std::uint64_t capacity =
            static_cast<std::uint64_t>(num_layers) *
            static_cast<std::uint64_t>(num_experts) *
            static_cast<std::uint64_t>(participant_count);
        if (capacity == 0u ||
            capacity > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(
                "Device-resident ExpertOverlay command capacity exceeds its fixed-width ABI");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    // =========================================================================
    // Construction
    // =========================================================================

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
        : config_(std::move(config)), plan_builder_(std::move(plan_builder)), sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
    }

    OrchestrationRunner::OrchestrationRunner(
        std::shared_ptr<IMPIContext> execution_context,
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder)
        : OrchestrationRunner(std::move(config), std::move(plan_builder))
    {
        if (!execution_context || execution_context->communicator() == MPI_COMM_NULL)
            throw std::invalid_argument("Runner admission requires an exact live execution context");
        mpi_ctx_ = std::move(execution_context);
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder,
        std::shared_ptr<ModelContext> model_ctx)
        : config_(std::move(config)),
          plan_builder_(std::move(plan_builder)),
          model_ctx_(std::move(model_ctx)),
          sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
        if (!plan_builder_)
            plan_builder_ = createExecutionPlanBuilder();
        if (!model_ctx_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner preloaded ModelContext must be non-null");
        }
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        std::unique_ptr<IExecutionPlanBuilder> plan_builder,
        ModelContextReuseContract reuse_contract)
        : config_(std::move(config)),
          plan_builder_(std::move(plan_builder)),
          model_ctx_(std::move(reuse_contract.context)),
          reusable_execution_workspaces_(
              std::move(reuse_contract.reusable_execution_workspaces)),
          retained_prepared_weight_plan_(
              std::move(reuse_contract.prepared_weight_plan)),
          retained_prepared_routed_weight_plan_(
              std::move(reuse_contract.prepared_routed_weight_plan)),
          retained_expert_overlay_memory_admission_(
              std::move(
                  reuse_contract.expert_overlay_memory_admission)),
          moe_pipeline_prepared_plan_(std::move(reuse_contract.pipeline_prepared_plan)),
          physical_memory_authority_(
              std::move(reuse_contract.physical_memory_authority)),
          retained_routed_weight_authority_identity_(
              std::move(reuse_contract.routed_weight_authority_identity)),
          retained_expert_overlay_migration_profile_(
              std::move(
                  reuse_contract.expert_overlay_migration_profile)),
          retained_expert_overlay_migration_profile_identity_(
              std::move(
                  reuse_contract
                      .expert_overlay_migration_profile_identity)),
          model_context_reuse_authority_(
              std::move(reuse_contract.reuse_authority)),
          sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
        if (!plan_builder_)
        {
            plan_builder_ = createExecutionPlanBuilder();
        }
        if (!model_ctx_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract must contain a ModelContext");
        }
        if (!model_context_reuse_authority_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has no exclusive lifecycle authority");
        }
        if (!reusable_execution_workspaces_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has no reusable execution-workspace authority");
        }
        if (!physical_memory_authority_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has no physical-memory authority");
        }
        physical_memory_admission_ =
            physical_memory_authority_->admission();
        if (!physical_memory_admission_)
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has no physical-memory admission proof");
        }
        if (static_cast<bool>(
                retained_expert_overlay_migration_profile_) !=
            !retained_expert_overlay_migration_profile_identity_.empty())
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has inconsistent ExpertOverlay migration-profile evidence");
        }
        if (static_cast<bool>(retained_prepared_routed_weight_plan_) !=
            static_cast<bool>(
                retained_expert_overlay_memory_admission_))
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has inconsistent ExpertOverlay placement and memory authorities");
        }
        if (retained_expert_overlay_memory_admission_ &&
            retained_expert_overlay_memory_admission_
                    ->physical_memory_admission.get() !=
                physical_memory_admission_.get())
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract split its ExpertOverlay and physical-memory authorities");
        }
        if (moe_pipeline_prepared_plan_ &&
            (retained_prepared_routed_weight_plan_ || retained_expert_overlay_memory_admission_ ||
             moe_pipeline_prepared_plan_->admission()->physicalAdmission() != physical_memory_admission_))
            throw std::invalid_argument("Prepared MoE pipeline must retain one exclusive aggregate physical authority");
        const auto installed_memory_authority =
            model_ctx_->concreteWeightManager()
                ? model_ctx_->concreteWeightManager()
                      ->physicalMemoryAuthority()
                : nullptr;
        if (!installed_memory_authority ||
            installed_memory_authority.get() !=
                physical_memory_authority_.get())
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract does not carry the ModelContext's physical-memory authority");
        }
        if (retained_expert_overlay_migration_profile_ &&
            (!retained_expert_overlay_migration_profile_->valid() ||
             retained_routed_weight_authority_identity_.empty()))
        {
            throw std::invalid_argument(
                "OrchestrationRunner reuse contract has malformed or unscoped ExpertOverlay migration-profile evidence");
        }
        // Reject a foreign or differently prepared predictor before consuming
        // the original reusable lease/retention receipt. Such rejection has no
        // runtime owner to seal and must leave the model reusable by its owner.
        const auto& original_loader=model_ctx_->concreteLoader();
        original_loader.requireSharedPredictorSource(config_.shared_predictor,
            original_loader.predictorMaterialization());
        if(config_.shared_predictor){
            // Automatic intent is resolved by the SAME metadata-only startup
            // authority used during planning, against this retained source inventory.
            auto predictor_request=config_;
            resolveMTPStartupPolicy(predictor_request,ModelMemoryProfile::fromLoader(original_loader));
            original_loader.requireSharedPredictorSource(config_.shared_predictor,
                retainsMTPGraphCapacity(predictor_request.mtp)?SharedPredictorMaterialization::Active:
                    SharedPredictorMaterialization::Inactive);
            // Carry the canonical startup result forward; later planning applies
            // the same idempotent resolver instead of reconstructing another policy.
            config_=std::move(predictor_request);
            requested_routed_weight_authority_identity_=routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,config_);
        }
        std::string reuse_error;
        if (!model_context_reuse_authority_->acquireRunnerExclusive(
                &reuse_error))
        {
            throw std::logic_error(
                "OrchestrationRunner could not acquire prepared model context: " +
                (reuse_error.empty()
                     ? std::string("unknown reuse lifecycle failure")
                     : reuse_error));
        }
        imported_model_context_reuse_authority_ = true;
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true), sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          runner_(std::move(runner)),
          initialization_lifecycle_(OrchestrationInitializationLifecycle::Completion::Inference), sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
    }

    OrchestrationRunner::OrchestrationRunner(
        OrchestrationConfig config,
        RankExecutionPlan plan,
        std::unique_ptr<IInferenceRunner> runner,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : config_(std::move(config)), plan_(std::move(plan)), plan_built_(true),
          mpi_ctx_(std::move(mpi_ctx)), runner_(std::move(runner)),
          initialization_lifecycle_(OrchestrationInitializationLifecycle::Completion::Inference), sampler_(0)
    {
        requested_routed_weight_authority_identity_ =
            routedWeightAuthorityRequestIdentity(
                config_.moe_routed_expert_plan,
                config_);
        mpi_coordinated_root_rank_ = coordinatedRootRankForRunner(
            config_.moe_routed_expert_plan,
            mpi_ctx_, runner_.get());
        if (!initializeContinuationRequestGroup())
            throw std::invalid_argument(lastError());
    }

    OrchestrationRunner::~OrchestrationRunner()
    {
        shutdown();
    }

    // =========================================================================
    // Lifecycle
    // =========================================================================

    bool OrchestrationRunner::runRankSynchronizedInitializationPhase(
        std::string_view workflow,
        RankInitializationPhaseIdentity identity,
        InitializationConsensusScope scope,
        const std::function<bool()> &step)
    {
        const auto consensus =
            [this, scope](
                RankInitializationPhaseIdentity phase_identity,
                RankInitializationLocalOutcome local_outcome)
                -> RankInitializationConsensusResult
        {
            const auto local_result = [local_outcome]
            {
                return RankInitializationConsensusResult{
                    .outcome =
                        local_outcome ==
                                RankInitializationLocalOutcome::Succeeded
                            ? RankInitializationConsensusOutcome::
                                  AllRanksSucceeded
                            : RankInitializationConsensusOutcome::
                                  AtLeastOneRankFailed,
                    .detail = {},
                };
            };

            if (scope == InitializationConsensusScope::BootstrapContext && !mpi_ctx_)
            {
                if (!requiresMultiRankMPI(config_))
                    return local_result();

                int mpi_initialized = 0;
                const int initialized_result =
                    MPI_Initialized(&mpi_initialized);
                if (initialized_result != MPI_SUCCESS || !mpi_initialized)
                {
                    return {
                        .outcome = RankInitializationConsensusOutcome::
                            TransportFailure,
                        .detail =
                            "MPI world is unavailable for bootstrap consensus",
                    };
                }
                return MPIRankInitializationConsensus::reach(
                    MPI_COMM_WORLD,
                    phase_identity,
                    local_outcome);
            }

            const auto consensus_context =
                scope == InitializationConsensusScope::ExpertOverlayContext &&
                        moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_;
            if (!consensus_context)
            {
                return {
                    .outcome =
                        RankInitializationConsensusOutcome::TransportFailure,
                    .detail =
                        "runner MPI context is unavailable after bootstrap",
                };
            }
            int consensus_size = consensus_context->world_size();
            if (scope == InitializationConsensusScope::BootstrapContext &&
                (consensus_context->communicator() == MPI_COMM_NULL ||
                 MPI_Comm_size(consensus_context->communicator(), &consensus_size) != MPI_SUCCESS))
                return {.outcome = RankInitializationConsensusOutcome::TransportFailure,
                        .detail = "Admitted MPI communicator is unavailable for bootstrap consensus"};
            if (consensus_size <= 1)
                return local_result();

            return MPIRankInitializationConsensus::reach(
                consensus_context->communicator(),
                phase_identity,
                local_outcome);
        };

        const auto result = RankInitializationLifecycle::execute(
            identity,
            step,
            consensus);
        if (result.succeeded())
            return true;

        if (result.status ==
            RankInitializationPhaseStatus::LocalStepReturnedFailure)
        {
            bool has_step_diagnostic = false;
            {
                std::lock_guard<std::mutex> lock(error_mutex_);
                has_step_diagnostic = !last_error_.empty();
            }
            if (has_step_diagnostic)
                return false;
        }
        return setError(result.diagnostic(workflow));
    }

    bool OrchestrationRunner::initialize()
    {
        // Direct API callers share the public startup authority boundary.
        // Reject authored scope before MPI, planning, model or graph work.
        if (config_.mtp.depth_policy.scope != MTPDepthPolicyScope{})
            return setError("MTP learned-policy scope is runtime-derived and cannot be authored in startup configuration");
        if (debugEnv().runtime_debug.mtp_depth_transaction_trace)
        {
            try
            {
                (void)sampling_math::resolveMTPDiagnosticRecordMode(
                    debugEnv().runtime_debug.mtp_device_phase_snapshots,
                    DebugEnv::isTruthyEnv("LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS"), true);
            }
            catch (const std::invalid_argument &error) { return setError(error.what()); }
            const auto scope = config_.benchmark_mode && !config_.serve_mode
                ? sampling_math::MTPDepthTransactionTraceScope::BoundedBenchmark
                : sampling_math::MTPDepthTransactionTraceScope::Other;
            if (!debugEnv().runtime_debug.benchmark_iteration_overrides_valid ||
                !validMTPDepthTransactionTracePolicy(config_.mtp) ||
                !sampling_math::validMTPDepthTransactionTraceScope(scope,
                    debugEnv().runtime_debug.benchmark_warmup_iterations,
                    debugEnv().runtime_debug.benchmark_iterations) ||
                !sampling_math::validMTPDepthTransactionTraceAdmission(
                    sampling_math::DeviceGenerationAdmissionKind::NewRequest,
                    config_.batch_size, config_.n_predict < 0 ? 128 : config_.n_predict))
                return setError("MTP depth-transaction trace requires a fresh scalar bounded benchmark with enabled positive-depth MTP (CPU bypass unsupported), warmup0/measure1, tokens1..256");
            if (!PerfStatsCollector::isDomainEnabled("mtp"))
                return setError("MTP depth-transaction trace requires enabled PerfStats collection of the mtp domain");
        }
        if (initialization_lifecycle_.readyForInference())
        {
            return true;
        }
        if (initialization_lifecycle_.admissionComplete())
        {
            return setError("Admission-only initialization cannot serve inference; shut down this runner before full initialization");
        }
        prefix_shutdown_lifecycle_ = PrefixShutdownLifecycle::Pending;

        std::uint32_t phase_ordinal = 0;
        const auto run_phase =
            [this, &phase_ordinal](
                const char *step_name,
                const std::function<bool()> &step,
                InitializationConsensusScope scope =
                    InitializationConsensusScope::ActiveContext)
        {
            return runRankSynchronizedInitializationPhase(
                "Initialization",
                RankInitializationPhaseIdentity{
                    .ordinal = phase_ordinal++,
                    .name = step_name,
                },
                scope,
                step);
        };

        {
            // Step 1: Initialize MPI if needed
            if (!run_phase(
                    "initializeMPI",
                    [&] { return initializeMPI(); },
                    InitializationConsensusScope::BootstrapContext))
            {
                return false;
            }

            // Step 2: Build execution plan (if not pre-built)
            if (!run_phase(
                    "buildExecutionPlan",
                    [&] { return buildExecutionPlan(); }))
            {
                return false;
            }

            // Step 3: Setup LOCAL TP context
            if (!run_phase(
                    "setupLocalTPContext",
                    [&] { return setupLocalTPContext(); }))
            {
                return false;
            }

            // Step 3.5: Setup LOCAL PP context
            if (!run_phase(
                    "setupLocalPPContext",
                    [&] { return setupLocalPPContext(); }))
            {
                return false;
            }

            // Step 4: Load model weights
            if (!run_phase(
                    "loadWeights", [&] { return loadWeights(); }))
            {
                return false;
            }

            /*
             * A retained ModelContext can outlive its previous runner while
             * that runner's post-prefill worker releases host pages.  Cross
             * the explicit dependency before this runner performs capacity
             * admission or graph/workspace allocation.  The rank-synchronized
             * phase also prevents one topology participant from entering
             * serving while a peer is still reclaiming its prior context.
             */
            if (!run_phase(
                    "awaitMmapReclaimBeforeRuntimeAllocation",
                    [&]
                    {
                        return awaitMmapReclaimBeforeRuntimeAllocation();
                    }))
            {
                return false;
            }

            // Step 4b: Validate topology and requested context before either
            // becomes an input to physical capacity admission.
            if (!run_phase(
                    "validateTPPPConfiguration",
                    [&] { return validateTPPPConfiguration(); }))
            {
                return false;
            }

            if (!run_phase(
                    "validateContextLength",
                    [&] { return validateContextLength(); }))
            {
                return false;
            }

            // Step 4c: Freeze model-aware placement and captured-prefill
            // capacity together before constructing any endpoint graph.
            if (!run_phase(
                    "freezeMoEExpertOverlayPlanForLoadedModel",
                    [&]
                    {
                        return freezeMoEExpertOverlayPlanForLoadedModel();
                    }))
            {
                return false;
            }

            // Step 4c.5: Prompt payloads and prefix/KV consensus have the same
            // continuation owners. Freeze one group before request admission
            // so expert-only followers never receive unused token bytes.
            if (!run_phase(
                    "initializeContinuationRequestGroup",
                    [&]
                    {
                        return initializeContinuationRequestGroup();
                    }))
            {
                return false;
            }

            // Step 4d: Create one owner-map authority before any participant
            // graph can bind its dispatch ticket and runtime placement banks.
            if (!run_phase(
                    "initializeMoEExpertOverlayResidencyAuthority",
                    [&]
                    {
                        return initializeMoEExpertOverlayResidencyAuthority();
                    }))
            {
                return false;
            }

            // Step 5: Validate the frozen memory plan.
            if (!run_phase(
                    "validateMemoryPlan",
                    [&] { return validateMemoryPlan(); }))
            {
                return false;
            }

            // A distributed ExpertOverlay is one sparse-collective protocol,
            // not independent rank-local prefill loops. Freeze its common
            // bucket ladder after all local memory plans are known and before
            // any root or expert-only graph binds persistent arena addresses.
            if (!run_phase(
                    "establishMoEOverlayPrefillScheduleContract",
                    [&]
                    {
                        return establishMoEOverlayPrefillScheduleContract();
                    }))
            {
                return false;
            }

            /*
             * Same-node shared activation pages need real consumer-rank NUMA
             * first-touch. Complete that bilateral rendezvous now, while both
             * ranks are still in the same lightweight setup phase, rather
             * than letting graph builders race against prepared-weight work.
             */
            if (!run_phase(
                    "initializeMoEOverlayRankBatchTransports",
                    [&]
                    {
                        return initializeMoEOverlayRankBatchTransports();
                    }))
            {
                return false;
            }

            // Print the consolidated startup banner only after all preflight
            // passes, while retaining the same rank-wide failure boundary.
            if (!run_phase(
                    "printStartupBanner",
                    [&]
                    {
                        printStartupBanner();
                        return true;
                    }))
            {
                return false;
            }

            // Step 6: Build compute graph
            if (!run_phase(
                    "buildComputeGraph",
                    [&] { return buildComputeGraph(); }))
            {
                return false;
            }

            // Graph construction has completed its original all-rank phase.
            // Validate every local alias in a separate phase before session
            // topology discovery can enter any original WORLD collective.
            if (!run_phase("validateOriginalRankPipelineTransportContext", [&] {
                    const auto *global = dynamic_cast<const GlobalOrchestrator *>(runner_.get());
                    if (global && global->requiresOriginalRankGPUTransport())
                        global->validateOriginalRankGPUTransportContext(mpi_ctx_);
                    return true;
                })) return false;
            if (!run_phase("createOriginalRankPipelineTransportSession", [&] {
                    auto *global = dynamic_cast<GlobalOrchestrator *>(runner_.get());
                    if (global && global->requiresOriginalRankGPUTransport())
                    {
                        if (rank_pipeline_transfer_session_)
                            return setError("Original rank pipeline session was already constructed");
                        rank_pipeline_transfer_session_ = CapturedTransferRankSession::create(mpi_ctx_);
                        global->bindOriginalRankGPUTransportSession(mpi_ctx_, rank_pipeline_transfer_session_);
                    }
                    return true;
                })) return false;

            // Bind immutable economics once, before serving capture or warmup.
            // No model directory scan or geometry allocation enters decode.
            if (!run_phase("bindMTPDepthPolicyScopeGeometry", [&]
                {
                    if (model_ctx_ && model_ctx_->ggufDirectory())
                    {
                        const auto projection = projectMTPDepthPolicyScopeGeometry(
                            plan_, planningModelMetadata().memoryProfile(),
                            resolveMTPRetainedDraftCapacity(retainedMTPConfig()), cluster_inventory_);
                        mtp_depth_policy_scope_geometry_ = projection.scope;
                        mtp_depth_policy_scope_generic_reason_ = projection.generic_reason;
                    }
                    return true;
                }))
                return false;

            /*
             * Diagnostic D2D copies are part of native graph topology. Apply
             * the complete caller policy after participant graph construction
             * but before either ordinary or ExpertOverlay serving families are
             * captured. A request-time topology change is then a deliberate
             * reconfiguration, never an accidental parity warmup.
             */
            if (!run_phase(
                    "applyConfiguredSnapshotCaptureSetup",
                    [&] { return applyConfiguredSnapshotCaptureSetup(); }))
            {
                return false;
            }

            /*
             * Graph construction resolves the only authoritative prepared
             * engine lifetimes. Materialize inactive slots and transfer lanes,
             * install their exact progress epochs, and only then capture every
             * continuation/follower serving graph. The maintenance composer
             * performs that entire ordered transaction before it starts any
             * background worker.
             */
            if (!run_phase(
                    "initializeMoEExpertOverlayResidencyMaintenance",
                    [&]
                    {
                        return initializeMoEExpertOverlayResidencyMaintenance();
                    }))
            {
                return false;
            }

            /*
             * Graph-family capture is a setup transition for every GPU runner,
             * not a first-request side effect. ExpertOverlay performs the same
             * transition while composing its transfer-progress branch above;
             * ordinary dense/TP runners enter it here. Keeping this before
             * request admission prevents graph warmup from publishing a prefix
             * that changes the semantics of the first real request.
             */
            if (!run_phase(
                    "materializeOrdinaryServingGraphFamilyWithoutLaunch",
                    [&]
                    {
                        return materializeOrdinaryServingGraphFamilyWithoutLaunch();
                    }))
            {
                return false;
            }

            /*
             * A prepared-inactive diagnostic policy deliberately owns two
             * executable variants. The preceding setup pass materialized the
             * diagnostic topology. Select snapshots-disabled identity and seal
             * the lean variant now, before maintenance threads or request-ticket
             * authority can make graph construction a runtime operation.
             */
            if (!run_phase(
                    "deactivatePreparedSnapshotCaptureAndMaterializeLeanServingFamily",
                    [&]
                    {
                        return deactivatePreparedSnapshotCaptureAndMaterializeLeanServingFamily();
                    }))
            {
                return false;
            }

            if (!run_phase(
                    "initializeMoEOverlayInferenceTransactions",
                    [&]
                    {
                        return initializeMoEOverlayInferenceTransactions();
                    }))
            {
                return false;
            }

            /*
             * Service construction is deliberately dormant through serving
             * and MTP transaction setup. This rank-wide phase starts progress
             * only after every peer has admitted its immutable producer
             * streams and committed the initial retained families. A later
             * depth/policy-specific publication identity remains legal, but
             * its typed capture-activity scope makes maintenance return
             * Pending until native capture has closed on every exit path.
             */
            if (!run_phase(
                    "startMoEExpertOverlayResidencyMaintenance",
                    [&]
                    {
                        return startMoEExpertOverlayResidencyMaintenance();
                    },
                    InitializationConsensusScope::ExpertOverlayContext))
            {
                return false;
            }

            if (!run_phase(
                    "publishMoEOverlayCollectiveRequestGeneration",
                    [&]
                    {
                        return publishMoEOverlayCollectiveRequestGeneration(
                            "initialization");
                    }))
            {
                return false;
            }

            /*
             * GPU inference owns logits on device for the entire request.
             * Sampling and speculative publication consume the resident tensor;
             * only compact response tokens cross to the host. Install this once
             * at the lifecycle boundary rather than toggling gather behavior per
             * request, which could expose stale host data between transactions.
             * Tests that deliberately inspect full logits must explicitly opt
             * into host observation after initialization.
             */
            if (!run_phase(
                    "configureResidentInferencePolicy",
                    [&]
                    {
                        if (runner_ &&
                            runner_->primaryDeviceId().is_gpu())
                        {
                            runner_->setSkipLogitsGatherPrefill(true);
                            runner_->setSkipLogitsGatherDecode(true);
                        }
                        if (!runner_ ||
                            runner_->configureMTPRequestStopTokens(
                                stop_tokens_))
                        {
                            return true;
                        }
                        return setError(
                            "Inference runner rejected request stop-token policy");
                    }))
            {
                return false;
            }

            // One loaded-revision policy owns sampling and prompt defaults.
            // Tensor architecture alone cannot select a later model's template.
            if (!run_phase(
                    "cacheModelGenerationMetadata",
                    [&]
                    {
                        if (!model_ctx_)
                            return true;

                        const std::string arch = model_ctx_->architecture();
                        if (!SchemaFactoryRegistry::isSupported(arch))
                            return true;

                        auto factory =
                            SchemaFactoryRegistry::getFactory(arch);
                        if (!factory)
                            return true;

                        model_generation_policy_ = ModelGenerationPolicy::fromMetadata(*model_ctx_->loader());
                        if (tokenizer_)
                            model_generation_policy_.applyChatTemplate(*tokenizer_);
                        const auto &recommended_sampling_params =
                            model_generation_policy_.forMode(ThinkingMode::ModelDefault);
                        stop_thinking_prompt_ =
                            factory->getStopThinkingPrompt();
                        tool_call_format_ = factory->getToolCallFormat();
                        if (recommended_sampling_params.has_penalties() ||
                            recommended_sampling_params.temperature != 1.0f)
                        {
                            LOG_DEBUG(
                                "[OrchestrationRunner] Model-recommended sampling: "
                                << "temp="
                                << recommended_sampling_params.temperature
                                << " top_p="
                                << recommended_sampling_params.top_p
                                << " top_k="
                                << recommended_sampling_params.top_k
                                << " presence_penalty="
                                << recommended_sampling_params.presence_penalty
                                << " frequency_penalty="
                                << recommended_sampling_params.frequency_penalty
                                << " repetition_penalty=" << recommended_sampling_params.repetition_penalty);
                        }
                        if (!stop_thinking_prompt_.empty())
                        {
                            LOG_DEBUG(
                                "[OrchestrationRunner] Stop-thinking prompt configured ("
                                << stop_thinking_prompt_.size() << " chars)");
                        }
                        return true;
                    }))
            {
                return false;
            }

            if (!run_phase("requireImmutableRowTablePlacementComplete", [&] {
                    if (plan_.runtime.ngram_table_placement)
                    {
                        plan_.runtime.ngram_table_placement->requireRequestedPolicy(config_.ngram_table_policy);
                        if (!model_ctx_ || !physical_memory_authority_)
                            return setError("Ngram completion lost its admitted model or physical authority");
                        model_ctx_->requireImmutableRowTablePlacementComplete(
                            *plan_.runtime.ngram_table_placement, physical_memory_authority_);
                    }
                    return true;
                }))
                return false;

            /*
             * Capture one coherent owner-by-owner ledger view at the exact
             * ready-for-inference boundary. This is diagnostic only: admission
             * and allocation remain controlled by the typed authority, while
             * the report lets us close any missing materialization owner
             * without trying to reconstruct truth from driver free-memory or
             * independently timed PerfStats counters.
             */
            if (physical_memory_authority_)
            {
                LOG_DEBUG(
                    "[PhysicalMemoryAuthority] ready-for-inference attestation:\n"
                    << physical_memory_authority_
                           ->rankAttestationSummary());
                if (PerfStatsCollector::isDomainEnabled("physical_memory"))
                {
                    // Export the authority's own typed proof, not a new RSS or
                    // model-file-size budget. Lazy owner pools need not be
                    // fully materialized, but every live byte must remain
                    // inside its admitted/reserved owner envelope.
                    for (const auto &bom : physical_memory_authority_
                                               ->admission()->plan().resources())
                    {
                        if (bom.resource().world_rank !=
                            physical_memory_authority_->worldRank())
                            continue;
                        PerfStatsCollector::addCounter(
                            "physical_memory", "resource_admission", 1.0,
                            "initialized", bom.resource().device.toString(),
                            {{"rank", std::to_string(bom.resource().world_rank)},
                             {"incremental_bytes", std::to_string(bom.incrementalBytes())},
                             {"available_bytes", std::to_string(bom.resource().admission_available_bytes)},
                             {"owner_count", std::to_string(PhysicalMemoryBOM::ownerCount())}});
                    }
                    const auto attestation =
                        physical_memory_authority_->rankAttestation();
                    for (const auto &row : attestation)
                    {
                        PerfStatsCollector::addCounter(
                            "physical_memory", "owner_attestation", 1.0,
                            "initialized", row.identity.device.toString(),
                            {{"rank", std::to_string(row.identity.world_rank)},
                             {"owner", std::string(toString(row.owner))},
                             {"planned_new_bytes", std::to_string(row.planned_new_bytes)},
                             {"planned_resident_bytes", std::to_string(row.planned_resident_bytes)},
                             {"materialized_new_bytes", std::to_string(row.materialized_new_bytes)},
                             {"committed_new_bytes", std::to_string(row.committed_new_bytes)},
                             {"adopted_resident_bytes", std::to_string(row.adopted_resident_bytes)}});
                    }
                }
            }

            initialization_lifecycle_.publish(
                OrchestrationInitializationLifecycle::Completion::Inference);
            LOG_DEBUG("OrchestrationRunner initialized successfully");
            return true;
        }
    }

    bool OrchestrationRunner::initializeForDryRun()
    {
        // Direct API callers share the public startup authority boundary.
        // Reject authored scope before MPI, planning, model or graph work.
        if (config_.mtp.depth_policy.scope != MTPDepthPolicyScope{})
            return setError("MTP learned-policy scope is runtime-derived and cannot be authored in startup configuration");
        if (initialization_lifecycle_.admissionComplete())
        {
            return true;
        }

        std::uint32_t phase_ordinal = 0;
        const auto run_phase =
            [this, &phase_ordinal](
                const char *step_name,
                const std::function<bool()> &step,
                InitializationConsensusScope scope =
                    InitializationConsensusScope::ActiveContext)
        {
            return runRankSynchronizedInitializationPhase(
                "Dry-run preflight",
                RankInitializationPhaseIdentity{
                    .ordinal = phase_ordinal++,
                    .name = step_name,
                },
                scope,
                step);
        };

        if (!run_phase(
                "initializeMPI",
                [&] { return initializeMPI(); },
                InitializationConsensusScope::BootstrapContext))
            return false;
        if (!run_phase(
                "buildExecutionPlan",
                [&] { return buildExecutionPlan(); }))
            return false;
        if (!run_phase(
                "setupLocalTPContext",
                [&] { return setupLocalTPContext(); }))
            return false;
        if (!run_phase(
                "setupLocalPPContext",
                [&] { return setupLocalPPContext(); }))
            return false;
        if (!run_phase(
                "loadWeights",
                [&]
                {
                    return loadWeights(/*prepopulate_page_cache=*/false);
                }))
            return false;
        if (!run_phase(
                "awaitMmapReclaimBeforeRuntimeAllocation",
                [&]
                {
                    return awaitMmapReclaimBeforeRuntimeAllocation();
                }))
            return false;
        if (!run_phase(
                "validateTPPPConfiguration",
                [&] { return validateTPPPConfiguration(); }))
            return false;
        if (!run_phase(
                "validateContextLength",
                [&] { return validateContextLength(); }))
            return false;
        if (!run_phase(
                "freezeMoEExpertOverlayPlanForLoadedModel",
                [&]
                {
                    return freezeMoEExpertOverlayPlanForLoadedModel();
                }))
            return false;
        if (!run_phase(
                "initializeContinuationRequestGroup",
                [&]
                {
                    return initializeContinuationRequestGroup();
                }))
            return false;
        if (!run_phase(
                "validateMemoryPlan",
                [&] { return validateMemoryPlan(); }))
            return false;
        if (!run_phase(
                "printStartupBanner",
                [&]
                {
                    printStartupBanner(stdout);
                    return true;
                }))
            return false;
        if (!run_phase("dryRunComplete", [] { return true; }))
            return false;

        initialization_lifecycle_.publish(
            OrchestrationInitializationLifecycle::Completion::Admission);
        if (!mpi_ctx_ ||
            mpi_ctx_->rank() == mpi_coordinated_root_rank_)
        {
            LOG_INFO("[Main] --dry-run complete: preflight validation passed; no compute graph was built and no inference was started");
        }
        return true;
    }

    void OrchestrationRunner::shutdown()
    {
        if(finite_device_placement_submission_!=FiniteDevicePlacementSubmission::Inactive)
            throw std::logic_error("Runner shutdown cannot withdraw an active original device placement parent");
        std::string retired_storage_request_failure;
        const bool was_inference_ready = initialization_lifecycle_.readyForInference();
        const bool has_live_resources =
            created_native_context_source_ || plan_.runtime.native_context_preparation || runner_ || model_ctx_ || local_tp_ctx_ || local_pp_ctx_ ||
            rank_pipeline_transfer_session_ ||
            moe_overlay_inference_transaction_coordinator_ ||
            moe_overlay_inference_transaction_follower_ ||
            moe_overlay_rank_batch_transport_registry_ ||
            moe_overlay_device_controller_graph_service_ ||
            moe_overlay_device_controller_fabric_ ||
            moe_expert_overlay_physical_residency_fabric_ ||
            moe_expert_overlay_remote_projection_transport_ ||
            moe_expert_overlay_maintenance_service_ ||
            moe_expert_overlay_economy_calibration_ ||
            continuation_request_group_;
        if (!was_inference_ready && !has_live_resources)
        {
            return;
        }

        std::string reuse_seal_error;
        std::vector<ModelDeviceMemoryRetention>
            sealed_device_memory_retention;
        if (!beginModelContextReuseSealIfNeeded(&reuse_seal_error))
        {
            model_context_reuse_seal_succeeded_ = false;
            model_context_reuse_seal_error_ = reuse_seal_error.empty()
                ? "prepared model context could not enter terminal sealing"
                : std::move(reuse_seal_error);
            LOG_ERROR(
                "[OrchestrationRunner] "
                << model_context_reuse_seal_error_);
        }

        /*
         * Stop new proposals and drain transfer events while graph engines,
         * source banks, and backend contexts are still alive. A coordinated
         * root must release its remote worker loops first: SHUTDOWN makes each
         * rank enter the device-controller drain while the root does the same,
         * so the topology-wide worker remains complete until its last admitted
         * transaction reaches a durable terminal epoch.
         */
        if (was_inference_ready && mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            shutdownMPIWorkers();
        }
        else
        {
            shutdownMoEExpertOverlayResidencyMaintenance(
                MoEOverlayMaintenanceDrainIntent::TerminalContextSeal);
        }
        const auto uses_disk = [](const auto &prefix) {
            return prefix.enabled && prefix.storage_mode == PrefixCacheStorageMode::Tiered &&
                prefix.disk_budget_bytes > 0;
        };
        const bool persist_prefixes = was_inference_ready &&
            (uses_disk(config_.prefix_cache) || uses_disk(plan_.runtime.prefix_cache));
        PrefixCacheShutdown prefix_shutdown;
        if (persist_prefixes)
        {
            try
            {
                if (prefix_shutdown_lifecycle_ != PrefixShutdownLifecycle::Pending)
                    throw std::logic_error("prefix shutdown selection was reentered");
                prefix_shutdown_lifecycle_ = PrefixShutdownLifecycle::Preparing;
                // Every admitted rank joins metadata selection, including
                // expert-only ranks with no RAM payloads of their own.
                if (runner_)
                    runner_->collectPrefixCacheShutdown(prefix_shutdown);
                LOG_INFO("[PrefixCache] Selecting RAM entries for shutdown persistence");
                prefix_shutdown.prepare(mpi_ctx_.get());
                prefix_shutdown_lifecycle_ = PrefixShutdownLifecycle::Ready;
            }
            catch (const std::exception &error)
            {
                if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
                    terminateFailedMPIWorkerCommand("SHUTDOWN prefix selection", error.what());
                throw;
            }
        }

        if (runner_)
        {
            if (was_inference_ready)
            {
                try
                {
                    resetUnderlyingRunnerRequestState("shutdown");
                    // Native participants retain the device-authored terminal
                    // archive through reset. Preserve it before destroying the
                    // child; setup-time residency is not movement authority.
                    auto native_status = runner_->moeOptimizationStatus();
                    if (native_status.authority != MoEOptimizationAuthority::None)
                    {
                        terminal_moe_optimization_status_ = drainedMoEOptimizationStatus(std::move(native_status));
                        terminal_moe_optimization_movement_ledger_ = runner_->moeOptimizationMovementLedger();
                    }
                }
                catch (const RetiredStorageRequestFailure &error)
                {
                    // Every original native/adopted duty retired, but the request
                    // failed. Destroy owners safely and forbid prepared reuse.
                    retired_storage_request_failure = error.what();
                    LOG_ERROR("[OrchestrationRunner] Retired failed storage request: " << error.what());
                }
                catch (const std::exception &e)
                {
                    // A failed terminal reset leaves producer ownership unproven.
                    // Publishing reuse or dropping transfer leases would disguise
                    // that failure. Match the participant reset's fatal boundary.
                    std::fprintf(stderr, "[FATAL] OrchestrationRunner terminal request retirement failed: %s\n", e.what());
                    std::fflush(stderr);
                    std::terminate();
                }
                catch (...)
                {
                    std::fprintf(stderr, "[FATAL] OrchestrationRunner terminal request retirement raised a non-standard exception\n");
                    std::fflush(stderr);
                    std::terminate();
                }
            }
        }

        /*
         * The participant reset above drains its current MTP producer and
         * relinquishes the runner-owned transaction lease. Release the outer
         * orchestration lease while that participant's backend is still alive:
         * DeviceResidentMTPTransactionState owns the exact CUDA/HIP event
         * deleter, so carrying it across runner_.reset() would ask a retired
         * runtime generation to destroy the event during member teardown.
         */
        retireMTPRequestContinuationState();

        /*
         * Release participant-local runners directly. Every GPU runner's RAII
         * boundary retires its typed producer events before destroying captured
         * executables or arenas; RankOrchestrator declares child runners after
         * its collective contexts so reverse destruction preserves native graph
         * ownership as well. A device-wide drain here would wait on unrelated
         * streams and duplicate those exact lifecycle edges for every cell.
         */
        moe_overlay_inference_transaction_follower_.reset();
        moe_overlay_inference_transaction_coordinator_.reset();
        runner_.reset();
        if (rank_pipeline_transfer_session_)
        {
            // Every original local graph/cursor/registration is gone. Existing
            // session consensus now proves all remote mappings/FDs gone BEFORE
            // the sole host allocator drops the actual backing charge.
            try { rank_pipeline_transfer_session_->retireAfterOriginalGraphRelease(); }
            catch (...) { std::terminate(); }
            rank_pipeline_transfer_session_.reset();
        }
        if(cpu_expert_shard_ctx_) {
            // SHUTDOWN has stopped the original workers and runner destruction
            // released every stage/cache borrow. The still-live domain now
            // proves actual all-member unmap before creator backing retirement.
            if(!cpu_expert_shard_ctx_->retireOriginalCPUExpertDomain())std::terminate();
            cpu_expert_shard_ctx_.reset();
        }
        if (preparedContextSealBoundary() == ModelContextPhysicalSealBoundary::RetiredImmutableSources ||
            (moe_pipeline_prepared_plan_ && preparedContextSealBoundary() == ModelContextPhysicalSealBoundary::LiveMutableResidency))
        {
            std::string error;
            if (!sealReusableModelContextPhysicalState(&error))
            {
                model_context_reuse_seal_succeeded_ = false;
                if (model_context_reuse_seal_error_.empty())
                    model_context_reuse_seal_error_ = std::move(error);
                LOG_ERROR("[OrchestrationRunner] " << model_context_reuse_seal_error_);
            }
        }
        moe_pipeline_stage_bindings_.clear();
        model_context_pipeline_host_restored_ = false;
        if (reusable_execution_workspaces_ &&
            !reusable_execution_workspaces_->valid())
        {
            model_context_reuse_seal_succeeded_ = false;
            if (model_context_reuse_seal_error_.empty())
            {
                model_context_reuse_seal_error_ =
                    "reusable execution workspace sealing failed: " +
                    reusable_execution_workspaces_->diagnostic();
            }
        }
        continuation_request_group_.reset();
        moe_overlay_rank_batch_transport_registry_.reset();
        moe_overlay_device_controller_graph_service_.reset();
        moe_overlay_device_controller_fabric_.reset();
        moe_overlay_device_controller_topology_.reset();
        moe_expert_overlay_participant_residency_.reset();
        moe_expert_overlay_residency_authority_.reset();
        moe_expert_overlay_decode_histogram_.reset();
        moe_expert_overlay_interference_probe_.reset();
        moe_overlay_prefill_interference_owner_ =
            MoEOverlayPrefillInterferenceOwner::OrchestrationScope;
        llaminar::v2::kernels::KernelFactory::clearCache();
        local_pp_ctx_.reset();
        local_tp_ctx_.reset();

        std::exception_ptr prefix_persistence_failure;
        if (persist_prefixes)
        {
            // This rank's graphs and collective domains have retired. The
            // selected RAM may still own pinned backing from its original GPU
            // context, so finish local writes before native context retirement.
            try
            {
                const auto result = prefix_shutdown.persist();
                prefix_shutdown_lifecycle_ = PrefixShutdownLifecycle::Complete;
                LOG_INFO("[PrefixCache] Shutdown persisted " << result.selected << "/" << result.candidates
                    << " RAM entries: " << result.written << " payload writes ("
                    << result.payload_bytes_written << " bytes), " << result.reused << " existing backing reuses");
            }
            catch (const std::exception &error)
            {
                if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
                    terminateFailedMPIWorkerCommand("SHUTDOWN prefix persistence", error.what());
                prefix_persistence_failure = std::current_exception();
            }
        }

        /*
         * Every graph, stream-owning runner, overlay transport, migration bank,
         * and local collective context is now retired. The ModelContext remains
         * alive for prepared-weight reuse, so this edge trims only unused
         * runtime cache backing. TransferEngine returns a typed before/after
         * receipt; it never resets the device or touches another model's live
         * allocations. A failed receipt invalidates reuse rather than letting
         * the next cell discover the lifecycle defect as an unrelated OOM.
         */
        for (const DeviceId device : rankLocalGPUExecutionDevices(
                 plan_, config_.moe_routed_expert_plan, mpi_ctx_))
        {
            try
            {
                (void)TransferEngine::instance().reclaimDeviceMemory(
                    DeviceMemoryReclamationRequest::
                        retiredExecutionTopology(device));
            }
            catch (const std::exception &error)
            {
                model_context_reuse_seal_succeeded_ = false;
                if (model_context_reuse_seal_error_.empty())
                {
                    model_context_reuse_seal_error_ =
                        "rank-local execution cache reclamation failed for " +
                        device.toString() + ": " + error.what();
                }
                LOG_ERROR(
                    "[OrchestrationRunner] "
                    << model_context_reuse_seal_error_);
            }
        }

        /*
         * All runner-only owners are gone. Snapshot the allocations still held
         * by the model context and sealed workspace now; doing this at export
         * time would include Dynamic shadow pools that teardown legitimately
         * retires. An exported contract publishes this BOM with Reusable. An
         * ordinary runner is instead the exclusive final model owner: it binds
         * the same BOM to TransferEngine tickets before releasing the owners,
         * then retires the native runtime generations before shutdown returns.
         */
        const bool reusable_workspace_complete =
            reusable_execution_workspaces_ &&
            reusable_execution_workspaces_->valid();
        const bool exports_reusable_model_context =
            model_context_reuse_authority_ != nullptr;
        const bool retires_exclusive_model_context =
            was_inference_ready && !exports_reusable_model_context &&
            model_ctx_ && reusable_workspace_complete &&
            model_ctx_.use_count() == 1u &&
            reusable_execution_workspaces_.use_count() == 1u;
        // A failed fresh initialization has no ready model-retention contract.
        // Keep its exact original source alive until every partial owner below
        // is gone. Successful exclusive teardown instead borrows the final
        // model owner's source, including storage-only native participants.
        const bool exclusive_failed_source = !was_inference_ready &&
            !exports_reusable_model_context && created_native_context_source_ &&
            (!model_ctx_ || model_ctx_.use_count()==1u) &&
            (!reusable_execution_workspaces_ || reusable_execution_workspaces_.use_count()==1u);
        auto final_native_source = retires_exclusive_model_context
            ? model_ctx_->nativeContextPreparation()
            : (exclusive_failed_source ? created_native_context_source_ : nullptr);
        const bool needs_retention_seal =
            exports_reusable_model_context ||
            retires_exclusive_model_context;
        if (needs_retention_seal &&
            model_context_reuse_seal_succeeded_)
        {
            std::string retention_error;
            if (!sealModelContextDeviceMemoryRetention(
                    &sealed_device_memory_retention,
                    &retention_error))
            {
                model_context_reuse_seal_succeeded_ = false;
                model_context_reuse_seal_error_ =
                    retention_error.empty()
                        ? "prepared model context allocation retention sealing failed"
                        : std::move(retention_error);
                LOG_ERROR(
                    "[OrchestrationRunner] "
                    << model_context_reuse_seal_error_);
            }
        }

        std::optional<PendingExclusiveModelRetirement>
            pending_exclusive_model_retirement;
        if (retires_exclusive_model_context &&
            model_context_reuse_seal_succeeded_)
        {
            try
            {
                pending_exclusive_model_retirement.emplace(
                    PendingExclusiveModelRetirement::begin(
                        sealed_device_memory_retention));
            }
            catch (const std::exception &error)
            {
                model_context_reuse_seal_succeeded_ = false;
                model_context_reuse_seal_error_ =
                    "exclusive final model retirement could not capture its allocation BOM: " +
                    std::string(error.what());
                LOG_ERROR(
                    "[OrchestrationRunner] "
                    << model_context_reuse_seal_error_);
                pending_exclusive_model_retirement.reset();
            }
        }
        model_ctx_.reset();

        if (exports_reusable_model_context)
        {
            if (!retired_storage_request_failure.empty()) {
                model_context_reuse_seal_succeeded_ = false;
                model_context_reuse_seal_error_ = retired_storage_request_failure;
            }
            if (model_context_reuse_seal_succeeded_)
            {
                std::string publication_error;
                if (!model_context_reuse_authority_->publishReusable(
                        std::move(sealed_device_memory_retention),
                        &publication_error))
                {
                    model_context_reuse_seal_succeeded_ = false;
                    model_context_reuse_seal_error_ =
                        publication_error.empty()
                            ? "prepared model context seal could not publish its reusable state"
                            : std::move(publication_error);
                }
            }
            if (!model_context_reuse_seal_succeeded_)
            {
                model_context_reuse_authority_->invalidate(
                    model_context_reuse_seal_error_);
            }
        }
        else if (retires_exclusive_model_context)
        {
            /* The reusable backing is the second and final allocation owner
             * named by every ticket. Release it only after the ModelContext so
             * prepared-weight destructors can still consult their workspace
             * registry while tearing down device representations. */
            reusable_execution_workspaces_.reset();
            if (model_context_reuse_seal_succeeded_)
            {
                try
                {
                    if (!pending_exclusive_model_retirement)
                    {
                        throw std::logic_error(
                            "exclusive final model retirement lost its typed pending plan");
                    }
                    (void)pending_exclusive_model_retirement->complete(final_native_source);
                }
                catch (const std::exception &error)
                {
                    model_context_reuse_seal_succeeded_ = false;
                    model_context_reuse_seal_error_ =
                        "exclusive final model retirement was incomplete: " +
                        std::string(error.what());
                    LOG_ERROR(
                        "[OrchestrationRunner] "
                        << model_context_reuse_seal_error_);
                }
            }
        }

        if(final_native_source &&
            (exclusive_failed_source || model_context_reuse_seal_succeeded_)){
            // All graphs, weights and execution buffers must be gone before
            // native-only zero-allocation validation. Existing positive model
            // BOM retirement above remains authoritative; this final source
            // pass covers residual storage-only births as well as failed setup.
            reusable_execution_workspaces_.reset();
            final_native_source->retireReleasedContexts();
            if(plan_.runtime.native_context_preparation==final_native_source)
                plan_.runtime.native_context_preparation.reset();
            if(created_native_context_source_==final_native_source)
                created_native_context_source_.reset();
        }

        // Quota/planner failure may precede ModelContext publication entirely.
        // Retire only this original Preparing source's installed births after
        // all possible runner/model owners have gone. The source-owned final
        // cleanup above also handles admitted fresh failures. Borrowed/exported
        // admitted contexts retain their existing model-retirement/reuse path.
        if (plan_.runtime.native_context_preparation &&
            plan_.runtime.native_context_preparation->memory()->phase() ==
                PhysicalMemoryAuthorityPhase::Preparing)
        {
            plan_.runtime.native_context_preparation->retireUnadmitted();
            if(created_native_context_source_==plan_.runtime.native_context_preparation)
                created_native_context_source_.reset();
            plan_.runtime.native_context_preparation.reset();
        }

        // Exported/shared ModelContext and reuse-contract owners keep the SAME
        // source/PMA. Relinquish this retired runner's plan borrow as well as its
        // creation handle: otherwise a destructor's second shutdown mistakes
        // that borrow for live runner state and tries to seal a Reusable (or
        // successor-owned) authority again. Actual source retirement above is
        // unchanged; dropping a borrow never retires the external model owner.
        plan_.runtime.native_context_preparation.reset();
        created_native_context_source_.reset();
        mtp_depth_policy_scope_geometry_ = {};
        mtp_depth_policy_scope_generic_reason_ = "model_free";
        // Failed request data is never reusable. Exclusive physical retirement
        // above still used its original positive BOM and native-zero proof.
        if (!retired_storage_request_failure.empty()) {
            model_context_reuse_seal_succeeded_ = false;
            model_context_reuse_seal_error_ = retired_storage_request_failure;
        }
        initialization_lifecycle_.retire();

        // Include terminal durable-cache publication in the saved native proof.
        PerfStatsCollector::flushFromEnv();
        LOG_DEBUG("OrchestrationRunner shut down");
        if (prefix_persistence_failure)
            std::rethrow_exception(prefix_persistence_failure);
    }

    // =========================================================================
    // Initialization Helpers
    // =========================================================================

    bool OrchestrationRunner::initializeMPI()
    {
        // An explicit admission already selected the participant namespace.
        // Local execution on a selected rank is not permission to replace it
        // with SELF, nor does cross-rank TP imply the launcher's entire WORLD.
        if (mpi_ctx_)
        {
            int actual_rank = -1, actual_size = 0;
            if (mpi_ctx_->communicator() == MPI_COMM_NULL ||
                MPI_Comm_rank(mpi_ctx_->communicator(), &actual_rank) != MPI_SUCCESS ||
                MPI_Comm_size(mpi_ctx_->communicator(), &actual_size) != MPI_SUCCESS ||
                actual_rank != mpi_ctx_->rank() || actual_size != mpi_ctx_->world_size())
                return setError("Admitted runner context disagrees with MPI membership");
            return true;
        }
        if (!requiresMultiRankMPI(config_))
        {
            /*
             * Participant-local planning still performs typed topology queries.
             * MPI_COMM_SELF is the exact one-process authority for those calls;
             * MPI_COMM_NULL is not a local communicator and becomes invalid as
             * soon as model-driven normalization asks for topology metadata.
             * Cross-rank graph edges retain their explicitly owned domain/world
             * contexts and never inherit this participant-local communicator.
             */
            LOG_DEBUG("Creating single-rank MPI context for local execution");
            mpi_ctx_ = MPIContextFactory::self();
            return true;
        }

        // Create or reuse MPI context using factory
        mpi_ctx_ = MPIContextFactory::global();
        if (!mpi_ctx_)
        {
            return setError("Failed to get MPI context");
        }

        LOG_DEBUG("MPI initialized: rank " << mpi_ctx_->rank()
                                           << " of " << mpi_ctx_->world_size());

        return true;
    }

    void OrchestrationRunner::installResolvedExecutionPlan(
        const ResolvedRankOrchestration &resolved)
    {
        // Prepare all owning values before publishing any part of the result.
        // Deriving this later from config_ would encode capacity-frozen placements
        // rather than the user's exact requested reuse/migration authority.
        auto config = resolved.config();
        auto plan = resolved.rankPlan();
        auto topology = resolved.pipelineStages().empty()
            ? std::shared_ptr<const ResolvedRankOrchestration>{}
            : std::make_shared<const ResolvedRankOrchestration>(resolved);
        auto identity = topology
            ? pipelineRoutedWeightAuthorityRequestIdentity(resolved)
            : routedWeightAuthorityRequestIdentity(config.moe_routed_expert_plan, config);
        config_ = std::move(config);
        plan_ = std::move(plan);
        moe_pipeline_topology_ = std::move(topology);
        requested_routed_weight_authority_identity_ = std::move(identity);
    }

    bool OrchestrationRunner::buildExecutionPlan()
    {
        if (plan_built_)
            return true;
        if (!plan_builder_)
            return setError("No plan builder available");

        cluster_inventory_ = gatherClusterInventory();
        // Metadata publication must complete collectively before any local
        // placement failure can enter the outer initialization phase vote.
        try
        {
            planning_model_metadata_.emplace(exchangePlanningModelMetadata(mpi_ctx_, [&] {
                return readPlanningModelMetadata(config_.model_path, config_.shared_predictor);
            }));
        }
        catch (const std::exception &error)
        {
            return setError("Failed to publish planning metadata for " + config_.model_path + ": " + error.what());
        }

        try
        {
            const int rank = mpi_ctx_->rank();
            const auto resolved = ResolvedRankOrchestration::resolve(
                config_, *planning_model_metadata_, cluster_inventory_, *plan_builder_, rank,
                ordinary_terminal_projection_setup_);
            if (!resolved.pipelineStages().empty())
            {
                // Movement storage is not the authority locus. Singleton
                // domains retain their host/device owner and Dynamic intent;
                // a complete no-peer residency proof is sealed after quotas.
                // Movable domains still require their real native controller.
                for (const auto &stage : resolved.pipelineStages())
                {
                    const auto storage = MoEOverlayCapacityAdmission::selectMigrationStorage(
                        *stage.config().moe_routed_expert_plan, config_.moe_rebalance.mode);
                    if (config_.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
                        storage != MoEOverlayMigrationStorageKind::NoPeer &&
                        storage != MoEOverlayMigrationStorageKind::StationaryTensorShard &&
                        storage != MoEOverlayMigrationStorageKind::DeviceTransferDirectory &&
                        !(storage == MoEOverlayMigrationStorageKind::PhysicalResidencyFabric &&
                          std::all_of(stage.config().moe_routed_expert_plan->domains.begin(),
                              stage.config().moe_routed_expert_plan->domains.end(), [](const auto &domain) {
                                  return std::all_of(domain.participants.begin(), domain.participants.end(),
                                      [](const auto &participant) { return participant.toLocalDeviceId().is_cpu(); });
                              })))
                        return setError("Movable local MoE pipeline stages require an admitted native maintenance domain");
                }
            }
            installResolvedExecutionPlan(resolved);
            mpi_coordinated_root_rank_ = resolved.overlayExecution()
                ? resolved.overlayExecution()->continuation_root_rank : 0;
            if (mpi_coordinated_root_rank_ < 0 || mpi_coordinated_root_rank_ >= mpi_ctx_->world_size())
                return setError("Resolved orchestration has an invalid continuation authority rank");

            if (rank == 0 && resolved.overlayExecution())
            {
                // Diagnostics consume the sealed role result; they cannot
                // rebind placement or create another authority in the frontend.
                if (debugEnv().moe_expert_overlay.trace)
                    LOG_INFO("[MoEExpertOverlayExecutionPlan]\n" << resolved.overlayExecution()->diagnostics());
                if (config_.explain_placement)
                    std::cout << "\n=== MoE Expert Overlay Role Plan ===\n"
                              << resolved.overlayExecution()->diagnostics() << std::endl;
            }

            // Pure compilation returns an origin; observability stays here and
            // cannot select the authority or trigger another normalization.
            if (resolved.overlayOrigin() == MoEExpertOverlayAuthorityPlanDisposition::SynthesizedLocalTP)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency", "implicit_single_tier_authority_plan", 1.0,
                    "model_setup", "priority_0",
                    {{"participants", std::to_string(config_.moe_routed_expert_plan->domains.front().participants.size())},
                     {"world_rank", std::to_string(rank)},
                     {"topology", "single-domain"}, {"authority", "expert-overlay-rcu"}});
            }
            LOG_INFO("[MTPDefaults] continuation_profile="
                     << mtpDepthDefaultsProfileToString(config_.mtp.depth_defaults_profile)
                     << " zero_accept_demote=" << resolveMTPZeroAcceptDemotionRate(config_.mtp)
                     << " source=" << (config_.mtp.depth_policy.demote_zero_accept_rate ? "explicit" : "hardware_default"));
        }
        catch (const std::exception &error)
        {
            return setError(std::string("Failed to resolve rank orchestration: ") + error.what());
        }
        plan_built_ = true;
        return true;
    }

    const PlanningModelMetadata &OrchestrationRunner::planningModelMetadata()
    {
        if (!planning_model_metadata_)
        {
            // Explicit pre-built plans bypass buildExecutionPlan and arrive
            // with a loaded model authority. Project that same loader once;
            // never reopen its file or start another discovery exchange here.
            if (!model_ctx_)
                throw std::logic_error("Planning metadata requires root admission or an explicitly preloaded model");
            planning_model_metadata_.emplace(
                PlanningModelMetadata::fromLoader(model_ctx_->concreteLoader()));
        }
        return *planning_model_metadata_;
    }

    ClusterInventory OrchestrationRunner::gatherClusterInventory()
    {
        // This admission-local snapshot receives the existing runtime-stable
        // free-capacity observations. Never mutate the immutable discovery owner
        // used for physical identity and automatic measurement authentication.
        return *llaminar2::gatherClusterInventory(mpi_ctx_, config_.hostfile);
    }

    bool OrchestrationRunner::setupLocalTPContext()
    {
        // A global graph constructs each stage's exact domain context itself.
        // Do not allocate a duplicate rank-wide collective before that phase.
        if (requiresNamedDomainGlobalGraph(config_))
            return true;
        // Check if LOCAL TP is configured
        if (plan_.local_tp_devices.empty())
        {
            LOG_DEBUG("No LOCAL TP devices configured");
            return true;
        }

        /*
         * The overlay rank-role resolver has already reduced non-root ranks to
         * endpoint-only plans with no local_tp_devices. If devices remain here,
         * they are the continuation root's real dense LocalTP graph and must
         * receive the same collective context as any other production LocalTP
         * model. Overlay sparse-expert endpoints remain independent inside that
         * graph; they do not justify dropping its dense TP authority.
         *
         * LOCAL PP is different: each nested stage owns its own TP context.
         */
        if (plan_.usesLocalPP())
        {
            LOG_DEBUG("LOCAL PP active — TP will be handled per-stage by nested MDOs");
            return true;
        }

        // Create LOCAL TP context using factory function
        local_tp_ctx_ = createLocalTPContext(
            plan_.local_tp_devices,
            plan_.local_tp_weights,
            plan_.local_tp_backend,
            config_.moe_routed_expert_plan && config_.moe_routed_expert_plan->usesGFX906MI50Kernels()
                ? RankOrderedQuotePolicy::MeasuredDomain : RankOrderedQuotePolicy::ExplicitOnly);
        if (!local_tp_ctx_)
        {
            return setError("Failed to create LOCAL TP context");
        }

        plan_.runtime.rank_ordered_resources = local_tp_ctx_->rankOrderedResources();
        LOG_DEBUG("LOCAL TP context created with " << plan_.local_tp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::setupLocalPPContext()
    {
        if (requiresNamedDomainGlobalGraph(config_))
            return true;
        // Check if LOCAL PP is configured
        if (plan_.local_pp_devices.size() <= 1)
        {
            LOG_DEBUG("No LOCAL PP devices configured (or only single device)");
            return true;
        }

        // Build LocalPPConfig from execution plan
        LocalPPConfig pp_config;
        pp_config.stage_devices = plan_.local_pp_devices;
        pp_config.layer_boundaries = plan_.local_pp_layer_boundaries;

        // Validate configuration
        if (!pp_config.isValid())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // Create LOCAL PP context using factory function
        local_pp_ctx_ = createLocalPPContext(pp_config);
        if (!local_pp_ctx_)
        {
            return setError("Failed to create LOCAL PP context");
        }

        LOG_DEBUG("LOCAL PP context created with " << pp_config.numStages()
                                                   << " stages on " << plan_.local_pp_devices.size() << " devices");
        return true;
    }

    bool OrchestrationRunner::loadWeights(bool prepopulate_page_cache_enabled)
    {
        // Get model path from config
        std::string model_path = config_.model_path;

        // Skip weight loading if no model path (for testing)
        if (model_path.empty())
        {
            if (model_ctx_)
            {
                return setError(
                    "A preloaded ModelContext requires an explicit model_path");
            }
            LOG_DEBUG("No model path specified, skipping weight loading");
            return true;
        }

        // Create ModelContextConfig from the execution plan
        // This automatically configures:
        // - Layer range (first_layer, last_layer) for PP
        // - Global weight flags (has_embedding, has_lm_head) for PP
        // - Shard info (shard_index, total_shards, work_fraction) for TP
        // - Appropriate strategy (REPLICATED, SHARDED, or layer-partitioned)
        ModelContextConfig weight_config = ModelContextConfig::fromExecutionPlan(plan_);
        weight_config.mpi_ctx = mpi_ctx_;
        weight_config.weight_precision = WeightPrecision::NATIVE;
        weight_config.use_mmap = config_.use_mmap;
        weight_config.shared_predictor = config_.shared_predictor;
        weight_config.predictor_materialization = retainsMTPGraphCapacity(config_.mtp)
            ? SharedPredictorMaterialization::Active : SharedPredictorMaterialization::Inactive;

        /*
         * A process-resident parity campaign may retain the immutable model and
         * PreparedWeightStore while rebuilding every runner-owned graph, arena,
         * stream, controller, and request state.  Prove that the retained
         * authority exactly matches this execution plan before consuming it.
         * A mismatch is fatal: silently reloading would make reuse evidence lie.
         */
        if (model_ctx_)
        {
            if (model_ctx_->path() != model_path)
            {
                return setError(
                    "Preloaded ModelContext path does not match model_path: " +
                    model_ctx_->path() + " != " + model_path);
            }

            try {
                model_ctx_->concreteLoader().requireSharedPredictorSource(config_.shared_predictor,
                    weight_config.predictor_materialization);
            } catch(const std::invalid_argument& error) {
                return setError(error.what());
            }
            const auto weight_manager = model_ctx_->concreteWeightManager();
            if (!weight_manager)
            {
                return setError(
                    "Preloaded ModelContext has no concrete WeightManager");
            }
            if ((config_.moe_routed_expert_plan || moe_pipeline_topology_) &&
                !retained_prepared_weight_plan_)
            {
                return setError(
                    "A preloaded ModelContext cannot enter ExpertOverlay "
                    "without an exact prepared-weight reuse contract");
            }
            if (retained_prepared_weight_plan_)
            {
                const bool retained_has_routed_authority =
                    !retained_routed_weight_authority_identity_.empty();
                const bool current_has_routed_authority =
                    !requested_routed_weight_authority_identity_.empty();
                if (retained_has_routed_authority !=
                    static_cast<bool>(
                        retained_prepared_routed_weight_plan_ || moe_pipeline_prepared_plan_))
                {
                    return setError(
                        "Retained ExpertOverlay identity and model-frozen "
                        "physical plan presence disagree");
                }
                if (retained_has_routed_authority !=
                    current_has_routed_authority)
                {
                    return setError(
                        "Retained prepared weights and the current runner disagree "
                        "on whether an ExpertOverlay weight authority exists");
                }
                if (current_has_routed_authority &&
                    retained_routed_weight_authority_identity_ !=
                        requested_routed_weight_authority_identity_)
                {
                    return setError(
                        "Retained prepared weights belong to a different exact "
                        "ExpertOverlay participant/tier/placement authority");
                }
                if (const auto mismatch = retainedPreparedWeightPlanMismatch(
                        *retained_prepared_weight_plan_, plan_))
                {
                    return setError(*mismatch);
                }
            }
            WeightDistributionStrategy expected_strategy =
                weight_config.strategy;
            if (weight_config.isLayerPartitioned() &&
                expected_strategy == WeightDistributionStrategy::REPLICATED)
            {
                expected_strategy =
                    WeightDistributionStrategy::LAYER_PARTITIONED;
            }
            if (weight_manager->strategy() != expected_strategy)
            {
                return setError(
                    "Preloaded ModelContext distribution strategy does not match "
                    "the execution plan");
            }
            if (weight_manager->hasEmbedding() != weight_config.has_embedding ||
                weight_manager->hasLMHead() != weight_config.has_lm_head)
            {
                return setError(
                    "Preloaded ModelContext global-weight ownership does not match "
                    "the execution plan");
            }
            if (expected_strategy ==
                WeightDistributionStrategy::LAYER_PARTITIONED)
            {
                if (!weight_manager->hasLayerRange())
                {
                    return setError(
                        "Preloaded ModelContext lacks the execution plan's layer range");
                }
                const auto [loaded_first, loaded_last_exclusive] =
                    weight_manager->layerRange();
                const int expected_last_exclusive =
                    weight_config.last_layer < 0
                        ? -1
                        : weight_config.last_layer + 1;
                if (loaded_first != weight_config.first_layer ||
                    loaded_last_exclusive != expected_last_exclusive)
                {
                    return setError(
                        "Preloaded ModelContext layer range does not match the "
                        "execution plan");
                }
            }

            /*
             * Only this successful comparison permits memory preflight to count
             * prepared weights as already resident. Store population and the
             * lifecycle completion gates are checked separately per device.
             */
            retained_prepared_weight_plan_validated_ =
                retained_prepared_weight_plan_.has_value();
            if(retained_prepared_weight_plan_validated_)
            {
                // Every reuse path consumes the same complete immutable
                // admission before graph/memory factories run. Keep active
                // request policy fresh; never replan the retained physical bytes.
                restoreRetainedAdmittedRuntime(plan_.runtime,retained_prepared_weight_plan_->runtime,plan_.rank);
            }

            /*
             * A preloaded context can either be a freshly parsed model handed
             * to its first runner or a process-campaign authority whose packed
             * device weights already survived a prior runner.  Publish those
             * cases separately: setup-time economy tests must prove the latter
             * instead of inferring it from a caller-owned cache-hit boolean.
             * PreparedWeightStore is model-owned, so observing it here neither
             * creates a host mirror nor changes its lifetime.
             */
            const auto prepared_store =
                weight_manager->preparedWeightStoreIfInitialized();
            const auto prepared_devices =
                rankLocalPreparedWeightAuthorityDevices(
                    plan_,
                    config_.moe_routed_expert_plan,
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_);
            size_t prepared_entry_count = 0u;
            if (prepared_store)
            {
                for (const DeviceId device : prepared_devices)
                {
                    prepared_entry_count +=
                        weight_manager->preparedRecordCountForDevice(device);
                }
            }
            retained_prepared_entry_count_ =
                imported_model_context_reuse_authority_
                    ? prepared_entry_count
                    : 0u;
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_model_context_uses",
                1.0,
                "load",
                plan_.primary_device.toString(),
                {{"prepared_entries",
                  std::to_string(prepared_entry_count)}});
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_prepared_weight_store_reuses",
                prepared_entry_count > 0u ? 1.0 : 0.0,
                "load",
                plan_.primary_device.toString(),
                {{"prepared_entries",
                  std::to_string(prepared_entry_count)}});
            PerfStatsCollector::addCounter(
                "weight_loading",
                "preloaded_prepared_weight_entries",
                static_cast<double>(prepared_entry_count),
                "load",
                plan_.primary_device.toString());

            tokenizer_ = createTokenizer(model_ctx_);
            if (!tokenizer_)
            {
                LOG_WARN("Failed to create tokenizer from preloaded model context");
            }
            LOG_INFO("Reusing preloaded ModelContext for production runner: "
                     << model_path);
            return true;
        }

        /*
         * Publish payload lifetime independently from the rank's synthetic
         * primary device. An auxiliary ExpertOverlay rank is represented by a
         * CPU primary in RankExecutionPlan because it does not own the dense
         * continuation graph, but it may own CPU and GPU expert endpoints.
         * Its GGUF mapping is only a source for exact expert selections and
         * must never be mistaken for the dense, inference-resident CPU model.
         */
        const auto overlay_execution_for_load =
            resolveOverlayExecutionPlanForRunner(
                config_.moe_routed_expert_plan,
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_);
        const bool sparse_overlay_payload =
            overlay_execution_for_load &&
            !overlay_execution_for_load->currentRankPlan().loads_root_weights;
        if (sparse_overlay_payload)
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::SparseSelection;
        }
        else if (plan_.primary_device.device_type != DeviceType::CPU)
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::DeviceStaging;
        }
        else
        {
            weight_config.payload_access_pattern =
                ModelPayloadAccessPattern::DenseCpuResident;
        }

        // Resolve from authenticated planning metadata before either local or
        // node-leader warmup. Low-level ModelLoader applies the same semantic
        // resolver after parsing, so direct and orchestrated callers cannot
        // accidentally pre-read a separately admitted immutable row table.
        for (const auto &tensor : planningModelMetadata().memoryProfile().tensors)
            weight_config.payload_access_pattern = resolveModelPayloadAccessPattern(
                weight_config.payload_access_pattern, inferWeightRole(tensor.name));

        // Multi-rank page cache pre-population:
        // In multi-rank mode, each rank independently mmaps and first-touches the
        // same file. Without coordination, this creates N concurrent page fault
        // streams that destroy disk readahead and cause ~60 MB/s throughput
        // instead of ~1 GB/s. Fix: node leaders read the file sequentially to warm
        // the page cache, then all ranks on that node skip POSIX_FADV_DONTNEED so
        // the OMP first-touch loop faults from cache (memory speed) instead of disk.
        //
        // Multi-node scaling: each physical machine has its own page cache, so we
        // use the node-leader (lowest local rank) on each node to prepopulate
        // independently. An intra-node barrier ensures same-node ranks wait only
        // for their own leader, not a global rank-0.
        const bool is_multi_rank = mpi_ctx_ && mpi_ctx_->world_size() > 1;
        bool node_has_dense_cpu_payload =
            weight_config.payload_access_pattern ==
            ModelPayloadAccessPattern::DenseCpuResident;
        if (is_multi_rank)
        {
            int local_dense_cpu_payload =
                node_has_dense_cpu_payload ? 1 : 0;
            int any_dense_cpu_payload = local_dense_cpu_payload;
            MPI_Comm cache_comm = mpi_ctx_->intra_node_comm();
            if (cache_comm == MPI_COMM_NULL)
                cache_comm = mpi_ctx_->communicator();
            if (cache_comm != MPI_COMM_NULL)
            {
                MPI_Allreduce(
                    &local_dense_cpu_payload,
                    &any_dense_cpu_payload,
                    1,
                    MPI_INT,
                    MPI_MAX,
                    cache_comm);
            }
            node_has_dense_cpu_payload = any_dense_cpu_payload != 0;
        }
        const auto prepopulate_page_cache = [&](const std::string &reason)
        {
            LOG_DEBUG(reason << " pre-populating page cache for mmap load...");
            uintmax_t model_file_bytes = 0;
            try
            {
                model_file_bytes = std::filesystem::file_size(model_path);
            }
            catch (const std::exception &)
            {
                model_file_bytes = 0;
            }
            auto start = std::chrono::steady_clock::now();
            const bool ok = MmapRegion::prepopulatePageCache(model_path);
            auto elapsed = std::chrono::steady_clock::now() - start;
            const double elapsed_ms = std::chrono::duration<double, std::milli>(elapsed).count();
            const double elapsed_s = elapsed_ms / 1000.0;
            const double mb = static_cast<double>(model_file_bytes) / (1024.0 * 1024.0);
            const double mbps = (model_file_bytes > 0 && elapsed_s > 0.0) ? (mb / elapsed_s) : 0.0;
            WeightLoadingProfiler::addDetail("weights.mmap_page_cache_prepopulate", elapsed_ms);
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_success",
                                           ok ? 1.0 : 0.0, "load", {},
                                           {{"reason", reason}});
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_bytes",
                                           static_cast<double>(model_file_bytes), "load", {},
                                           {{"reason", reason},
                                            {"success", perfBool(ok)}});
            PerfStatsCollector::addCounter("weight_loading", "mmap_page_cache_prepopulate_mbps",
                                           mbps, "load", {},
                                           {{"reason", reason},
                                            {"success", perfBool(ok)}});
            if (!ok)
            {
                LOG_WARN(reason << " page-cache prepopulation failed; continuing with mmap demand faults");
            }
            else if (model_file_bytes > 0 && mbps > 0.0 && mbps < 256.0)
            {
                LOG_INFO(reason << " page-cache prepopulation was slow: "
                                << std::fixed << std::setprecision(0) << mbps
                                << " MB/s for " << std::fixed << std::setprecision(0)
                                << mb << " MB. Subsequent counters can distinguish disk/cache speed from GPU staging.");
            }
            return ok;
        };

        const bool should_prepopulate_page_cache =
            prepopulate_page_cache_enabled &&
            config_.use_mmap &&
            node_has_dense_cpu_payload;

        if (should_prepopulate_page_cache && !is_multi_rank)
        {
            // CPU decode reads weights from the mapping for the lifetime of the
            // model, so a NUMA-local prewarm remains useful. GPU-only startup is
            // deliberately demand-paged: pre-reading the entire GGUF defeats the
            // bounded staging contract on low-RAM/high-VRAM hosts.
            prepopulate_page_cache("Single-rank CPU");
            weight_config.skip_mmap_cache_eviction = true;
        }
        else if (should_prepopulate_page_cache && is_multi_rank)
        {
            const auto *topo = mpi_ctx_->topology();
            if (topo)
            {
                // Per-node prepopulation: each node leader warms its own page cache
                if (topo->is_node_leader())
                {
                    std::ostringstream reason;
                    reason << "Node leader (rank " << mpi_ctx_->rank()
                           << ", node " << topo->placement().node_id << ")";
                    prepopulate_page_cache(reason.str());
                }
                // Intra-node barrier: same-node ranks wait for their node leader only.
                // Ranks on other nodes proceed independently with their own leader.
                MPI_Comm intra = mpi_ctx_->intra_node_comm();
                if (intra != MPI_COMM_NULL)
                    MPI_Barrier(intra);
                else
                    MPI_Barrier(mpi_ctx_->communicator());
            }
            else
            {
                // Fallback: no topology available (mock or non-standard context)
                if (mpi_ctx_->rank() == 0)
                {
                    prepopulate_page_cache("Rank 0 fallback");
                }
                MPI_Barrier(mpi_ctx_->communicator());
            }
            weight_config.skip_mmap_cache_eviction = true;
        }
        else if (prepopulate_page_cache_enabled &&
                 config_.use_mmap &&
                 weight_config.payload_access_pattern ==
                     ModelPayloadAccessPattern::DeviceStaging)
        {
            LOG_DEBUG("GPU-only weight loading: skipping whole-file page-cache prepopulation; "
                      "GGUF pages will be consumed and discarded incrementally");
        }
        else if (prepopulate_page_cache_enabled &&
                 config_.use_mmap &&
                 weight_config.payload_access_pattern ==
                     ModelPayloadAccessPattern::SparseSelection)
        {
            LOG_DEBUG(
                "Selected-source weight loading: skipping whole-file "
                "page-cache prepopulation; only explicitly selected payloads will "
                "fault source pages");
            PerfStatsCollector::addCounter(
                "weight_loading",
                "mmap_sparse_selection_prepopulate_bypass",
                1.0,
                "load",
                {},
                {{"rank", std::to_string(mpi_ctx_ ? mpi_ctx_->rank() : 0)}});
        }

        // Validate config
        auto errors = weight_config.validate();
        if (!errors.empty())
        {
            std::ostringstream oss;
            oss << "Invalid ModelContextConfig from execution plan:\n";
            for (const auto &err : errors)
            {
                oss << "  - " << err << "\n";
            }
            return setError(oss.str());
        }

        LOG_DEBUG("Weight loading config: " << weight_config.toString());

        // Create ModelContext using the unified config-based factory method
        // Use NATIVE weight precision to preserve quantization (Q4_0, Q8_0, etc.)
        // for efficient GPU kernels rather than dequantizing to FP32
        {
            ScopedWeightLoadTimer timer(WeightLoadPhase::GGUF_PARSE);
            model_ctx_ = ModelContext::create(model_path, weight_config);
        }

        if (!model_ctx_)
        {
            return setError("Failed to create ModelContext for: " + model_path);
        }

        // Create tokenizer from model context
        tokenizer_ = createTokenizer(model_ctx_);
        if (!tokenizer_)
        {
            LOG_WARN("Failed to create tokenizer from model context");
        }

        LOG_DEBUG("Model context created from: " << model_path
                                                 << " (layers " << weight_config.first_layer << "-" << weight_config.last_layer
                                                 << ", embedding=" << weight_config.has_embedding
                                                 << ", lm_head=" << weight_config.has_lm_head << ")");

        return true;
    }

    bool OrchestrationRunner::awaitMmapReclaimBeforeRuntimeAllocation()
    {
        if (!model_ctx_)
            return true;

        const auto weight_manager = model_ctx_->weightManager();
        if (!weight_manager)
        {
            return setError(
                "Runtime allocation admission requires a model WeightManager");
        }

        try
        {
            const auto completion =
                weight_manager->awaitMmapReclaimBeforeHostAllocation();
            if (completion.state != MmapReclaimLifecycle::State::Idle &&
                completion.state != MmapReclaimLifecycle::State::Complete)
            {
                return setError(
                    "Runtime allocation admission observed a non-terminal "
                    "post-prefill host reclaim state");
            }

            PerfStatsCollector::addCounter(
                "host_memory",
                "reclaim_allocation_barriers",
                1.0,
                "initialization",
                plan_.primary_device.toString(),
                {{"state",
                  completion.state == MmapReclaimLifecycle::State::Complete
                      ? "complete"
                      : "idle"},
                 {"advised_bytes",
                  std::to_string(completion.advised_bytes)}});
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Post-prefill host reclamation failed before runtime "
                    "allocation: ") +
                error.what());
        }
        catch (...)
        {
            return setError(
                "Post-prefill host reclamation failed before runtime "
                "allocation with a non-standard exception");
        }
    }

    bool OrchestrationRunner::validateTPPPConfiguration()
    {
        // Skip validation if no model loaded (testing mode)
        if (!model_ctx_)
        {
            LOG_DEBUG("No model context, skipping TP/PP validation");
            return true;
        }

        TPPPValidationResult result;
        if (moe_pipeline_topology_ && moe_pipeline_topology_->globalPipeline())
        {
            // setupLocalTPContext deliberately creates no enclosing context:
            // the graph factory later gives every original stage its sole TP
            // owner. Validate the same sealed stage declarations before PMA or
            // native graph construction, rather than inventing another owner.
            if (local_tp_ctx_ || !mpi_ctx_)
                return setError("Global stage TP validation found a duplicate rank context or missing original MPI owner");
            result = validateResolvedGlobalStageTP(*moe_pipeline_topology_, *model_ctx_, mpi_ctx_->rank());
        }
        else
        {
            if(plan_.usesLocalTP() && !plan_.usesLocalPP() && !local_tp_ctx_)
                return setError("TP validation lacks the already-resolved LocalTP context");
            const auto membership=local_tp_ctx_
                ?TPPPAssignmentMembership::resolved(local_tp_ctx_->devices()):TPPPAssignmentMembership::structural();
            result = TPPPValidator::validate(config_, *model_ctx_,membership);
        }

        // Log warnings (but don't fail)
        for (const auto &warning : result.warnings)
        {
            LOG_WARN("[TP/PP Config] " << warning);
        }

        // Check for errors
        if (!result.valid)
        {
            std::ostringstream oss;
            oss << "TP/PP configuration is incompatible with model architecture:\n";
            for (const auto &error : result.errors)
            {
                oss << "  - " << error << "\n";
            }
            return setError(oss.str());
        }

        LOG_DEBUG("TP/PP configuration validated against model architecture");
        return true;
    }

    bool OrchestrationRunner::validateContextLength()
    {
        if (!model_ctx_)
            return true;

        const int model_max = model_ctx_->contextLength();
        if (model_max <= 0)
        {
            LOG_DEBUG("Model does not report max context length, skipping validation");
            return true;
        }

        if (config_.max_seq_len > model_max)
        {
            LOG_ERROR("Requested context length " << config_.max_seq_len
                                                  << " exceeds model maximum of " << model_max
                                                  << ". Use -c " << model_max << " or smaller.");
            return setError("Context length " + std::to_string(config_.max_seq_len) +
                            " exceeds model maximum of " + std::to_string(model_max));
        }

        LOG_DEBUG("Context length: " << config_.max_seq_len
                                     << " / " << model_max << " (model max)");
        return true;
    }

    bool OrchestrationRunner::prepareRankLocalGPUContexts(
        const std::vector<DeviceId> &devices,
        std::string_view purpose)
    {
        try
        {
            const auto overlay_world =
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_;
            const int world_rank = overlay_world
                                       ? overlay_world->rank()
                                       : plan_.rank;
            std::vector<DeviceId> unique_gpus;
            unique_gpus.reserve(devices.size());
            for (const DeviceId device : devices)
            {
                if (!device.is_gpu() ||
                    std::find(
                        unique_gpus.begin(),
                        unique_gpus.end(),
                        device) != unique_gpus.end())
                {
                    continue;
                }
                unique_gpus.push_back(device);
            }

            const auto computeWorkerCount = unique_gpus.size();
            const auto storageParticipants = plan_.runtime.ngram_table_policy.isGPUAdaptive()
                ? plan_.runtime.ngram_table_policy.gpu_adaptive.placement.declaredParticipants()
                : std::vector<AdaptiveGPUCacheParticipant>{};
            if (!unique_gpus.empty() || !storageParticipants.empty()) {
                if (world_rank < 0 || world_rank >= static_cast<int>(cluster_inventory_.ranks.size()))
                    return setError("Native GPU preparation has no original rank inventory");
                // Every GPU graph can own native event/graph resources. Their
                // typed BOM requires the same original prepared worker whether
                // ngrams use RAM, SSD, VRAM or adaptive caches. Storage policy
                // contributes endpoints; it never selects another birth path.
                // Storage endpoints are mandatory source-worker births, not
                // extra compute/expert participants. Preserve authored ordering
                // while deduplicating coordinator/consumer aliases exactly.
                for(const auto& endpoint:storageParticipants){
                    if(endpoint.world_rank!=world_rank || !endpoint.device().is_gpu())
                        throw std::invalid_argument("Declared immutable source worker must be an exact rank-local native GPU");
                    if(std::find(unique_gpus.begin(),unique_gpus.end(),endpoint.device())==unique_gpus.end())
                        unique_gpus.push_back(endpoint.device());
                }
                // Validate every selected role even when borrowing an existing
                // source. Storage-only/coordinator workers are not represented
                // by the plan's primary/TP devices and cannot borrow visibility
                // as permission to execute on another NUMA node.
                const auto &owner_inventory = cluster_inventory_.ranks.at(static_cast<std::size_t>(world_rank));
                for (const auto device : unique_gpus)
                    requireRankLocalGPUOwnership(owner_inventory, device,
                        "Native GPU preparation (" + std::string(purpose) + ")");
                // Immutable inventory startup is a separately measured baseline.
                // These ORIGINAL observations precede selected worker birth; no
                // later free-memory sample may adopt that birth after growth.
                auto source=plan_.runtime.native_context_preparation;
                if(!source && model_ctx_)source=model_ctx_->nativeContextPreparation();
                if(!source){
                    if(unique_gpus.empty())throw std::invalid_argument("Native source has no selected GPU participants");
                    const auto& observed=cluster_inventory_.ranks.at(static_cast<std::size_t>(world_rank));
                    std::vector<PhysicalMemoryResource> resources;
                    resources.push_back({world_rank,DeviceId::cpu(),observed.cpu.memory_bytes?observed.cpu.memory_bytes:observed.cpu_memory_bytes,
                        PhysicalMemoryAuthority::admissionCapacity(observed.cpu.free_memory_bytes,config_.memoryLimitBytes(DeviceType::CPU))});
                    for(const auto device:unique_gpus){
                        const auto found=std::find_if(observed.gpus.begin(),observed.gpus.end(),[&](const auto& gpu){return gpu.type==device.type && gpu.local_device_id==device.ordinal;});
                        if(found==observed.gpus.end())throw std::invalid_argument("Selected source worker is absent from original rank inventory");
                        resources.push_back(withObservedBacking({world_rank,device,found->memory_bytes,
                            PhysicalMemoryAuthority::admissionCapacity(found->free_memory_bytes,config_.memoryLimitBytes(device.type))},observed));
                    }
                    if(config_.moe_routed_expert_plan && config_.moe_routed_expert_plan->usesExpertOverlayAuthority()){
                        // The one source PMA must retain the SAME topology-wide
                        // registry later frozen by global capacity admission.
                        // Only its explicit local devices below create workers.
                        const int world_size=overlay_world?overlay_world->world_size():1;
                        const auto execution=resolveMoEExpertOverlayExecutionPlan(config_.moe_routed_expert_plan,
                            {.current_world_rank=world_rank,.world_size=world_size});
                        const auto metadata=resolveMoERoutedExpertModelMetadataForModel(*model_ctx_,retainedMTPConfig());
                        const auto policy=resolveMoEOverlayCapacityAdmissionPolicy(*config_.moe_routed_expert_plan,
                            config_,world_size,metadata.num_layers,metadata.num_experts);
                        resources=originalMoEOverlayNativePreparationResources(execution,cluster_inventory_,config_,policy,
                            plan_.runtime.ngram_table_policy);
                    }
                    source=NativeContextPreparation::create(resources,world_rank,unique_gpus);
                    try {
                        std::vector<PreparedGPUContextUsePermit> creation;
                        for(const auto device:unique_gpus)creation.push_back(source->acquire(device));
                        source->freezeForPlanning();
                    }catch(...){
                        // Actual created/failed births only. Foreign legacy workers
                        // must never be reset merely because this source rejected them.
                        source->retireUnadmitted();throw;
                    }
                    created_native_context_source_=source;
                    plan_.runtime.native_context_preparation=source;
                }else{
                    if(source->memory()->worldRank()!=world_rank)throw std::logic_error("Source worker plan crossed original physical ranks");
                    for(const auto device:unique_gpus){source->requireSelectedDevice(device);
                        const auto original=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,source->memory());
                        if(original.device()!=device)throw std::logic_error("Original source worker device identity changed");}
                    plan_.runtime.native_context_preparation=source;
                }
                PerfStatsCollector::addCounter("native_context_source","original_prequota_source",1.0,"model_setup",{},
                    {{"selected_workers",std::to_string(unique_gpus.size())},{"compute_workers",std::to_string(computeWorkerCount)},{"declared_storage_workers",std::to_string(storageParticipants.size())},{"runtime_startup_population","separate_baseline"},{"purpose",std::string(purpose)}});
                return true;
            }

            // A CPU-only rank has no native worker birth. Do not resample
            // physical capacity after setup: the original source PMA retains
            // its observations and accounts every original context commitment.
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(
                "Runtime GPU capacity observation failed: " +
                std::string(error.what()));
        }
    }

    bool OrchestrationRunner::publishPhysicalMemoryAuthority()
    {
        if (!model_ctx_ || !physical_memory_admission_)
        {
            return setError(
                "Physical-memory authority publication requires a model context and topology-wide admission certificate");
        }

        try
        {
            const int current_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            if(plan_.runtime.native_context_preparation){
                const auto& source=plan_.runtime.native_context_preparation;
                source->admitFinal(physical_memory_admission_);
                if(physical_memory_authority_ && physical_memory_authority_!=source->memory())
                    throw std::logic_error("Canonical publication cannot replace original source PMA");
                physical_memory_authority_=source->memory();
                if(!model_ctx_->nativeContextPreparation())model_ctx_->bindNativeContextPreparation(source);
                else if(model_ctx_->nativeContextPreparation()!=source)throw std::logic_error("Model source differs from original native admission");
            }
            if (!physical_memory_authority_)
            {
                physical_memory_authority_ = std::make_shared<
                    PhysicalMemoryAuthority>(
                    physical_memory_admission_, current_rank);
            }
            else if (physical_memory_authority_->worldRank() != current_rank ||
                     physical_memory_authority_->admission().get() !=
                         physical_memory_admission_.get())
            {
                throw std::logic_error(
                    "Physical-memory authority publication attempted to replace an admitted rank or plan");
            }

            const auto weight_manager =
                model_ctx_->concreteWeightManager();
            if (!weight_manager)
            {
                throw std::logic_error(
                    "Physical-memory authority publication has no concrete WeightManager");
            }
            weight_manager->installPhysicalMemoryAuthority(
                physical_memory_authority_);
            // Ngram storage is selected in the final immutable plan. Materialization
            // occurs at each participant factory after its ordinary upload drains;
            // publishing PMA must not consume that lower-priority allocation early.
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(
                "Physical-memory authority publication failed: " +
                std::string(error.what()));
        }
    }

    bool OrchestrationRunner::validateMemoryPlan()
    {
        if (!model_ctx_)
        {
            LOG_WARN("[MemoryPlanner] No model context — skipping memory validation");
            return true;
        }

        if(!plan_.runtime.native_context_preparation){
            if(!prepareRankLocalGPUContexts(rankLocalGPUExecutionDevices(plan_,config_.moe_routed_expert_plan,
                moe_expert_overlay_mpi_ctx_?moe_expert_overlay_mpi_ctx_:mpi_ctx_),"source_prequota_birth"))return false;
        }

        if (snapshot_capture_setup_.enabled() &&
            !snapshot_capture_setup_.memory_capacity.valid())
        {
            return setError(
                "Graph snapshot capture requires a complete pre-initialization device-memory capacity declaration");
        }

        if (moe_pipeline_topology_)
        {
            if (!moe_pipeline_prepared_plan_ || !physical_memory_authority_ ||
                physical_memory_authority_->admission() != moe_pipeline_prepared_plan_->admission()->physicalAdmission() ||
                physical_memory_admission_ != physical_memory_authority_->admission() ||
                !physical_memory_admission_->plan().fits())
                return setError("MoE pipeline final validation lost its aggregate physical authority");
            return true;
        }

        const bool has_expert_overlay =
            config_.moe_routed_expert_plan &&
            config_.moe_routed_expert_plan
                ->usesExpertOverlayAuthority();
        if (has_expert_overlay)
        {
            /*
             * Capacity resolution already joined model geometry, every local
             * and remote CPU/GPU allocator observation, graph/KV/workspace
             * owners, transport staging, shadow banks, and live expert copies.
             * Re-running MemoryPlanner here used to create a second authority
             * from a later telemetry sample and a reconstructed participant
             * map. Validate and publish the exact certificate selected before
             * any stable graph or migration address was materialized instead.
             */
            if (!moe_expert_overlay_memory_admission_)
            {
                return setError(
                    "ExpertOverlay reached final memory validation without its immutable CPU/GPU admission certificate");
            }
            try
            {
                if (!physical_memory_admission_ ||
                    physical_memory_admission_.get() !=
                        moe_expert_overlay_memory_admission_
                            ->physical_memory_admission.get())
                {
                    throw std::logic_error(
                        "ExpertOverlay final validation lost or replaced its topology-wide physical-memory authority");
                }
                (void)MoEOverlayCapacityResolver::installResolvedQuotas(
                    *config_.moe_routed_expert_plan,
                    *moe_expert_overlay_memory_admission_);

                const int current_rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
                bool owns_rank_resource = false;
                std::string report;
                for (const auto &resource :
                     moe_expert_overlay_memory_admission_
                         ->physical_resources)
                {
                    const auto &bom = resource.bom();
                    if (!bom.fits())
                    {
                        throw std::logic_error(
                            "ExpertOverlay retained an over-budget physical-memory certificate for " +
                            resource.resourceId());
                    }
                    owns_rank_resource = owns_rank_resource ||
                                         bom.resource().world_rank ==
                                             current_rank;
                    if (!report.empty())
                        report += '\n';
                    report += "  " + resource.resourceId() + " " +
                              bom.summary();
                }
                if (!owns_rank_resource)
                {
                    throw std::logic_error(
                        "ExpertOverlay memory certificate has no physical resource for the current rank");
                }
                if (!physical_memory_authority_ ||
                    physical_memory_authority_->worldRank() != current_rank ||
                    physical_memory_authority_->admission().get() !=
                        physical_memory_admission_.get())
                {
                    throw std::logic_error(
                        "ExpertOverlay final validation did not retain the rank-bound physical-memory authority published at admission");
                }
                LOG_DEBUG(
                    "[MemoryPlanner] ExpertOverlay immutable memory admission passed:\n"
                    << report);
                PerfStatsCollector::addCounter(
                    "memory_admission",
                    "expert_overlay_certificate_validated",
                    1.0,
                    "model_setup",
                    plan_.primary_device.toString(),
                    {{"physical_resources",
                      std::to_string(
                          moe_expert_overlay_memory_admission_
                              ->physical_resources.size())},
                     {"world_rank", std::to_string(current_rank)},
                     {"single_authority", "true"}});
                return true;
            }
            catch (const std::exception &error)
            {
                return setError(
                    "ExpertOverlay immutable memory admission validation failed: " +
                    std::string(error.what()));
            }
        }
        if (moe_expert_overlay_memory_admission_)
        {
            return setError(
                "A non-ExpertOverlay execution unexpectedly retained an ExpertOverlay memory authority");
        }

        if (retained_prepared_weight_plan_validated_)
        {
            /*
             * The retained pool is already absent from driver-reported free
             * memory and is credited explicitly below. Refresh after the
             * previous runner's topology has retired; a process-start snapshot
             * would otherwise reject valid reuse or admit against bytes owned
             * by a now-retired external process.
             */
            if (!prepareRankLocalGPUContexts(
                    rankLocalGPUExecutionDevices(
                        plan_,
                        config_.moe_routed_expert_plan,
                        moe_expert_overlay_mpi_ctx_
                            ? moe_expert_overlay_mpi_ctx_
                            : mpi_ctx_),
                    "retained_memory_preflight"))
            {
                return false;
            }
        }

        // Ordinary and ExpertOverlay admission consume the same immutable
        // descriptor used for placement, not separately reconstructed metadata.
        const auto &profile = planningModelMetadata().memoryProfile();

        /*
         * Fresh GPU preparation owns one device upload ring and one pinned-host
         * ring per participant. Resolve that geometry once from the same model
         * manifest and policy consumed by WeightManager. A certified retained
         * prepared set performs no upload, so it deliberately owns no staging
         * allocation in the new generation's plan.
         */
        const std::optional<GPUWeightLoadMemoryGeometry>
            fresh_gpu_weight_load_geometry =
                retained_prepared_weight_plan_validated_
                    ? std::nullopt
                    : std::optional<GPUWeightLoadMemoryGeometry>{
                          resolveGPUWeightLoadMemoryGeometry(
                              maximumGGUFTensorPayloadBytes(
                                  model_ctx_->model()),
                              configuredGPUWeightLoadMemoryPolicy())};

        std::optional<std::vector<int>> captured_prefill_buckets;
        if (debugEnv().execution.gpu_graphs && debugEnv().execution.prefill_graph_buckets)
            captured_prefill_buckets = debugEnv().execution.prefill_graph_bucket_sizes;
        auto device_configs = buildRankMemoryPlanInputs({
            .model = profile,
            .plan = plan_,
            .inventory = cluster_inventory_.ranks.at(plan_.rank),
            .weight_load_geometry = fresh_gpu_weight_load_geometry,
            .native_weight_source_policy = retained_prepared_weight_plan_validated_
                ? std::nullopt : std::optional{configuredGPUWeightLoadMemoryPolicy()},
            .weight_precision = WeightPrecision::NATIVE,
            .native_source_placement = retained_prepared_weight_plan_validated_
                ? std::nullopt : std::optional{model_ctx_->concreteLoader().nativeWeightSourcePlacement()},
            .snapshot_capacity = snapshot_capture_setup_.enabled()
                ? std::optional{snapshot_capture_setup_.memory_capacity} : std::nullopt,
            .captured_prefill_buckets = std::move(captured_prefill_buckets),
            .max_gpu_memory_bytes = config_.memoryLimitBytes(DeviceType::CUDA),
            .max_cpu_memory_bytes = config_.memoryLimitBytes(DeviceType::CPU),
        });

        // CPU threads are not needed to observe GPUs. Only immutable role/BOM
        // declarations precede quotas; actual CPU0 map/object/pthread birth is
        // normal final-Admitted materialization before source/graph capture.
        if(!plan_.runtime.retained_host_services)
            plan_.runtime.retained_host_services=MemoryPlanner::declareRetainedHostServices(profile,device_configs,plan_.rank,
                config_.moe_routed_expert_plan.get(),(moe_expert_overlay_mpi_ctx_?moe_expert_overlay_mpi_ctx_:mpi_ctx_).get());
        else {
            if(plan_.runtime.retained_host_services->worldRank()!=plan_.rank)throw std::logic_error("Runner CPU declarations crossed physical ranks");
            for(auto& cfg:device_configs)cfg.retained_host_services=plan_.runtime.retained_host_services;
        }

        const auto retained_table=imported_model_context_reuse_authority_
            ? model_ctx_->resolvedImmutableRowTablePlacement() : nullptr;
        if(retained_table)
        {
            if(!plan_.runtime.ngram_table_placement || *retained_table!=*plan_.runtime.ngram_table_placement)
                throw std::invalid_argument("Retained model and execution plan have different resolved ngram placements");
            for(auto &cfg:device_configs)
                if(cfg.ngram_table_placement)
                {
                    const auto execution=retained_table->lookupForConsumer({cfg.world_rank,cfg.device});
                    if(execution==ImmutableRowLookupExecution::AdaptiveGPUCache)
                    {
                        const auto& layout=retained_table->gpuAdaptiveLayout();
                        const auto& request=retained_table->gpuAdaptiveRequest();
                        const bool independent=layout.adaptiveGPUDomain()==ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal;
                        const auto sourceAllocator=independent?PhysicalMemoryAllocatorIdentity{cfg.world_rank,cfg.device}:
                            AdaptiveImmutableLookupGeometry::sourceLayout(layout,request).consumers().front();
                        const auto boundary=model_ctx_->adaptiveImmutableLookupBoundary(sourceAllocator.device);
                        if(!boundary)throw std::invalid_argument("Retained mixed ngrams lost their original source boundary");
                        const auto source=boundary->retainedSourceOwnerObservation();
                        if(source.authority().get()!=physical_memory_authority_.get() ||
                           source.authority().owner_before(physical_memory_authority_) || physical_memory_authority_.owner_before(source.authority()) ||
                           source.allocator()!=layout.sourceServices().front().allocator ||
                           source.hostPlacement()!=retained_table->hostPlacement())
                            throw std::invalid_argument("Retained mixed source changed its exact allocator, locality or canonical authority");
                        const auto checkedAdd=[](std::size_t& total,std::size_t bytes) {
                            if(bytes>std::numeric_limits<std::size_t>::max()-total)
                                throw std::overflow_error("Retained immutable lookup byte observation overflowed");
                            total+=bytes;
                        };
                        for(const auto& gpu:model_ctx_->retainedAdaptiveImmutableLookupOwners({cfg.world_rank,cfg.device})) {
                            if(gpu.authority().get()!=physical_memory_authority_.get() ||
                               gpu.authority().owner_before(physical_memory_authority_) || physical_memory_authority_.owner_before(gpu.authority()) ||
                               gpu.allocator()!=PhysicalMemoryAllocatorIdentity{cfg.world_rank,cfg.device})
                                throw std::invalid_argument("Retained mixed native roots changed their exact GPU/PMA authority");
                            checkedAdd(cfg.retained_immutable_table_bytes,gpu.tableStorageBytes());
                            checkedAdd(cfg.retained_immutable_lookup_staging_bytes,gpu.lookupStagingBytes());
                        }
                        // The shared source and local ready pages are distinct
                        // CPU owners. Only the authored coordinator imports
                        // their rank-wide credit; a reader's local GPU roots
                        // never manufacture an independent host source.
                        if(cfg.device==sourceAllocator.device) {
                            cfg.retained_immutable_source_table_bytes=source.tableStorageBytes();
                            cfg.retained_immutable_source_lookup_staging_bytes=source.lookupStagingBytes();
                            for(const auto& ready:model_ctx_->retainedAdaptiveImmutableLookupLocalReadyOwners()) {
                                if(ready.authority().get()!=physical_memory_authority_.get() ||
                                   ready.authority().owner_before(physical_memory_authority_) || physical_memory_authority_.owner_before(ready.authority()) ||
                                   ready.allocator()!=PhysicalMemoryAllocatorIdentity{cfg.world_rank,DeviceId::cpu()} || ready.tableStorageBytes())
                                    throw std::invalid_argument("Retained local ready page changed its original CPU/PMA owner");
                                checkedAdd(cfg.retained_immutable_local_ready_staging_bytes,ready.lookupStagingBytes());
                            }
                            if(!cfg.retained_immutable_lookup_staging_bytes || !cfg.retained_immutable_source_lookup_staging_bytes ||
                               !cfg.retained_immutable_source_table_bytes)
                                throw std::invalid_argument("Retained mixed C0 still requires its complete original frame/source/channel leases");
                        }
                    }
                    else if(immutableRowLookupUsesNativeDeviceStorage(execution))
                        cfg.retained_immutable_table_bytes=model_ctx_->residentImmutableRowTableBytes(cfg.device);
                    else
                    {
                        const auto table=model_ctx_->hostImmutableRowProvider();
                        if(!table)throw std::invalid_argument("Retained hosted ngrams lost their actual provider");
                        const auto owner=table->retainedOwnerObservation();
                        if(!retained_table->containsAllocator(owner.allocator()) || owner.hostPlacement()!=retained_table->hostPlacement() ||
                            owner.authority().get()!=physical_memory_authority_.get())
                            throw std::invalid_argument("Retained hosted ngram owner does not match its exact admitted placement/PMA");
                        cfg.retained_immutable_table_bytes=owner.tableStorageBytes();
                        cfg.retained_immutable_lookup_staging_bytes=owner.lookupStagingBytes();
                    }
                    if(execution!=ImmutableRowLookupExecution::AdaptiveGPUCache && !cfg.retained_immutable_table_bytes)
                        throw std::invalid_argument("Retained ngram placement has no complete physical allocation");
                    if(cfg.device.is_gpu())
                    {
                        if(plan_.runtime.resident_graph_rows<=0)
                            throw std::invalid_argument("Retained ngram graph has no original frozen row capacity");
                        cfg.activation_seq_len=plan_.runtime.resident_graph_rows;
                    }
                }
        }

        // Retained credits require the existing live owner attestations. The
        // shared metadata builder cannot manufacture this lifecycle evidence.
        for (auto &cfg : device_configs)
        {
            const DeviceId device = cfg.device;
            if (retained_prepared_weight_plan_validated_)
            {
                const auto weight_manager = model_ctx_->concreteWeightManager();
                const auto store = weight_manager
                                       ? weight_manager->preparedWeightStoreIfInitialized()
                                       : nullptr;
                const bool lifecycle_complete =
                    weight_manager &&
                    weight_manager->lifecycleGates().device_preparation_complete &&
                    weight_manager->lifecycleGates().graph_materialization_complete;
                if (!lifecycle_complete || !store ||
                    weight_manager->preparedRecordCountForDevice(device) == 0u)
                {
                    throw std::runtime_error(
                        "Certified retained prepared weights are no longer "
                        "complete for " +
                        device.toString());
                }

                /*
                 * Backend inventory reports live free bytes after the retained
                 * prepared pool allocation. MemoryPlanner must preserve that
                 * pool in its final BOM while charging only new graph/runtime
                 * storage. This applies equally to GPU and CPU participants:
                 * HostMemoryCapacity deliberately exposes live allocatable
                 * memory, so charging a certified CPU bank again would reject
                 * a process-campaign solely because that same bank is resident.
                 */
                cfg.prepared_weight_admission =
                    PreparedWeightAdmission::ReuseCertifiedCompleteSet;
                if (device.is_gpu())
                {
                    if (!reusable_execution_workspaces_ ||
                        !reusable_execution_workspaces_->valid())
                    {
                        throw std::runtime_error(
                            "Certified retained prepared weights have no valid reusable execution-workspace authority for " +
                            device.toString());
                    }
                    /*
                     * This is a physical-device credit, applied once per
                     * DevicePlanConfig in supported production topology. The
                     * registry reports only slots that completed graph teardown
                     * and reached backing-only state; live/invalid leases are
                     * never inferred from allocator telemetry.
                     */
                    cfg.retained_workspace_bytes =
                        reusable_execution_workspaces_
                            ->retainedPrimaryBytes(device);
                }
            }
        }

        const bool has_resident_graph_participant = std::any_of(
            device_configs.begin(),
            device_configs.end(),
            [](const DevicePlanConfig &config)
            {
                return config.device.is_gpu() ||
                       config.execution_role ==
                           DeviceExecutionMemoryRole::RoutedExpertParticipant;
            });

        MemoryPlan plan;
        if (!retained_table && has_resident_graph_participant &&
            debugEnv().execution.gpu_graphs &&
            debugEnv().execution.prefill_graph_buckets)
        {
            const auto configured_buckets = normalizePrefillGraphBuckets(
                debugEnv().execution.prefill_graph_bucket_sizes);
            auto resident_plan =
                MemoryPlanner::planLargestFittingResidentGraphRows(
                    profile,
                    device_configs,
                    configured_buckets);
            plan_.runtime.resident_graph_rows =
                resident_plan.resident_graph_rows;
            plan = std::move(resident_plan.memory_plan);
            device_configs=std::move(resident_plan.device_inputs);

            if (plan.fits())
            {
                LOG_INFO("[MemoryPlanner] Selected resident graph capacity "
                         << plan_.runtime.resident_graph_rows
                         << " rows; full KV context remains "
                         << plan_.runtime.max_seq_len
                         << " tokens");
            }
        }
        else
        {
            plan = MemoryPlanner::plan(profile, device_configs);
            plan_.runtime.resident_graph_rows =
                device_configs.empty()
                    ? 0
                    : device_configs.front().activation_seq_len;
        }

        if (!plan.fits())
        {
            std::string msg = "Memory plan validation failed — model does not fit on assigned device(s):\n";
            msg += plan.renderTable();
            for (const auto &d : plan.diagnostics)
            {
                msg += "\n  " + d;
            }
            return setError(msg);
        }

        PhysicalMemoryPlan final_plan=plan.physicalPlan();
        if(!retained_table)
        {
            const std::vector<std::vector<DevicePlanConfig>> groups{device_configs};
            const auto upload=MemoryPlanner::immutableRowTableUploadBindings(profile,groups);
            const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,
                plan_.runtime.ngram_table_policy,groups,upload,ImmutableRowTableHostPlacement::fromObservedNUMANode(plan_.numa_node));
            if(request)
            {
                if(imported_model_context_reuse_authority_)
                    throw std::invalid_argument("Retained model context has no exact complete ngram placement");
                auto selected=ImmutableRowTablePlacementPlanner::select(*request,final_plan);
                plan_.runtime.ngram_table_placement=std::move(selected.placement);
                final_plan=std::move(selected.physical_plan);
            }
        }

        if (imported_model_context_reuse_authority_)
        {
            if (!physical_memory_admission_ ||
                !physical_memory_authority_ ||
                physical_memory_authority_->admission().get() !=
                    physical_memory_admission_.get())
            {
                return setError(
                    "Prepared-model reuse lost its identity-preserving physical-memory authority");
            }
            if (const auto mismatch =
                    physical_memory_admission_->plan()
                        .requiredFootprintMismatch(
                            final_plan))
            {
                return setError(
                    "Prepared-model reuse exceeds its admitted physical-memory envelope: " +
                    *mismatch);
            }
            if (!publishPhysicalMemoryAuthority())
                return false;

            // The validated reuse handoff already retained the original
            // attention admission. This footprint check cannot resolve new
            // defaults or replace that immutable policy object.
            LOG_DEBUG(
                "[MemoryPlanner] Retained physical-memory authority accepts "
                "the current runner footprint:\n"
                << plan.renderTable());
            return true;
        }

        if (physical_memory_admission_ || physical_memory_authority_)
        {
            return setError(
                "Ordinary memory validation encountered an already-installed physical-memory authority");
        }
        physical_memory_admission_ = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(final_plan);
        if (!publishPhysicalMemoryAuthority())
            return false;
        plan_.runtime.attention_participants =
            projectAdmittedQwenSparseAttentionPrograms(profile, device_configs, plan_.rank);

        LOG_DEBUG("[MemoryPlanner] Memory validation passed:\n"
                  << plan.renderTable());
        return true;
    }

    void OrchestrationRunner::printStartupBanner(FILE *stream)
    {
        // The rank owning dense continuation state is the banner authority.
        if (mpi_ctx_ && mpi_ctx_->rank() != mpi_coordinated_root_rank_)
            return;

        StartupBannerData data;

        // Phase 1: Cluster topology
        data.cluster = &cluster_inventory_;
        if (config_.n_threads > 0)
            data.threads_per_rank = config_.n_threads;
        else
        {
            // cpu_cores is per-socket (this rank's local cores) — use directly as threads/rank
            if (!cluster_inventory_.ranks.empty())
            {
                data.threads_per_rank = cluster_inventory_.ranks[0].cpu_cores;
            }
        }
        data.bind_policy = "socket";

        // Phase 2: Inference configuration
        {
            DeviceId device = DeviceAddressAdapter::toDeviceId(plan_.primary_device);
            std::ostringstream dev_oss;
            dev_oss << device.to_string();
            if (device.is_cpu() && cluster_inventory_.world_size > 1)
            {
                int sockets = cluster_inventory_.ranks[0].cpu_sockets;
                int cores_per_socket = cluster_inventory_.ranks[0].cpu_cores;
                dev_oss << " (" << sockets << "S x " << cores_per_socket << "C, TP=" << cluster_inventory_.world_size << ")";
            }
            data.device_description = dev_oss.str();

            // Parallelism
            std::ostringstream par_oss;
            int effective_tp = plan_.totalTPDegree();
            par_oss << "TP=" << effective_tp;
            if (effective_tp > 1)
            {
                if (config_.tp_scope == TPScope::GLOBAL || config_.cpu_global_tp_all_local)
                    par_oss << " (global)";
                else if (plan_.usesLocalTP())
                    par_oss << " (local)";
            }
            par_oss << " | PP=" << config_.pp_degree;
            data.parallelism = par_oss.str();

            // Precision
            std::ostringstream prec_oss;
            prec_oss << "Activations: "
                     << activationPrecisionToString(
                            plan_.runtime.activation_precision)
                     << " | KV Cache: ";
            if (plan_.runtime.kv_cache_precision ==
                KVCachePrecision::TQ)
            {
                prec_oss << kvCachePrecisionToString(
                    plan_.runtime.kv_cache_precision);
            }
            else
            {
                prec_oss << activationPrecisionToString(
                    resolveKVCacheStoragePrecision(
                        plan_.runtime.kv_cache_precision,
                        device.is_cpu()));
            }
            data.precision = prec_oss.str();

            // Context length
            std::ostringstream ctx_oss;
            int model_max = model_ctx_ ? static_cast<int>(model_ctx_->contextLength()) : 0;
            ctx_oss << config_.max_seq_len;
            if (model_max > 0)
                ctx_oss << " / " << model_max << " (model max)";
            data.context_length = ctx_oss.str();

            // Backend
            if (device.is_cpu())
                data.backend = cpuBackendDescription();
            else if (device.is_cuda())
                data.backend = "CUDA (GPU " + std::to_string(device.ordinal) + ")";
            else if (device.is_rocm())
                data.backend = "ROCm (GPU " + std::to_string(device.ordinal) + ")";
        }

        // Phase 3: Model
        if (model_ctx_)
        {
            // Filename (basename)
            const std::string &path = config_.model_path;
            size_t slash = path.find_last_of('/');
            data.model_filename = (slash != std::string::npos) ? path.substr(slash + 1) : path;

            // File size
            const auto &model = model_ctx_->model();
            size_t file_bytes = 0;
            for (const auto &t : model.tensors)
                file_bytes += t.size_bytes;
            double file_gb = static_cast<double>(file_bytes) / (1024.0 * 1024.0 * 1024.0);
            char size_buf[32];
            snprintf(size_buf, sizeof(size_buf), "%.1f GB", file_gb);
            data.model_size = size_buf;

            // Architecture
            std::ostringstream arch_oss;
            arch_oss << model.architecture << " (" << model.block_count << " layers";
            // Check for MoE
            uint64_t n_experts = 0;
            auto it_experts = model.metadata.find("expert_count");
            if (it_experts != model.metadata.end())
                n_experts = it_experts->second.asUInt64();
            if (n_experts == 0)
            {
                auto it2 = model.metadata.find(model.architecture + ".expert_count");
                if (it2 != model.metadata.end())
                    n_experts = it2->second.asUInt64();
            }
            if (n_experts > 0)
            {
                arch_oss << ", " << n_experts << " experts";
                // top-k
                uint64_t top_k = 0;
                auto it_topk = model.metadata.find("expert_used_count");
                if (it_topk != model.metadata.end())
                    top_k = it_topk->second.asUInt64();
                if (top_k == 0)
                {
                    auto it2 = model.metadata.find(model.architecture + ".expert_used_count");
                    if (it2 != model.metadata.end())
                        top_k = it2->second.asUInt64();
                }
                if (top_k > 0)
                    arch_oss << ", top-" << top_k;
            }
            arch_oss << ")";
            data.architecture = arch_oss.str();

            // Vocab
            std::ostringstream vocab_oss;
            if (model.vocab_size > 0)
            {
                // Format with comma separators
                std::string vs = std::to_string(model.vocab_size);
                std::string formatted;
                int count = 0;
                for (int i = static_cast<int>(vs.size()) - 1; i >= 0; --i)
                {
                    if (count > 0 && count % 3 == 0)
                        formatted = "," + formatted;
                    formatted = vs[static_cast<size_t>(i)] + formatted;
                    count++;
                }
                vocab_oss << formatted << " tokens";
            }
            data.vocab = vocab_oss.str();

            // Thinking model detection
            auto it_think = model.metadata.find("tokenizer.chat_template");
            if (it_think != model.metadata.end())
            {
                const std::string &tmpl = it_think->second.asString();
                if (tmpl.find("<think>") != std::string::npos)
                    data.thinking = "Enabled (<think>...</think>)";
            }
        }

        // Phase 4: Preflight checks (all passed if we got here)
        {
            // Host RAM — we know it passed since we're past validateMemoryPlan
            PreflightCheckResult ram_check;
            ram_check.name = "Host RAM (weight staging)";
            ram_check.passed = true;
            if (model_ctx_)
            {
                const auto &model = model_ctx_->model();
                size_t weight_bytes = 0;
                for (const auto &t : model.tensors)
                    weight_bytes += t.size_bytes;
                char buf[64];
                const DeviceId primary_device =
                    DeviceAddressAdapter::toDeviceId(plan_.primary_device);
                const auto &load_config = debugEnv().rocm;
                if (primary_device.is_gpu() && config_.use_mmap &&
                    load_config.repack_budget_mb > 0)
                {
                    snprintf(
                        buf, sizeof(buf),
                        "%d MiB bounded staging cap",
                        load_config.repack_budget_mb);
                }
                else
                {
                    const double weight_gb =
                        static_cast<double>(weight_bytes) /
                        (1024.0 * 1024.0 * 1024.0);
                    snprintf(buf, sizeof(buf), "%.1f GB required", weight_gb);
                }
                ram_check.detail = buf;
            }
            data.preflight_checks.push_back(ram_check);

            PreflightCheckResult mem_check;
            mem_check.name = "Device memory (weights + KV + activ.)";
            mem_check.passed = true;
            mem_check.detail = "fits";
            data.preflight_checks.push_back(mem_check);

            PreflightCheckResult schema_check;
            schema_check.name = "Weight schema validation";
            schema_check.passed = true;
            if (model_ctx_)
            {
                char buf[64];
                snprintf(buf, sizeof(buf), "%lu tensors in model",
                         static_cast<unsigned long>(model_ctx_->model().tensor_count));
                schema_check.detail = buf;
            }
            data.preflight_checks.push_back(schema_check);
        }

        // Render and print
        bool use_color = StartupBanner::shouldUseColor();
        std::string banner = StartupBanner::render(data, use_color);

        // Print directly (bypassing LOG_INFO) to preserve ANSI colors.
        // LOG_INFO strips escape codes via its formatting pipeline.
        if (!banner.empty())
        {
            std::print(stream ? stream : stderr, "{}\n", banner);
        }
    }

    MoEOverlayParticipantGraphRunnerConfig OrchestrationRunner::makeMoEOverlayParticipantConfig(
        MoEOverlayParticipantGraphOwnership ownership) const
    {
        auto overlay_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto &active_mtp = retainedMTPConfig();
        const bool retains_mtp_capacity = retainsMTPGraphCapacity(active_mtp);
        return MoEOverlayParticipantGraphRunnerConfig{
                        .model_context = model_ctx_,
                        .ownership = ownership,
                        .cpu_serial_expert_storage = plan_.runtime.cpu_serial_expert_storage,
                        .prepared_weight_admission =
                            retained_prepared_weight_plan_validated_
                                ? PreparedWeightAdmission::
                                      ReuseCertifiedCompleteSet
                                : PreparedWeightAdmission::
                                      AllocateCompleteSet,
                        .mpi_context = std::move(overlay_ctx),
                        .placement_plan =
                            config_.moe_routed_expert_plan,
                        .residency_authority =
                            moe_expert_overlay_residency_authority_,
                        .participant_residency =
                            moe_expert_overlay_participant_residency_,
                        .decode_histogram =
                            moe_expert_overlay_decode_histogram_,
                        .device_controller_fabric =
                            moe_overlay_device_controller_fabric_,
                        .durable_maintenance_policy =
                            plan_.runtime.moe_rebalance.mode ==
                                    MoERebalanceRuntimeMode::Dynamic
                                ? MoEOverlayDurableMaintenancePolicy::
                                      DynamicPlacement
                                : MoEOverlayDurableMaintenancePolicy::
                                      Immutable,
                        .rank_batch_transport_registry =
                            moe_overlay_rank_batch_transport_registry_,
                        .max_seq_len = config_.max_seq_len,
                        /* The memory planner admits this exact bounded graph
                         * family.  Passing the full KV context here would
                         * over-allocate every remote compact route arena and
                         * make graph admission disagree with runtime storage. */
                        .max_graph_activation_rows =
                            std::max(
                                plan_.runtime.overlay_prefill_schedule
                                    .graph_row_capacity,
                                retains_mtp_capacity
                                    ? resolveMTPRetainedTargetQueryRows(
                                          active_mtp)
                                    : 1),
                        .prefill_graph_row_capacity = plan_.runtime.overlay_prefill_schedule.graph_row_capacity,
                        .prefill_graph_row_shapes =
                            plan_.runtime.overlay_prefill_schedule.bucket_rows,
                        .max_decode_activation_rows =
                            retains_mtp_capacity
                                ? resolveMTPRetainedTargetQueryRows(active_mtp)
                                : 1,
                        .mtp_graph_family_policy =
                            retains_mtp_capacity
                                ? MoEOverlayMTPGraphFamilyPolicy::
                                      RetainModelSidecars
                                : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                        .max_mtp_draft_depth =
                            resolveMTPRetainedDraftCapacity(active_mtp),
                        .max_request_count =
                            std::max(1, plan_.runtime.batch_size),
                        .cpu_joint_context=cpu_expert_shard_ctx_,
                    };
    }

    bool OrchestrationRunner::buildComputeGraph()
    {
        ScopedWeightLoadTimer timer(WeightLoadPhase::GRAPH_BUILD);

        if (config_.moe_routed_expert_plan && moe_expert_overlay_residency_authority_)
        {
            const auto snapshot = moe_expert_overlay_residency_authority_->snapshot();
            const auto context = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            if (!snapshot || !snapshot->valid() || !context)
                return setError("Expert endpoint construction lost original admitted ownership");
            const auto endpoints = MoEOverlayHostServiceContract::resolveAdmitted(
                *config_.moe_routed_expert_plan, snapshot->originalOwnerMap(), *context);
            if (endpoints.requiresRankLocalInferencePublisher() &&
                (!hasLocalTP() || plan_.usesLocalPP() || requiresNamedDomainGlobalGraph(config_)))
                return setError("Same-rank cross-vendor expert endpoints require their original flat LocalTP parent; other enclosing graph roles are not implemented");
        }

        // Cross-rank topology changes physical graph construction only. It must
        // not bypass model admission, snapshot installation, or readiness.
        if (requiresNamedDomainGlobalGraph(config_))
        {
            if (!plan_builder_)
                return setError("Named-domain graph has no topology builder");
            runner_ = buildNamedDomainGlobalGraph(
                config_, plan_, cluster_inventory_, *plan_builder_, model_ctx_, mpi_ctx_,
                moe_pipeline_stage_bindings_);
            // Resolve before readiness, while every rank still follows the
            // common initialization schedule. The public frontend then runs
            // on the same tail that owns logits, with no vocabulary transfer.
            mpi_coordinated_root_rank_ = coordinatedRootRankForRunner(
                config_.moe_routed_expert_plan, mpi_ctx_, runner_.get());
            return static_cast<bool>(runner_);
        }

        auto overlay_execution_plan = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_);
        if (overlay_execution_plan)
        {
            const auto execution_kind =
                overlay_execution_plan->currentRankPlan().execution_kind;
            if (execution_kind ==
                OverlayRankExecutionKind::RelayOnly)
            {
                return setError(
                    "ExpertOverlay relay-only ranks cannot enter model graph construction");
            }
            if (execution_kind ==
                OverlayRankExecutionKind::ExpertOnlyFollower)
            {
                auto overlay_ctx =
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_;
                const MTPRuntimeConfig &active_mtp = retainedMTPConfig();
                const bool retains_mtp_capacity =
                    retainsMTPGraphCapacity(active_mtp);
                runner_ = createMoEOverlayParticipantGraphRunner(
                    makeMoEOverlayParticipantConfig(MoEOverlayParticipantGraphOwnership::ExpertOnlyRank));
                if (!runner_)
                {
                    return setError(
                        "Failed to create the rank-local MoE overlay "
                        "participant graph");
                }

                /*
                 * The expert-only participant is a complete production graph,
                 * not a partial precursor to the continuation rank's graph.
                 * Its constructor has prepared every local endpoint binding and
                 * resolved those bindings into retained graph stages. Publish
                 * the same model-owned lifecycle gates that RankOrchestrator
                 * publishes for a continuation graph. Without this edge, a
                 * later runner cannot distinguish a complete four-device ROCm
                 * authority from a half-built PreparedWeightStore even though
                 * no further materialization is possible on this rank.
                 */
                const auto weight_manager =
                    model_ctx_->concreteWeightManager();
                if (!weight_manager)
                {
                    return setError(
                        "ExpertOverlay participant graph has no concrete "
                        "WeightManager lifecycle authority");
                }
                weight_manager->markDevicePreparationComplete();
                weight_manager->markGraphMaterializationComplete();
                LOG_DEBUG(
                    "[OrchestrationRunner] Built rank-local expert-only "
                    "MoE overlay graph for rank "
                    << overlay_execution_plan->currentRankPlan().world_rank);
                return true;
            }
            if (!overlay_execution_plan->ownsContinuationGraph())
            {
                return setError(
                    "ExpertOverlay rank graph construction received an "
                    "unhandled execution lifecycle");
            }
        }

        // Check if LOCAL PP is configured (takes priority over TP, because
        // TP-in-PP composition creates per-stage TP contexts inside the MDO)
        if (plan_.usesLocalPP())
        {
            return buildLocalPPComputeGraph();
        }

        // Check if LOCAL TP is configured (multiple devices within this rank)
        if (hasLocalTP())
        {
            return buildMultiDeviceComputeGraph();
        }

        // Single-device path
        return buildSingleDeviceComputeGraph();
    }

    bool OrchestrationRunner::buildMultiDeviceComputeGraph()
    {
        // Validate that all requested devices actually exist in hardware
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            auto local_device = plan_.local_tp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("TP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        LOG_DEBUG("[OrchestrationRunner] Execution strategy: MULTI-DEVICE (LOCAL TP)");
        LOG_DEBUG("[OrchestrationRunner]   TP degree: " << plan_.local_tp_devices.size());

        // Log each device
        for (size_t i = 0; i < plan_.local_tp_devices.size(); ++i)
        {
            const auto &dev = plan_.local_tp_devices[i];
            std::string weight_str = "";
            if (i < plan_.local_tp_weights.size())
            {
                weight_str = " (weight=" + std::to_string(plan_.local_tp_weights[i]) + ")";
            }
            LOG_DEBUG("[OrchestrationRunner]   Device " << i << ": " << dev.toString() << weight_str);
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = RankOrchestrator::Config::fromPlan(plan_);
        if (retained_prepared_weight_plan_validated_)
        {
            const auto weight_manager =
                model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            const auto prepared_store = weight_manager
                                            ? weight_manager
                                                  ->preparedWeightStoreIfInitialized()
                                            : nullptr;
            if (!prepared_store)
            {
                return setError(
                    "Validated prepared-weight reuse lost its model-owned "
                    "PreparedWeightStore before LocalTP graph construction");
            }
            mdo_config.prepared_weight_store = prepared_store;
            mdo_config.prepared_weight_admission =
                PreparedWeightAdmission::ReuseCertifiedCompleteSet;
        }
        mdo_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
        mdo_config.moe_expert_overlay_residency_authority =
            moe_expert_overlay_residency_authority_;
        mdo_config.moe_expert_overlay_participant_residency =
            moe_expert_overlay_participant_residency_;
        mdo_config.moe_expert_overlay_cpu_joint_context=cpu_expert_shard_ctx_;
        mdo_config.moe_expert_overlay_decode_histogram =
            moe_expert_overlay_decode_histogram_;
        mdo_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        mdo_config.moe_rank_batch_transport_registry =
            moe_overlay_rank_batch_transport_registry_;
        mdo_config.moe_device_controller_fabric =
            moe_overlay_device_controller_fabric_;
        mdo_config.reusable_execution_workspaces =
            reusable_execution_workspaces_;
        if (config_.moe_routed_expert_plan && moe_expert_overlay_residency_authority_)
        {
            const auto snapshot = moe_expert_overlay_residency_authority_->snapshot();
            const auto context = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            if (!snapshot || !snapshot->valid() || !context)
                return setError("Rank auxiliary expert selection lost its original admitted ownership");
            const auto endpoints = MoEOverlayHostServiceContract::resolveAdmitted(
                *config_.moe_routed_expert_plan, snapshot->originalOwnerMap(), *context);
            if (endpoints.requiresRankLocalInferencePublisher())
                mdo_config.rank_local_expert_endpoints = makeMoEOverlayParticipantConfig(
                    MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary);
        }

        LOG_DEBUG("[OrchestrationRunner] Multi-device precision config: activation="
                  << activationPrecisionToString(mdo_config.activation_precision)
                  << ", kv_cache=" << kvCachePrecisionToString(mdo_config.kv_cache_precision));

        // Validate config
        if (!mdo_config.validate())
        {
            return setError("Invalid multi-device configuration");
        }

        // Create RankOrchestrator via factory
        // Note: local_tp_ctx_ was already created in setupLocalTPContext()
        auto multi_orchestrator = createRankOrchestrator(
            model_ctx_,
            std::move(local_tp_ctx_),
            mdo_config);

        if (!multi_orchestrator)
        {
            return setError("Failed to create RankOrchestrator");
        }

        // Store as IInferenceRunner (RankOrchestrator extends it)
        runner_ = std::move(multi_orchestrator);

        LOG_DEBUG("Multi-device compute graph built successfully");
        return true;
    }

    bool OrchestrationRunner::buildLocalPPComputeGraph()
    {
        const auto &pp_devices = plan_.local_pp_devices;
        const auto &boundaries = plan_.local_pp_layer_boundaries;

        if (pp_devices.size() < 2 || boundaries.size() < pp_devices.size() + 1)
        {
            return setError("Invalid LOCAL PP plan: need >=2 devices and matching layer boundaries");
        }

        LOG_DEBUG("[OrchestrationRunner] Execution strategy: LOCAL PIPELINE PARALLEL");
        LOG_DEBUG("[OrchestrationRunner]   PP stages: " << pp_devices.size());

        // Validate all devices exist
        const auto &dm = DeviceManager::instance();
        for (size_t i = 0; i < pp_devices.size(); ++i)
        {
            auto local_device = pp_devices[i].toLocalDeviceId();
            if (!dm.deviceExists(local_device))
            {
                return setError("PP device " + std::to_string(i) + " (" +
                                local_device.toString() +
                                ") is not available. Available devices: " +
                                dm.availableDevicesString());
            }
        }

        // Build config from execution plan via canonical factory
        auto mdo_config = RankOrchestrator::Config::fromPlan(plan_);
        mdo_config.original_pipeline_mpi_ctx = mpi_ctx_;
        if (retained_prepared_weight_plan_validated_)
        {
            const auto prepared = model_ctx_->concreteWeightManager()->preparedWeightStoreIfInitialized();
            if (!prepared) return setError("Retained PP weights lost their prepared-store authority");
            mdo_config.prepared_weight_admission = PreparedWeightAdmission::ReuseCertifiedCompleteSet;
            mdo_config.prepared_weight_store = prepared;
        }
        if (config_.moe_routed_expert_plan || moe_expert_overlay_residency_authority_ ||
            moe_expert_overlay_participant_residency_ || moe_expert_overlay_decode_histogram_ ||
            moe_overlay_rank_batch_transport_registry_ || moe_overlay_device_controller_fabric_)
            return setError("LOCAL PP requires stage-owned expert admission; a model-wide overlay cannot be installed");
        if (!moe_pipeline_stage_bindings_.empty())
        {
            if (moe_pipeline_stage_bindings_.size() != mdo_config.pp_stages.size())
                return setError("LOCAL PP expert admission omitted an authored pipeline stage");
            for (std::size_t stage = 0; stage < mdo_config.pp_stages.size(); ++stage)
                mdo_config.pp_stages[stage].moe_runtime = moe_pipeline_stage_bindings_[stage];
        }
        mdo_config.reusable_execution_workspaces =
            reusable_execution_workspaces_;

        // Log PP stage details
        for (size_t i = 0; i < mdo_config.pp_stages.size(); ++i)
        {
            const auto &stage = mdo_config.pp_stages[i];
            LOG_DEBUG("[OrchestrationRunner]   Stage " << i << ": "
                                                       << pp_devices[i].toString()
                                                       << " layers [" << stage.first_layer << ", "
                                                       << stage.last_layer << ") "
                                                       << (stage.has_embedding ? "[+embed] " : "")
                                                       << (stage.has_lm_head ? "[+lm_head] " : ""));
        }

        if (!mdo_config.validate())
        {
            return setError("Invalid LOCAL PP configuration");
        }

        // The original active MPI context already owns the canonical discovery
        // membership projection. Captured PP edges own their native transports;
        // a router must not create copy communicators from host-wide visibility.
        std::vector<std::vector<GlobalDeviceAddress>> router_stages;
        for(const auto& stage:mdo_config.pp_stages)router_stages.push_back(stage.stage_devices);
        std::unique_ptr<RankOrchestrator> orch;
        orch = std::make_unique<RankOrchestrator>(model_ctx_, mdo_config);
        GlobalBackendRouter::initForPipeline(PipelineRouterAdmission::admit(
            mpi_ctx_,router_stages,orch->pipelineGraphExecutionPlan()));
        runner_ = std::move(orch);

        LOG_DEBUG("Local PP compute graph built successfully");
        return true;
    }

    bool OrchestrationRunner::buildSingleDeviceComputeGraph()
    {
        // Determine target device from execution plan
        DeviceId device = DeviceId::cpu();
        std::string device_source = "default (CPU)";

        if (!plan_.local_tp_devices.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.local_tp_devices[0]";
        }
        else if (!plan_.primary_device.hostname.empty())
        {
            device = plan_.primary_device.toLocalDeviceId();
            device_source = "plan.primary_device";
        }

        // Validate that the requested device actually exists in hardware
        const auto &dm = DeviceManager::instance();
        const bool strict_numa =
            plan_.hasResolvedPrimaryDeviceNuma();

        const bool device_available = strict_numa
                                          ? dm.deviceExists(plan_.primary_device, true)
                                          : dm.deviceExists(device);

        if (!device_available)
        {
            if (strict_numa)
            {
                return setError("Requested device " + plan_.primary_device.toString() +
                                " is not available on the specified NUMA node. Available devices: " +
                                dm.availableDevicesString());
            }

            return setError("Requested device " + device.toString() +
                            " is not available. Available devices: " +
                            dm.availableDevicesString());
        }

        // Log execution strategy decision
        LOG_DEBUG("[OrchestrationRunner] Execution strategy: SINGLE-DEVICE");
        LOG_DEBUG("[OrchestrationRunner]   Target device: " << device.toString());
        LOG_DEBUG("[OrchestrationRunner]   Device source: " << device_source);
        if (device.is_cpu())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: " << cpuBackendDescription());
        }
        else if (device.is_cuda())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: CUDA (GPU " << device.ordinal << ")");
        }
        else if (device.is_rocm())
        {
            LOG_DEBUG("[OrchestrationRunner]   Backend: ROCm (GPU " << device.ordinal << ")");
        }

        // Build config from execution plan via canonical factory
        auto runner_config = InferenceRunnerConfig::fromPlan(plan_);
        if (retained_prepared_weight_plan_validated_)
        {
            const auto weight_manager =
                model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            const auto prepared_store = weight_manager
                                            ? weight_manager
                                                  ->preparedWeightStoreIfInitialized()
                                            : nullptr;
            if (!prepared_store)
            {
                return setError(
                    "Validated prepared-weight reuse lost its model-owned "
                    "PreparedWeightStore before single-device graph construction");
            }
            runner_config.prepared_weight_store = prepared_store;
            runner_config.prepared_weight_admission =
                PreparedWeightAdmission::ReuseCertifiedCompleteSet;
        }
        runner_config.hostfile = config_.hostfile;
        runner_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
        runner_config.moe_expert_overlay_residency_authority =
            moe_expert_overlay_residency_authority_;
        runner_config.moe_expert_overlay_participant_residency =
            moe_expert_overlay_participant_residency_;
        runner_config.moe_expert_overlay_cpu_joint_context=cpu_expert_shard_ctx_;
        runner_config.moe_expert_overlay_decode_histogram =
            moe_expert_overlay_decode_histogram_;
        runner_config.moe_expert_overlay_mpi_ctx = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        runner_config.moe_rank_batch_transport_registry =
            moe_overlay_rank_batch_transport_registry_;
        runner_config.moe_device_controller_fabric =
            moe_overlay_device_controller_fabric_;
        runner_config.reusable_execution_workspaces =
            reusable_execution_workspaces_;

        LOG_DEBUG("[OrchestrationRunner] Single-device precision config: activation="
                  << activationPrecisionToString(runner_config.activation_precision)
                  << ", kv_cache=" << kvCachePrecisionToString(runner_config.kv_cache_precision));

        // Create runner via factory (returns IInferenceRunner)
        if (model_ctx_)
        {
            std::shared_ptr<IMPIContext> graph_mpi_ctx = mpi_ctx_;
            const auto overlay_execution =
                resolveOverlayExecutionPlanForRunner(
                    config_.moe_routed_expert_plan,
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_);
            if (overlay_execution &&
                overlay_execution->ownsContinuationGraph() &&
                overlay_execution->world_size > 1 &&
                !plan_.usesGlobalTP())
            {
                /*
                 * A non-TP dense continuation graph is participant-local. Its
                 * only cross-rank edges are the explicitly configured sparse
                 * MoE stages, which retain the overlay world in runner_config.
                 * A NodeTP continuation is different: every rank builds
                 * one dense shard and ordinary graph construction must retain
                 * that exact global TP communicator.
                 */
                graph_mpi_ctx = MPIContextFactory::self();
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "root_graph_rank_local_mpi_scope",
                    1.0,
                    "graph_build",
                    {},
                    {{"overlay_world_size",
                      std::to_string(overlay_execution->world_size)}});
            }
            runner_ = createInferenceRunner(
                model_ctx_,
                std::move(graph_mpi_ctx),
                device,
                runner_config);
        }

        if (!runner_ && model_ctx_)
        {
            return setError("Failed to create inference runner");
        }

        LOG_DEBUG("[OrchestrationRunner] Compute graph built successfully");
        return true;
    }

} // namespace llaminar2
