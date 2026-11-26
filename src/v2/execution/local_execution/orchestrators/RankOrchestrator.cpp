/**
 * @file RankOrchestrator.cpp
 * @brief RankOrchestrator construction, lifecycle, and rank_orchestrator_detail helpers.
 */
#include "RankOrchestratorInternal.h"

namespace llaminar2
{
    namespace rank_orchestrator_detail
    {
        /**
         * @brief Emit a structured reason when a rank-level prefix snapshot cannot be built.
         *
         * Multi-device MTP treats rollback checkpoint capture as an all-or-nothing
         * transaction.  If one participant cannot provide a snapshot, or if two
         * participants disagree about the live token count, the parent
         * RankOrchestrator must reject the entire checkpoint.  This helper keeps
         * that rejection observable in perfstats and logs so a failed end-to-end
         * parity cell does not collapse into the generic
         * "could not capture live prefix state" message.
         *
         * @param operation Human-readable operation name, for example
         *        "capture_live_prefix_checkpoint".
         * @param runner_group Which child collection was being traversed
         *        ("device" for LocalTP or "pp" for LocalPP).
         * @param participant_index Stable index among non-null participants.
         * @param runner Child runner that produced the failure, or nullptr if
         *        the failure happened before a child call.
         * @param reason Short machine-readable reason tag.
         * @param expected_tokens Common token count already observed, or -1 when
         *        no common count exists yet.
         * @param actual_tokens Token count from the current child, or -1 when
         *        unavailable.
         */
        void recordRankPrefixSnapshotFailure(
            const char *operation,
            const char *runner_group,
            size_t participant_index,
            const IInferenceRunner *runner,
            const std::string &reason,
            int expected_tokens,
            int actual_tokens)
        {
            const std::string operation_name =
                operation && operation[0] != '\0'
                    ? operation
                    : "capture_live_prefix_snapshot";
            const std::string group_name =
                runner_group && runner_group[0] != '\0'
                    ? runner_group
                    : "unknown";
            const std::string device_name =
                runner ? runner->primaryDeviceId().toString() : "unknown";

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_live_prefix_snapshot_failures",
                1.0,
                "decode",
                device_name,
                {{"operation", operation_name},
                 {"runner_group", group_name},
                 {"participant_index", std::to_string(participant_index)},
                 {"reason", reason},
                 {"expected_tokens", std::to_string(expected_tokens)},
                 {"actual_tokens", std::to_string(actual_tokens)}});

            LOG_WARN("[RankOrchestrator] " << operation_name
                                           << " failed for " << group_name
                                           << " participant=" << participant_index
                                           << " device=" << device_name
                                           << " reason=" << reason
                                           << " expected_tokens=" << expected_tokens
                                           << " actual_tokens=" << actual_tokens);
        }

        bool sameBackendGpuExpertTransferIsDirectOnly(DeviceId destination, DeviceId source)
        {
            return destination.is_gpu() &&
                   source.is_gpu() &&
                   destination.type == source.type;
        }

        std::vector<std::vector<std::vector<bool>>> buildReplicaArrivalTransferMasks(
            const MoERebalanceController &controller,
            const ExpertReplicaSet &arrivals)
        {
            const int participants = arrivals.participantCount();

            const int num_layers = controller.numLayers();
            const int num_experts = controller.numExperts();
            std::vector<std::vector<std::vector<bool>>> masks_by_participant(
                static_cast<size_t>(std::max(0, participants)),
                std::vector<std::vector<bool>>(
                    static_cast<size_t>(std::max(0, num_layers)),
                    std::vector<bool>(static_cast<size_t>(std::max(0, num_experts)), false)));

            if (participants <= 0 || num_layers <= 0 || num_experts <= 0)
                return masks_by_participant;

            for (int participant = 0; participant < participants; ++participant)
            {
                for (int layer = 0; layer < num_layers; ++layer)
                {
                    for (int expert_id = 0; expert_id < num_experts; ++expert_id)
                    {
                        const int owner =
                            arrivals.ownerParticipant(layer, expert_id);
                        if (owner < 0 || owner >= participants || participant == owner)
                            continue;
                        if (arrivals.hasReplicaOnParticipant(layer, expert_id, participant))
                            masks_by_participant[static_cast<size_t>(participant)][static_cast<size_t>(layer)][static_cast<size_t>(expert_id)] = true;
                    }
                }
            }

            return masks_by_participant;
        }

        std::vector<std::vector<std::vector<bool>>> buildOwnershipArrivalTransferMasks(
            const MoERebalanceController &controller,
            const MoELayeredExpertOwnership &previous_ownership)
        {
            const int participants = controller.participantCount();
            const int num_layers = controller.numLayers();
            const int num_experts = controller.numExperts();
            std::vector<std::vector<std::vector<bool>>> masks_by_participant(
                static_cast<size_t>(std::max(0, participants)),
                std::vector<std::vector<bool>>(
                    static_cast<size_t>(std::max(0, num_layers)),
                    std::vector<bool>(static_cast<size_t>(std::max(0, num_experts)), false)));

            if (participants <= 0 || num_layers <= 0 || num_experts <= 0)
                return masks_by_participant;

            const auto &current_ownership = controller.currentOwnership();
            for (const auto &change :
                 current_ownership.changesFrom(previous_ownership))
            {
                if (change.current_participant < 0 ||
                    change.current_participant >= participants)
                {
                    throw std::logic_error(
                        "Layered MoE ownership change targets an invalid participant");
                }
                masks_by_participant
                    [static_cast<size_t>(change.current_participant)]
                    [static_cast<size_t>(change.layer_idx)]
                    [static_cast<size_t>(change.expert_id)] = true;
            }

            return masks_by_participant;
        }
    }


    // =========================================================================
    // Config Implementation
    // =========================================================================

    // =========================================================================
    // Factory Methods
    // =========================================================================

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTest(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
    {
        // Use the private constructor
        return std::unique_ptr<RankOrchestrator>(
            new RankOrchestrator(
                std::move(model_ctx),
                std::move(device_runners),
                std::move(tp_ctx),
                config));
    }

    std::unique_ptr<RankOrchestrator> RankOrchestrator::createForTestWithPipelineStages(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> pp_stage_runners,
        const Config &config)
    {
        auto orchestrator = createForTest(
            std::move(model_ctx),
            {},
            nullptr,
            config);
        orchestrator->mode_ = ParallelismMode::PP;
        orchestrator->pp_stage_runners_ = std::move(pp_stage_runners);
        orchestrator->initializePPGraphExecutionPlan();

        /*
         * Injected unit-test runners stand in for already prepared endpoints.
         * The public production constructor never takes this shortcut: its
         * native children cross the real materialization transition before
         * request admission.
         */
        std::string materialization_error;
        if (!orchestrator->pp_graph_execution_plan_->markMaterialized(
                orchestrator->pp_graph_execution_plan_->nativeSegmentCount(),
                &materialization_error))
        {
            throw std::runtime_error(materialization_error);
        }
        return orchestrator;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        const Config &config,
        std::unique_ptr<ILocalTPContext> tp_ctx)
        : model_ctx_(std::move(model_ctx)),
          config_(config),
          current_batch_size_(std::max(1, config.batch_size))
    {
        // This constructor is also called directly by serving and nested PP;
        // reject a test-only issuer before scratch or child setup can start.
        if (config_.test_runner_overrides.test_graph_allocation_admission &&
            std::dynamic_pointer_cast<ModelContext>(model_ctx_))
            throw std::invalid_argument("Concrete rank model cannot accept test graph allocation admission");

        if (!config_.validate())
        {
            throw std::invalid_argument("Invalid RankOrchestrator configuration");
        }
        if (resolvedMoEOverlayDiagnosticPrefillTransport())
        {
            // Auxiliary expert ranks share the original global overlay plan;
            // only its exact multi-GPU continuation cell applies this policy.
            const auto plan = config_.moe_routed_expert_plan;
            if (!plan)
                throw std::invalid_argument(
                    "Diagnostic MoE prefill route transport requires an ExpertOverlay continuation plan");
            const auto domain = std::find_if(plan->domains.begin(), plan->domains.end(),
                [&](const auto &candidate) {
                    return candidate.name == plan->continuation_domain_spec.domain;
                });
            if (domain == plan->domains.end())
                throw std::invalid_argument(
                    "Diagnostic MoE prefill route transport has no original continuation domain");
            std::vector<DeviceId> local_devices, continuation_devices;
            for (const auto &address : tp_ctx ? tp_ctx->devices() : config_.devices)
                local_devices.push_back(address.toLocalDeviceId());
            for (const auto &address : domain->participants)
                continuation_devices.push_back(address.toLocalDeviceId());
            (void)validateMoEOverlayDiagnosticPrefillTransportOwner(
                local_devices, continuation_devices,
                tp_ctx || config_.effectiveMode() == ParallelismMode::TP);
        }
        initializeRankMTPScratch();

        // Initialize stage sharding map from model architecture
        stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(model_ctx_->architecture());

        if (tp_ctx)
        {
            // Pre-existing TP context provided — force TP mode
            tp_ctx_ = std::move(tp_ctx);
            mode_ = ParallelismMode::TP;
            applyRuntimeSnapshotShardingOverrides();

            if (tp_ctx_->degree() < 2)
            {
                LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                           << ", multi-device orchestration may not be beneficial");
            }

            LOG_DEBUG("RankOrchestrator: Creating with pre-existing TP context, "
                      << tp_ctx_->degree() << " devices");

            initializeDeviceRunners();

            LOG_DEBUG("RankOrchestrator: Initialized with " << device_runners_.size() << " device runners");
        }
        else
        {
            // Auto-detect mode from config
            mode_ = config_.effectiveMode();

            LOG_DEBUG("RankOrchestrator: Creating with mode="
                      << (mode_ == ParallelismMode::TP ? "TP" : mode_ == ParallelismMode::PP ? "PP"
                                                                                             : "TP_PP")
                      << ", backend=" << static_cast<int>(config_.backend));

            if (mode_ == ParallelismMode::TP)
            {
                // Pure TP mode - create LOCAL TP context from config
                tp_ctx_ = createLocalTPContext(
                    config_.devices,
                    config_.getNormalizedWeights(),
                    config_.backend);

                if (!tp_ctx_)
                {
                    throw std::runtime_error("Failed to create LOCAL TP context");
                }
                applyRuntimeSnapshotShardingOverrides();

                if (tp_ctx_->degree() < 2)
                {
                    LOG_WARN("RankOrchestrator: TP degree is " << tp_ctx_->degree()
                                                               << ", multi-device orchestration may not be beneficial");
                }

                initializeDeviceRunners();

                LOG_DEBUG("RankOrchestrator: Initialized TP mode with " << device_runners_.size() << " device runners");
            }
            else
            {
                // PP or TP_PP mode - create PP context and stage runners
                initializePPDeviceRunners();
                initializePPContext();

                LOG_DEBUG("RankOrchestrator: Initialized PP mode with " << config_.pp_stages.size() << " stages");
            }
        }
    }

    // Private constructor for createForTest
    RankOrchestrator::RankOrchestrator(
        std::shared_ptr<IModelContext> model_ctx,
        std::vector<std::unique_ptr<IInferenceRunner>> device_runners,
        std::unique_ptr<ILocalTPContext> tp_ctx,
        const Config &config)
        : model_ctx_(std::move(model_ctx)),
          tp_ctx_(std::move(tp_ctx)),
          mode_(ParallelismMode::TP), // Test factory currently only supports TP mode
          device_runners_(std::move(device_runners)),
          config_(config),
          logits_backend_resolver_(resolveNoBackendForInjectedUnitTest),
          external_device_backend_access_enabled_(false),
          current_batch_size_(std::max(1, config.batch_size))
    {
        initializeRankMTPScratch();

        // Initialize stage sharding map from model architecture (if registered)
        const auto arch = model_ctx_->architecture();
        if (SchemaFactoryRegistry::isSupported(arch))
        {
            stage_sharding_map_ = SchemaFactoryRegistry::getStageShardingConfig(arch);
            applyRuntimeSnapshotShardingOverrides();
        }

        LOG_DEBUG("RankOrchestrator: Created via createForTest with "
                  << device_runners_.size() << " injected device runners");

        if (ownsMainForwardLogits() && model_ctx_ && !device_runners_.empty())
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    vocab,
                    /*max_tokens=*/1,
                    logits_backend_resolver_);

                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (external_device_backend_access_enabled_ &&
                        primary_dev.is_gpu())
                    {
                        logits_gatherer_->pinForDevice(primary_dev);
                    }
                }
                applyLogitsGatherSkipFlags();
            }
        }

        wireLocalTPMoERuntimeHistogramSyncs();
    }

    std::unique_ptr<TPWorkerPool> RankOrchestrator::makeParticipantWorkerPool(
        const std::vector<std::unique_ptr<IInferenceRunner>> &participants) const
    {
        std::vector<DeviceId> members;
        members.reserve(participants.size());
        for (const auto &runner : participants)
        {
            if (!runner) throw std::logic_error("Cannot place an absent TP participant worker");
            members.push_back(runner->primaryDeviceId());
        }
        const auto scope = &participants == &device_runners_ &&
                mode_ == ParallelismMode::TP && pp_stage_runners_.empty() &&
                !config_.hasPP() && !config_.nested_pp_stage_config
            ? TPWorkerHostScope::FlatRank : TPWorkerHostScope::PipelineOrComposed;
        return std::make_unique<TPWorkerPool>(members.size(), tpWorkerHostPlacement(members, scope));
    }



    void RankOrchestrator::retireRetainedSourceFailureCallback()
    {
        if (!tp_worker_pool_) return;
        tp_worker_pool_->retireFailureCallback([context=tp_ctx_.get()] {
            if (context) context->requestAbort();
        });
    }

    void RankOrchestrator::failRetainedSourceAttempt(const char* operation) noexcept
    {
        if (!retained_source_admission_) return;
        retained_source_admission_->failed=true;
        try {
            failOriginalSourceAttempt(device_runners_,retained_source_admission_->cancel,[&](const auto& runner) {
                if (auto* device=dynamic_cast<DeviceGraphOrchestrator*>(runner.get()))
                    device->prepareGroupedRetainedStorageProducerProof(operation);
            },[&] {
                std::exception_ptr failure;
                try { retained_source_admission_->original->cancelStorageChannelsAfterSourceProof(); }
                catch (...) { failure=std::current_exception(); }
                // Sources are proven and exact owner waits released before
                // awaiting the existing independent collective-abort worker.
                // Channels remain admitted until the later full retirement.
                try { retireRetainedSourceFailureCallback(); }
                catch (...) { if (!failure) failure=std::current_exception(); }
                if (failure) std::rethrow_exception(failure);
            });
        } catch (const RetiredStorageRequestFailure& failure) {
            // Every native producer proof completed and exact storage release
            // was attempted. Keep the failed request owned for its ordinary
            // final group/native retirement; never turn it into reusable state.
            LOG_ERROR("[RankOrchestrator] Failed source attempt preserved retired service failure operation="
                << operation << " reason=" << failure.what());
        } catch (const std::exception& failure) {
            LOG_ERROR("[RankOrchestrator] Failed source attempt has no complete original producer proof operation="
                << operation << " reason=" << failure.what());
            std::terminate();
        } catch (...) {
            LOG_ERROR("[RankOrchestrator] Failed source attempt has no complete original producer proof operation="
                << operation << " reason=non-standard original retirement failure");
            std::terminate();
        }
    }

    void RankOrchestrator::prepareRetainedSourceAdmission()
    {
        using Boundary=qwen4exp::ImmutableLookupExecutionBoundary;
        std::vector<Boundary::SourceAdmissionParticipant> participants;
        for(const auto& runner:device_runners_){
            const auto* device=dynamic_cast<const DeviceGraphOrchestrator*>(runner.get());
            if(!device)continue;
            const auto [declared,epoch]=device->retainedStorageSourceAdmission();
            const auto boundary=std::dynamic_pointer_cast<Boundary>(declared);
            if(boundary&&boundary->requiresSourceGroupAdmission())participants.push_back({boundary,epoch});
        }
        if(participants.empty())return;
        if (retained_source_admission_ && retained_source_admission_->retired_failure)
            std::rethrow_exception(retained_source_admission_->retired_failure);
        if (retained_source_admission_ && retained_source_admission_->failed)
            throw std::logic_error("Failed original source group requires terminal retirement before another launch");
        if(retained_source_admission_&&retained_source_admission_->original->reuses(participants))return;
        if(retained_source_admission_){
            // A different epoch/member set cannot replace a still-owned
            // invocation merely because its CPU worker can be joined. Only
            // the original terminal/reset aggregation retires native channels.
            retained_source_admission_->failed = true;
            retained_source_admission_->cancel();
            throw std::logic_error("Original source group requires full native retirement before replacement");
        }
        auto retained=std::make_unique<RetainedSourceAdmission>();
        retained->original=Boundary::armSourceGroup(participants);
        retained->cancel = retained->original->failureCallback();
        // Bind the independent failure worker before any participant can launch.
        // The callback owns the original source/ticket, never this mutable slot.
        if (!tp_worker_pool_)
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
        tp_worker_pool_->setFailureCallback([cancel = retained->cancel, context = tp_ctx_.get()] {
            std::exception_ptr failure;
            try { cancel(); } catch (...) { failure = std::current_exception(); }
            // A source failure must still unblock original TP rendezvous even
            // when cancellation itself discovers an earlier ownership fault.
            if (context) context->requestAbort();
            if (failure) std::rethrow_exception(failure);
        });
        if (rank_local_expert_endpoints_)
            rank_local_expert_endpoints_->bindOriginalPooledSourceCancellation(retained->cancel);
        retained_source_admission_=std::move(retained);
    }

    void RankOrchestrator::awaitRetainedSourceAdmission(std::exception_ptr prior_unproven)
    {
        if (retained_source_admission_ && retained_source_admission_->retired_failure)
            std::rethrow_exception(retained_source_admission_->retired_failure);
        if (!retained_source_admission_) return;
        try {
            retireOriginalStorageParticipants(device_runners_, [](const auto &runner) {
                if (auto *device = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                    device->awaitGroupedRetainedStorageTerminal();
            }, [&] {
                retireRetainedSourceFailureCallback();
                const bool success = retained_source_admission_->original->finish();
                if (!success)
                    throw RetiredStorageRequestFailure("Original TP source group failed terminal request retirement");
            }, [&] {
                // The actual aggregation owner calls this only after every
                // original GPU/native proof, including semantic-failure peers.
                // A lost cancellation identity also forbids host channel
                // retirement, but never skips another original peer's join.
                if (prior_unproven) std::rethrow_exception(prior_unproven);
                if (!retained_source_admission_->original->finishNativeChannelsAfterProof(retained_source_admission_->failed))
                    throw RetiredStorageRequestFailure("Original shared reader channels retired a cancelled request");
            });
            if (rank_local_expert_endpoints_)
                rank_local_expert_endpoints_->clearOriginalPooledSourceCancellation();
            retained_source_admission_.reset();
        } catch (const RetiredStorageRequestFailure &) {
            // The aggregation's typed semantic result comes only after every
            // original native proof and exact channel retirement was attempted.
            if (rank_local_expert_endpoints_)
                rank_local_expert_endpoints_->clearOriginalPooledSourceCancellation();
            if (!retained_source_admission_)
                throw std::logic_error("Retired failed source group lost its original owner");
            retained_source_admission_->retired_failure = std::current_exception();
            throw;
        } catch (...) {
            if (prior_unproven) std::rethrow_exception(prior_unproven);
            throw;
        }
    }

    RankOrchestrator::~RankOrchestrator()
    {
        // Complete parents borrow the stages' events, streams, arenas and native
        // child recordings. Retire them before any stage destructor, but retain
        // the collective/edge owner until those original child graphs retire.
        if (pp_device_generation_)
            pp_device_generation_->retireForwardOnlyBeforeStageRelease();
        retained_source_admission_.reset();
        retireChildTopologyBeforeSharedStorageDetach();
    }

    void RankOrchestrator::retireChildTopologyBeforeSharedStorageDetach()
    {
        // The original follower parent owns finite storage siblings. Close
        // those exact native/cache borrows before any continuation source cache
        // revokes its preparation token or releases its original arena.
        if (rank_local_expert_endpoints_)
            rank_local_expert_endpoints_->retireOriginalPooledStorageAttachments();
        // Keep EVERY arena and original collective context alive until all
        // child caches/auxiliary parents have released their native borrowers.
        for (const auto& runner : device_runners_)
            if (auto* device = dynamic_cast<DeviceGraphOrchestrator*>(runner.get()))
                device->retireExecutionTopologyBeforeSharedStorageDetach();
        if (auto model = std::dynamic_pointer_cast<ModelContext>(model_ctx_)) {
            std::vector<std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants>> originals;
            {
                std::lock_guard lock(model->immutable_row_table_mutex_);
                originals = originalPublishedStorageParticipants(model->adaptive_lookup_consumers_,
                    [&](DeviceId device, const auto& original) {
                        // A throwing DGO constructor never appears in device_runners_.
                        // Its already-published shared source still borrows this exact
                        // Rank communicator and must retire before tp_ctx_ unwinds.
                        if (original->belongsToOriginalCohort(replicated_ple_cohort_)) return true;
                        return std::any_of(device_runners_.begin(), device_runners_.end(), [&](const auto& runner) {
                            const auto* child = dynamic_cast<const DeviceGraphOrchestrator*>(runner.get());
                            return child && child->state_.device_id == device;
                        });
                    });
            }
            for (const auto& original : originals)
                model->retireAdaptiveImmutableLookupParticipants(original);
        }
    }

    void RankOrchestrator::initializeRankMTPScratch()
    {
        rank_max_draft_depth_ = resolveMTPMaximumDraftDepth(config_.mtp);
        rank_max_verifier_rows_ = rank_max_draft_depth_ + 1;
        rank_compact_output_token_stride_ = rank_max_verifier_rows_;

        /*
         * Target slots include one bonus row per request. Reusing that larger
         * capacity for the rank's draft-token bookkeeping costs only a handful
         * of integers and keeps every slot validation on one explicit bound.
         */
        rank_stochastic_slot_capacity_ =
            std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp));

        rank_compact_output_tokens_.assign(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        const size_t distribution_entry_capacity =
            static_cast<size_t>(rank_stochastic_slot_capacity_) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        rank_stochastic_target_token_ids_.assign(
            distribution_entry_capacity,
            -1);
        rank_stochastic_target_probs_.assign(
            distribution_entry_capacity,
            0.0f);
        rank_stochastic_target_top_k_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            0);
        rank_stochastic_target_sample_tokens_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            -1);
        rank_stochastic_draft_sample_tokens_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            -1);
        rank_mirrored_target_distribution_ready_.assign(
            static_cast<size_t>(rank_stochastic_slot_capacity_),
            false);
    }

    // =========================================================================
    // Private Methods
    // =========================================================================

    void RankOrchestrator::applyRuntimeSnapshotShardingOverrides()
    {
        if (!model_ctx_ || !tp_ctx_)
            return;

        if (model_ctx_->headCountKV() >= tp_ctx_->degree())
            return;

        /*
         * GQA models with fewer KV heads than TP participants replicate the K/V
         * stream. Keep every semantic snapshot from projection through the
         * effective attention input in lockstep. In particular, per-head
         * K_NORM is still a full replica even though Q_NORM remains sharded by
         * query head; concatenating those K replicas invents TP-degree times as
         * many logical KV columns as the model owns.
         */
        static constexpr std::array<std::string_view, 10>
            kReplicatedGQAKVSnapshotStageTypes = {
            "K_PROJECTION",
            "V_PROJECTION",
            "K_NORM",
            "K_ROPE",
            "KV_APPEND_SOURCE_K",
            "KV_APPEND_SOURCE_V",
            "KV_CACHE_K",
            "KV_CACHE_V",
            "ATTENTION_EFFECTIVE_K",
            "ATTENTION_EFFECTIVE_V",
        };

        for (const std::string_view stage_type :
             kReplicatedGQAKVSnapshotStageTypes)
        {
            stage_sharding_map_[std::string(stage_type)] =
                SnapshotShardingMode::REPLICATED;
        }

        LOG_DEBUG(
            "RankOrchestrator: runtime GQA snapshot policy treats the complete "
            "K/V checkpoint family as replicated (n_kv_heads="
            << model_ctx_->headCountKV()
            << " < tp_degree=" << tp_ctx_->degree() << ")");
    }

    bool RankOrchestrator::ownsMainForwardLogits() const noexcept
    {
        return !config_.nested_pp_stage_config.has_value() ||
               config_.nested_pp_stage_config->has_lm_head;
    }

    void RankOrchestrator::initializeDeviceRunners() try
    {
        if (!tp_ctx_)
        {
            throw std::runtime_error("Cannot initialize device runners: tp_ctx_ is null");
        }

        const auto &devices = tp_ctx_->devices();
        const bool replicated_consumer=config_.ngram_table_placement &&
            Qwen4ExpReplicatedPLECohort::requiresReplicatedQuery(*config_.ngram_table_placement) &&
            std::any_of(devices.begin(),devices.end(),[&](const auto &member){
                const auto owner=PhysicalMemoryAllocatorIdentity{config_.ngram_table_placement->worldRank(),member.toLocalDeviceId()};
                const auto &consumers=config_.ngram_table_placement->consumers();
                return std::find(consumers.begin(),consumers.end(),owner)!=consumers.end();
            });
        if(replicated_consumer)
        {
            const auto loader=model_ctx_?model_ctx_->loader():nullptr;
            if(!loader || !loader->ggufDirectory())
                throw std::invalid_argument("Fixed-head PLE Rank requires original source metadata before child construction");
            const auto context=std::dynamic_pointer_cast<ModelContext>(model_ctx_);
            const auto memory=context && context->concreteWeightManager()
                ? context->concreteWeightManager()->physicalMemoryAuthority() : nullptr;
            if(!memory || memory->worldRank()!=config_.ngram_table_placement->worldRank())
                throw std::invalid_argument("Fixed-head PLE requires its actual rank-local admitted memory authority");
            const Qwen4ExpConfig source(*loader->ggufDirectory());
            // The first serving slice uses this Rank's complete mirrored
            // terminal sampler. A nonterminal PP(TP) group needs a distinct
            // authoritative PP-token producer handoff, not a replication bit.
            if(config_.nested_pp_stage_config &&
                (!config_.nested_pp_stage_config->has_embedding || !config_.nested_pp_stage_config->has_lm_head ||
                 config_.nested_pp_stage_config->first_layer!=0 ||
                 config_.nested_pp_stage_config->last_layer!=static_cast<int>(source.mainLayers())))
                throw std::invalid_argument("Fixed-head PLE currently requires full-main local TP; PP token producer handoff is not implemented");
            replicated_ple_cohort_=std::shared_ptr<const Qwen4ExpReplicatedPLECohort>(
                new Qwen4ExpReplicatedPLECohort(*tp_ctx_,*model_ctx_,config_.ngram_table_placement,source));
        }
        device_runners_.reserve(devices.size());

        LOG_DEBUG("RankOrchestrator: Initializing " << devices.size() << " device runners");
        LOG_DEBUG(
            "RankOrchestrator: continuation route transport candidate"
            << " overlay_plan="
            << (config_.moe_routed_expert_plan != nullptr)
            << " devices=" << devices.size()
            << " preinstalled="
            << (config_.moe_node_local_route_exchange != nullptr)
            << " requested_prefill_transport="
            << moeOverlayNodeLocalRouteTransportName(
                   config_.moe_node_local_route_transport_policy
                       .ordinary_prefill)
            << " requested_decode_transport="
            << moeOverlayNodeLocalRouteTransportName(
                   config_.moe_node_local_route_transport_policy.decode)
            << " dense_policy="
            << (config_.moe_routed_expert_plan
                    ? denseParallelPolicyToString(
                          config_.moe_routed_expert_plan
                              ->continuation_domain_spec
                              .effectiveDensePolicy())
                    : "none")
            << " dense_tp_enabled="
            << (config_.moe_routed_expert_plan &&
                denseParallelPolicyEnablesTP(
                    config_.moe_routed_expert_plan
                        ->continuation_domain_spec
                        .effectiveDensePolicy())));

        /*
         * Routed-expert ownership is independent of dense tensor parallelism.
         * Every multi-device continuation therefore resolves one explicit
         * publication transport even when dense weights are replicated. Match
         * the declaration by device identity rather than vendor, ordinal,
         * socket, rank, or declaration order; auxiliary expert ranks do not
         * match and consequently own no continuation-local transport.
         */
        if (config_.moe_routed_expert_plan && devices.size() > 1u)
        {
            const auto &overlay_plan = *config_.moe_routed_expert_plan;
            const auto continuation_domain = std::find_if(
                overlay_plan.domains.begin(), overlay_plan.domains.end(),
                [&](const RoutedExpertDomain &domain)
                {
                    return domain.name ==
                           overlay_plan.continuation_domain_spec.domain;
                });
            LOG_DEBUG(
                "RankOrchestrator: continuation route domain lookup"
                << " requested="
                << overlay_plan.continuation_domain_spec.domain
                << " found="
                << (continuation_domain != overlay_plan.domains.end())
                << " domain_participants="
                << (continuation_domain != overlay_plan.domains.end()
                        ? continuation_domain->participants.size()
                        : 0u)
                << " cell_participants=" << devices.size());
            if (continuation_domain != overlay_plan.domains.end() &&
                continuation_domain->participants.size() == devices.size())
            {
                std::vector<DeviceId> cell_devices;
                std::vector<std::string> cell_keys;
                std::vector<std::string> continuation_keys;
                cell_devices.reserve(devices.size());
                cell_keys.reserve(devices.size());
                continuation_keys.reserve(devices.size());
                bool all_gpu = true;
                for (const auto &address : devices)
                {
                    const DeviceId local = address.toLocalDeviceId();
                    all_gpu = all_gpu && local.is_gpu();
                    cell_devices.push_back(local);
                    cell_keys.push_back(local.toString());
                }
                for (const auto &address : continuation_domain->participants)
                    continuation_keys.push_back(
                        address.toLocalDeviceId().toString());
                std::sort(cell_keys.begin(), cell_keys.end());
                std::sort(
                    continuation_keys.begin(), continuation_keys.end());

                LOG_DEBUG(
                    "RankOrchestrator: continuation route topology match"
                    << " domain=" << continuation_domain->name
                    << " all_gpu=" << all_gpu
                    << " cell_devices=" << cell_keys.size()
                    << " continuation_devices="
                    << continuation_keys.size()
                    << " identity_match="
                    << (cell_keys == continuation_keys));

                const int root_index = overlay_plan
                                           .continuation_domain_spec
                                           .logical_root_participant;
                if (all_gpu && cell_keys == continuation_keys &&
                    root_index >= 0 &&
                    root_index < static_cast<int>(
                                     continuation_domain->participants.size()))
                {
                    const bool homogeneous = std::all_of(cell_devices.begin(), cell_devices.end(),
                        [&](const auto& device) { return device.type == cell_devices.front().type; });
                    if (!config_.moe_node_local_route_transport_policy.unresolved() &&
                        !config_.moe_node_local_route_transport_policy.resolved())
                        throw std::invalid_argument("Rank continuation route transport policy is only partially resolved");
                    if (!overlay_plan.observed_route_transport || !overlay_plan.observed_route_transport->resolved())
                        throw std::invalid_argument("Rank continuation route requires its original hardware-bound transport policy");
                    const auto selected_policy = *overlay_plan.observed_route_transport;
                    if (config_.moe_node_local_route_transport_policy.resolved() &&
                        config_.moe_node_local_route_transport_policy != selected_policy)
                        throw std::invalid_argument("Rank continuation route changed its admitted original transport policy");
                    config_.moe_node_local_route_transport_policy =
                        selected_policy;

                    const DeviceId root_device =
                        continuation_domain
                            ->participants[static_cast<std::size_t>(root_index)]
                            .toLocalDeviceId();

                    if (selected_policy.requiresMappedExchange())
                    {
                        {
                            const auto model = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
                            const auto memory = model && model->concreteWeightManager()
                                ? model->concreteWeightManager()->physicalMemoryAuthority() : nullptr;
                            const auto mpi = config_.moe_expert_overlay_mpi_ctx;
                            const auto inventory = mpi ? mpi->clusterInventory() : nullptr;
                            if (!model || !memory || !inventory)
                                throw std::invalid_argument("Mapped continuation requires its original admitted model and MPI inventory");
                            const auto profile = ModelMemoryProfile::fromGGUF(model->concreteLoader().getModel());
                            const auto rows = OrdinaryTerminalProjectionCapacity::checkedFlattenedRows(config_.batch_size,
                                config_.resident_graph_rows > 0 ? config_.resident_graph_rows :
                                resolveActivationBufferSeqLen(config_.max_seq_len, root_device));
                            MoEOverlayNodeLocalRouteExchange::Config topology{
                                .devices = cell_devices, .root_device = root_device,
                                .identity = "continuation:" + continuation_domain->name};
                            auto storage = MoEOverlayNodeLocalRouteExchange::declareStorage(
                                topology, rows, profile.expert_used_count, profile.d_model);
                            if (!config_.moe_node_local_route_exchange)
                                config_.moe_node_local_route_exchange = std::make_shared<MoEOverlayNodeLocalRouteExchange>(
                                    std::move(topology), storage, memory, inventory);
                            else config_.moe_node_local_route_exchange->requireOriginalAdmission(storage, memory, inventory);
                        }
                    }
                    else if (config_.moe_node_local_route_exchange)
                    {
                        throw std::runtime_error(
                            "RankOrchestrator: all-native continuation route transport policy cannot retain a mapped sparse exchange");
                    }

                    const std::string peer_access = homogeneous ? "original_observed" : "heterogeneous";
                    const auto record_transport =
                        [&](std::string phase,
                            MoEOverlayNodeLocalRouteTransport transport)
                        {
                            const PerfStatsCollector::Tags transport_tags{
                                {"domain", continuation_domain->name},
                                {"phase", std::move(phase)},
                                {"transport",
                                 moeOverlayNodeLocalRouteTransportName(
                                     transport)},
                                {"peer_access", peer_access},
                                {"backend",
                                 homogeneous
                                     ? (cell_devices.front().is_cuda()
                                            ? "cuda"
                                            : "rocm")
                                     : "heterogeneous"},
                                {"participants",
                                 std::to_string(cell_devices.size())}};
                            PerfStatsCollector::addCounter(
                                "moe_overlay_transport",
                                "selection",
                                1.0,
                                {},
                                root_device.toString(),
                                transport_tags);
                        };
                    record_transport(
                        "ordinary_prefill", selected_policy.ordinary_prefill);
                    record_transport("decode", selected_policy.decode);
                    LOG_INFO(
                        "RankOrchestrator: selected continuation route "
                        "transport policy prefill="
                        << moeOverlayNodeLocalRouteTransportName(
                               selected_policy.ordinary_prefill)
                        << " decode="
                        << moeOverlayNodeLocalRouteTransportName(
                               selected_policy.decode)
                        << " peer_access=" << peer_access << " domain="
                        << continuation_domain->name
                        << " root=" << root_device.toString()
                        << " participants=" << devices.size());
                }
            }
        }

        // =====================================================================
        // BUILD WEIGHTMANAGERCONFIG FOR LOCAL TP WEIGHT SHARDING
        // =====================================================================
        // Configure WeightManager with TP config, model dimensions, and sharding
        // in a single configure() call. This replaces the previous multi-setter chain.
        // =====================================================================
        {
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr && tp_ctx_)
            {
                // Get model dimensions
                int n_heads = model_ctx_->headCount();
                int n_kv_heads = model_ctx_->headCountKV();
                int d_ff = model_ctx_->feedForwardLength();
                int vocab_size = model_ctx_->vocabSize();

                // feedForwardLength must be available from model metadata for correct TP sharding
                if (d_ff <= 0)
                {
                    throw std::runtime_error(
                        "RankOrchestrator: feedForwardLength() unavailable from model context. "
                        "This value is required for correct tensor parallel weight sharding. "
                        "Ensure the GGUF model contains the feed_forward_length metadata field.");
                }

                const bool dense_tp_enabled =
                    moeOverlayDenseTPEnabled(config_.moe_routed_expert_plan);

                std::shared_ptr<TensorParallelConfig> tp_config;
                if (dense_tp_enabled)
                {
                    if(model_ctx_->architecture()=="qwen4exp")
                    {
                        const auto *directory=model_ctx_->ggufDirectory();
                        if(!directory)throw std::invalid_argument("QFN TP weight assignment lacks original source directory");
                        const Qwen4ExpConfig original(*directory);
                        std::vector<DeviceId> members;
                        for(const auto &address:tp_ctx_->devices())members.push_back(address.toLocalDeviceId());
                        const auto dense=config_.moe_routed_expert_plan
                            ?config_.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy():DenseParallelPolicy::TensorParallel;
                        const auto source=resolveQfnTPDeclaration(*directory,
                            mode_==ParallelismMode::TP && config_.pp_stages.empty() &&
                                (!config_.moe_routed_expert_plan || config_.moe_routed_expert_plan->domains.size()==1)
                                ?TensorParallelMainPlacement::FullMainRankLocal:TensorParallelMainPlacement::OtherDeclared,
                            config_.mtp,dense,members);
                        tp_config=std::make_shared<TensorParallelConfig>(source.partition(members,tp_ctx_->weights(),d_ff,vocab_size));
                    }
                    else
                        tp_config = std::make_shared<TensorParallelConfig>(
                            TensorParallelConfig::fromLocalTPContext(
                                *tp_ctx_, n_heads, n_kv_heads, d_ff, vocab_size));
                }

                if (tp_config && usesNativeSharedExpertTP(config_.moe_routed_expert_plan.get()))
                {
                    // The sealed loader owns both selected source directories
                    // and whether the paired predictor can be materialized.
                    *tp_config = bindOriginalSharedExpertTP(
                        ModelMemoryProfile::fromOriginalModelContext(*model_ctx_), config_.mtp, *tp_config);
                }

                // Build unified WeightManagerConfig
                WeightManagerConfig wm_config;
                wm_config.tp_config = dense_tp_enabled ? tp_config : nullptr;
                wm_config.sharding = SchemaFactoryRegistry::getWeightShardingConfig(model_ctx_->architecture());
                if (routedOverlayUsesExpertIdApportionment(config_.moe_routed_expert_plan))
                {
                    forceRoutedMoEExpertParentsReplicated(wm_config.sharding);
                    LOG_DEBUG("RankOrchestrator: forcing routed MoE expert parent weights replicated for graph-native overlay");
                }

                // Model head dimensions for FusedQKV sub-block slicing
                const int embed_len = model_ctx_->embeddingLength();
                // Flash Next attention expands H into 24 independent 256-wide
                // heads. Its source key_length, never H/heads, owns slicing.
                const bool explicit_head_geometry = model_ctx_->architecture() == "qwen4exp";
                const int head_dim = explicit_head_geometry ? model_ctx_->keyLength()
                    : (n_heads > 0 ? embed_len / n_heads : 0);
                if (explicit_head_geometry && head_dim <= 0)
                    throw std::invalid_argument("qwen4exp TP slicing requires explicit source attention.key_length");
                if (n_heads > 0 && head_dim > 0)
                {
                    wm_config.dimensions.n_heads = n_heads;
                    wm_config.dimensions.n_kv_heads = n_kv_heads;
                    wm_config.dimensions.head_dim = head_dim;
                }

                // GDN dimensions from GGUF metadata for asymmetric FusedQKV
                auto *concrete_ctx = dynamic_cast<ModelContext *>(model_ctx_.get());
                if (concrete_ctx)
                {
                    const std::string arch_prefix = model_ctx_->architecture();
                    ModelLoader &loader = concrete_ctx->concreteLoader();
                    const int gdn_group_count = loader.getInt(arch_prefix + ".ssm.group_count", 0);
                    const int gdn_time_step_rank = loader.getInt(arch_prefix + ".ssm.time_step_rank", 0);
                    const int gdn_state_size = loader.getInt(arch_prefix + ".ssm.state_size", 0);

                    if (gdn_group_count > 0 && gdn_time_step_rank > 0 && gdn_state_size > 0)
                    {
                        wm_config.dimensions.gdn_n_k_heads = gdn_group_count;
                        wm_config.dimensions.gdn_n_v_heads = gdn_time_step_rank;
                        wm_config.dimensions.gdn_d_state = gdn_state_size;
                        LOG_DEBUG("RankOrchestrator: GDN dimensions for FusedQKV slicing"
                                  << " (group_count=" << gdn_group_count
                                  << " time_step_rank=" << gdn_time_step_rank
                                  << " state_size=" << gdn_state_size << ")");
                    }
                }

                // Apply all configuration in a single call
                weight_mgr->configure(wm_config);

                LOG_DEBUG("RankOrchestrator: Configured WeightManager for LOCAL TP ("
                          << tp_ctx_->degree() << " devices, "
                          << "heads=" << n_heads << ", kv_heads=" << n_kv_heads
                          << ", d_ff=" << d_ff << ", vocab=" << vocab_size
                          << ", dense_tp=" << (dense_tp_enabled ? "true" : "false") << ")");

                // Print per-device assignments for debugging TP parity issues
                for (int dev_idx = 0; dense_tp_enabled && dev_idx < tp_ctx_->degree(); ++dev_idx)
                {
                    const auto &addr = tp_ctx_->devices()[dev_idx];
                    DeviceId dev_id = addr.toLocalDeviceId();
                    try
                    {
                        // Logical TP rank owns the slice. Distinct CPU NUMA
                        // participants intentionally share one CPU address-space
                        // DeviceId, so a device-only lookup is ambiguous.
                        const auto &assignment = tp_config->forRank(dev_idx);
                        LOG_DEBUG("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") assignment:"
                                                              << " head_start=" << assignment.head_start
                                                              << " head_count=" << assignment.head_count
                                                              << " kv_head_start=" << assignment.kv_head_start
                                                              << " kv_head_count=" << assignment.kv_head_count
                                                              << " d_ff_start=" << assignment.d_ff_start
                                                              << " d_ff_count=" << assignment.d_ff_count);
                    }
                    catch (const std::out_of_range &e)
                    {
                        LOG_WARN("RankOrchestrator: Device " << dev_idx << " (" << dev_id.to_string() << ") NOT in TensorParallelConfig!");
                    }
                }

                /*
                 * Runtime snapshot layout was frozen from this same model and
                 * TP topology before device runners were initialized. Do not
                 * maintain a second, partial override list here: it previously
                 * omitted K_NORM and let parity concatenate four GQA replicas.
                 */
            }
        }

        // =====================================================================
        // PRE-RESERVE COLLECTIVE TEMP BUFFER
        // =====================================================================
        // Pre-allocate temp buffer for allreduce operations based on model dimensions
        // and activation precision. This avoids allocation in the hot path.
        // Buffer uses grow-only semantics (never shrinks during inference).
        // =====================================================================
        {
            const size_t hidden_size = model_ctx_->embeddingLength();
            const auto collective_bom =
                CollectiveMemoryEstimator::localTP(
                    config_.max_seq_len,
                    static_cast<int>(hidden_size),
                    tp_ctx_->backend());

            std::shared_ptr<PhysicalMemoryAuthority> memory_authority;
            if (const auto concrete_model =
                    std::dynamic_pointer_cast<ModelContext>(model_ctx_))
            {
                const auto weight_manager =
                    concrete_model->concreteWeightManager();
                memory_authority = weight_manager
                                       ? weight_manager
                                             ->physicalMemoryAuthority()
                                       : nullptr;
            }

            const bool reserved = tp_ctx_->reserveCollectiveResources(
                collective_bom.backend_payload_capacity_bytes,
                collective_bom.fp16_scratch_elements,
                memory_authority);
            logVramBomLine(
                "collective_temp_reservation",
                "source=RankOrchestrator backend=local_tp max_seq_len=" + std::to_string(config_.max_seq_len) +
                    " hidden_size=" + std::to_string(hidden_size) +
                    " precision=" + activationPrecisionToString(config_.activation_precision) +
                    " status=" + (reserved ? "pass" : "fail") +
                    " " + vramBomBytes(
                        collective_bom.perDeviceBytes()));
            if (reserved)
            {
                LOG_DEBUG("RankOrchestrator: Reserved collective resources: "
                          << "logical_payload_capacity="
                          << collective_bom.backend_payload_capacity_bytes
                          << " bytes, physical_fp16_scratch="
                          << collective_bom.fp16_scratch_bytes << " bytes ("
                          << "max_seq_len=" << config_.max_seq_len
                          << ", hidden_size=" << hidden_size
                          << ", precision=" << activationPrecisionToString(config_.activation_precision) << ")");
            }
            else
            {
                throw std::runtime_error(
                    "RankOrchestrator: failed to reserve complete LocalTP "
                    "collective resources");
            }
            if (config_.moe_routed_expert_plan)
            {
                const auto &plan = *config_.moe_routed_expert_plan;
                const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                    [&](const auto &candidate) { return candidate.name == plan.continuation_domain; });
                if (domain != plan.domains.end() &&
                    domain->routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
                {
                    std::vector<DeviceId> members;
                    for (const auto &address : devices) members.push_back(address.toLocalDeviceId());
                    const auto original_mpi = config_.moe_expert_overlay_mpi_ctx;
                    const auto original_inventory = original_mpi ? original_mpi->clusterInventory() : nullptr;
                    const auto model = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
                    if (!original_inventory || !model || !memory_authority)
                        throw std::logic_error("Projection transport requires its original rank inventory and admitted model/physical authority");
                    const auto &observed = original_inventory->getRank(memory_authority->worldRank());
                    const auto topology = DeviceCountedAllGatherTopology::resolve(observed, members);
                    const auto arena = MoEProjectionArenaGeometry::resolve(
                        ModelMemoryProfile::fromGGUF(model->concreteLoader().getModel()), members.size());
                    const int rows = std::max(config_.resident_graph_rows > 0
                        ? std::min(static_cast<int>(config_.max_seq_len), config_.resident_graph_rows)
                        : resolveActivationBufferSeqLen(config_.max_seq_len, members.front()),
                        retainsMTPGraphCapacity(config_.mtp) ? resolveMTPRetainedTargetQueryRows(config_.mtp) : 1);
                    std::vector<void *> streams(members.size());
                    for (std::size_t i = 0; i < members.size(); ++i)
                    {
                        auto &worker = GPUDeviceContextPool::instance().getContext(members[i]);
                        worker.submitAndWait([&] { streams[i] = worker.getOrCreateAuxiliaryStream("projection_exchange_setup"); });
                    }
                    tp_ctx_->installDeviceCountedAllGather(DeviceCountedAllGather::create(
                        *memory_authority, *tp_ctx_, DeviceId::cpu(), topology, streams, arena.packetCapacityBytes(rows)));
                    LOG_INFO("RankOrchestrator: installed device-counted projection exchange: participants="
                        << members.size() << " max_packet_bytes=" << arena.packetCapacityBytes(rows));
                }
            }
        }

        // Install one model-owned prepared store before participant graphs are
        // built concurrently. Each factory materializes and prepares its frozen
        // WeightPlan, whose main/MTP/overlay bindings are the admitted consumers.
        // A broad tensor-directory preload would pack unused NextN blocks when
        // MTP is disabled, and cannot express replicated sidecar/expert ownership.
        // Per-device cache keys and immutable frozen bindings own tensor identity;
        // preloading raw tensors is neither a race barrier nor an admission plan.
        {
            auto weight_mgr = model_ctx_->weightManager();
            if (weight_mgr)
            {
                if (auto concrete_weight_mgr = std::dynamic_pointer_cast<WeightManager>(weight_mgr))
                {
                    const auto installed_store =
                        concrete_weight_mgr->preparedWeightStoreIfInitialized();
                    if (config_.prepared_weight_store)
                    {
                        if (installed_store &&
                            installed_store != config_.prepared_weight_store)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator cannot replace the model-owned "
                                "PreparedWeightStore");
                        }
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                    else if (installed_store)
                    {
                        /*
                         * Prepared storage is additive and model-owned. A new
                         * rank graph must never silently install a second store
                         * and orphan handles whose raw sources may already have
                         * been released.
                         */
                        config_.prepared_weight_store = installed_store;
                    }
                    else
                    {
                        config_.prepared_weight_store = std::make_shared<PreparedWeightStore>(
                            ModelContextId{reinterpret_cast<uint64_t>(model_ctx_.get())});
                        concrete_weight_mgr->setPreparedWeightStore(config_.prepared_weight_store);
                    }
                }
                std::vector<DeviceId> device_ids;
                for (const auto &device_addr : devices)
                    device_ids.push_back(device_addr.toLocalDeviceId());
                if (config_.prepared_weight_admission ==
                    PreparedWeightAdmission::ReuseCertifiedCompleteSet)
                {
                    auto concrete_weight_mgr =
                        std::dynamic_pointer_cast<WeightManager>(weight_mgr);
                    if (!concrete_weight_mgr ||
                        !config_.prepared_weight_store)
                    {
                        throw std::runtime_error(
                            "RankOrchestrator certified prepared-weight "
                            "reuse lacks its concrete model authority");
                    }
                    for (const DeviceId device : device_ids)
                    {
                        if (concrete_weight_mgr
                                ->preparedRecordCountForDevice(device) == 0u)
                        {
                            throw std::runtime_error(
                                "RankOrchestrator certified prepared-weight "
                                "reuse has no records for " +
                                device.toString());
                        }
                    }
                    PerfStatsCollector::addCounter(
                        "weight_loading",
                        "rank_prepared_weight_materialization_reuses",
                        1.0,
                        "load",
                        {},
                        {{"devices", std::to_string(device_ids.size())}});
                }
            }
        }

        // =====================================================================
        // Create device runners in parallel
        // =====================================================================
        // GPU weights remain independent. A colocated CPU expert tier is the
        // one shared preparation dependency: the continuation root publishes
        // its completed CPU bank before sibling graphs resolve those engines.
        // No setup gate survives into request execution.
        // =====================================================================

        std::shared_ptr<MoEOverlayCpuPreparationGate> cpu_preparation_gate;
        DeviceId cpu_preparation_publisher = DeviceId::invalid();
        if (config_.moe_routed_expert_plan &&
            config_.moe_routed_expert_plan->usesExpertOverlayAuthority() &&
            devices.size() > 1u)
        {
            const int world_rank = config_.moe_expert_overlay_mpi_ctx
                                       ? config_.moe_expert_overlay_mpi_ctx->rank()
                                       : 0;
            const auto runtime_plan = resolveMoEExpertOverlayRuntimePlan(
                config_.moe_routed_expert_plan,
                MoEExpertOverlayRuntimeResolverOptions{
                    .current_world_rank = world_rank,
                });
            bool has_colocated_cpu_bank = false;
            for (size_t tier = 0; tier < runtime_plan->routedTiers().size(); ++tier)
            {
                const auto &domain = runtime_plan->domainForTier(tier);
                has_colocated_cpu_bank = has_colocated_cpu_bank ||
                    std::any_of(domain.participants.begin(), domain.participants.end(),
                                [](const MoEOverlayDomainParticipant &participant)
                                {
                                    return participant.owned_by_current_rank &&
                                           participant.local_device.is_cpu();
                                });
            }
            if (has_colocated_cpu_bank && runtime_plan->continuationDevice().is_gpu())
            {
                cpu_preparation_publisher = runtime_plan->continuationDevice();
                const size_t publishers = static_cast<size_t>(std::count_if(
                    devices.begin(), devices.end(),
                    [&](const GlobalDeviceAddress &address)
                    { return address.toLocalDeviceId() == cpu_preparation_publisher; }));
                if (publishers != 1u)
                {
                    throw std::runtime_error(
                        "RankOrchestrator colocated CPU ExpertOverlay bank lacks exactly one continuation-root GPU publisher");
                }
                cpu_preparation_gate =
                    std::make_shared<MoEOverlayCpuPreparationGate>();
            }
        }

        struct RunnerResult
        {
            std::unique_ptr<IInferenceRunner> runner;
            int device_idx;
            std::string error;
            std::exception_ptr failure;
        };

        const int num_devices = static_cast<int>(devices.size());
        std::vector<std::future<RunnerResult>> futures;
        futures.reserve(num_devices);
        std::exception_ptr constructor_failure;

        for (int device_idx = 0; device_idx < num_devices; ++device_idx)
        {
            const auto &device_addr = devices[device_idx];
            DeviceId device_id = device_addr.toLocalDeviceId();

            try
            {
                futures.push_back(std::async(std::launch::async,
                                         [this, device_idx, device_id,
                                          cpu_preparation_gate,
                                          cpu_preparation_publisher]() -> RunnerResult
                                         {
                                             RunnerResult result;
                                             result.device_idx = device_idx;
                                             std::optional<MoEOverlayCpuPreparationParticipation>
                                                 cpu_preparation;
                                             if (cpu_preparation_gate)
                                             {
                                                 cpu_preparation =
                                                     device_id == cpu_preparation_publisher
                                                         ? MoEOverlayCpuPreparationParticipation::publisher(
                                                               cpu_preparation_gate)
                                                         : MoEOverlayCpuPreparationParticipation::dependent(
                                                               cpu_preparation_gate);
                                             }
                                             try
                                             {
                                                 LOG_DEBUG("RankOrchestrator: Creating runner for device " << device_idx
                                                                                                           << " (" << device_id.toString() << ")");

                                                 // Build InferenceRunnerConfig for LOCAL TP
                                                 InferenceRunnerConfig runner_config;
                                                 runner_config.max_seq_len = static_cast<int>(config_.max_seq_len);
                                                 runner_config.activation_seq_len = config_.resident_graph_rows;
                                                 runner_config.attention_participants =
                                                     config_.attention_participants;
                                                 runner_config.cpu_serial_expert_storage = config_.cpu_serial_expert_storage;
                                                 runner_config.batch_size = config_.batch_size;
                                                 runner_config.ordinary_terminal_projection = config_.ordinary_terminal_projection;
                                                 runner_config.activation_precision = config_.activation_precision;
                                                 runner_config.fused_attention_backend = config_.fused_attention_backend;
                                                 runner_config.kv_cache_scale_k = config_.kv_cache_scale_k;
                                                 runner_config.kv_cache_scale_v = config_.kv_cache_scale_v;
                                                 runner_config.kv_cache_precision = config_.kv_cache_precision;
                                                 runner_config.tp_allreduce_precision_override =
                                                     config_.tp_allreduce_precision_override;
                                                 runner_config.prefix_cache = config_.prefix_cache;
                                                 runner_config.ngram_table_policy = config_.ngram_table_policy;
                                                 runner_config.ngram_lookup_schedule = config_.ngram_lookup_schedule;
                                                 runner_config.ngram_table_placement = config_.ngram_table_placement;
                                                 runner_config.retained_host_services = config_.retained_host_services;
                                                 runner_config.ngram_replicated_ple_cohort = replicated_ple_cohort_;
                                                 runner_config.mtp = config_.mtp;
                                                 runner_config.routed_expert_compute_policy = config_.routed_expert_compute_policy;
                                                 runner_config.routed_expert_owner_order = config_.routed_expert_owner_order;
                                                 runner_config.moe_hot_expert_cache = config_.moe_hot_expert_cache;
                                                 runner_config.moe_routed_prefill = config_.moe_routed_prefill;
                                                 runner_config.moe_rebalance = config_.moe_rebalance;
                                                 runner_config.prepared_weight_store = config_.prepared_weight_store;
                                                 runner_config.prepared_weight_admission =
                                                     config_.prepared_weight_admission;
                                                 runner_config.moe_overlay_cpu_preparation =
                                                     cpu_preparation;
                                                 runner_config.reusable_execution_workspaces =
                                                     config_.reusable_execution_workspaces;
                                                 runner_config.moe_routed_expert_plan = config_.moe_routed_expert_plan;
                                                 runner_config.moe_expert_overlay_residency_authority =
                                                     config_.moe_expert_overlay_residency_authority;
                                                 runner_config.moe_expert_overlay_participant_residency =
                                                     config_.moe_expert_overlay_participant_residency;
                                                 runner_config.moe_expert_overlay_cpu_joint_context = config_.moe_expert_overlay_cpu_joint_context;
                                                 runner_config.moe_expert_overlay_decode_histogram =
                                                     config_.moe_expert_overlay_decode_histogram;
                                                 runner_config.moe_expert_overlay_mpi_ctx = config_.moe_expert_overlay_mpi_ctx;
                                                 runner_config.moe_rank_batch_transport_registry =
                                                     config_.moe_rank_batch_transport_registry;
                                                 runner_config.moe_device_controller_fabric =
                                                     config_.moe_device_controller_fabric;
                                                 runner_config.moe_node_local_route_exchange =
                                                     config_.moe_node_local_route_exchange;
                                                 runner_config.moe_node_local_route_transport_policy =
                                                     config_.moe_node_local_route_transport_policy;
                                                 runner_config.cancellation_requested = [this]()
                                                 {
                                                     return tp_ctx_ && tp_ctx_->isAbortRequested();
                                                 };
                                                 runner_config.stage_failure_callback = [this, device_idx](const std::string &stage_name, const std::string &reason)
                                                 {
                                                     LOG_WARN("RankOrchestrator: device " << device_idx
                                                                                          << " stage failure triggers TP abort"
                                                                                          << " stage='" << stage_name << "' reason='" << reason << "'");
                                                     if (tp_ctx_)
                                                         tp_ctx_->requestAbort();
                                                 };

                                                 // Set TP parameters (LOCAL TP context here)
                                                 runner_config.tp_ctx = tp_ctx_.get();
                                                 runner_config.tp_device_index = device_idx;

                                                 // Pass PP stage config for nested TP-in-PP
                                                 if (config_.nested_pp_stage_config.has_value())
                                                 {
                                                     runner_config.pp_stage_config = config_.nested_pp_stage_config;
                                                 }

                                                 // Create the inference runner
                                                 result.runner = createTestableInferenceRunner(model_ctx_, device_id, runner_config, config_.test_runner_overrides);
                                                 if (!result.runner)
                                                 {
                                                     result.error = "Failed to create inference runner for device " +
                                                                    std::to_string(device_idx);
                                                     if (cpu_preparation)
                                                         cpu_preparation->publishFailureIfProducer(
                                                             result.error);
                                                 }
                                             }
                                             catch (const std::exception &e)
                                             {
                                                 result.failure = std::current_exception();
                                                 if (tp_ctx_) tp_ctx_->requestAbort();
                                                 result.error = std::string("Exception creating runner for device ") +
                                                                std::to_string(device_idx) + ": " + e.what();
                                                 if (cpu_preparation)
                                                     cpu_preparation->publishFailureIfProducer(
                                                         result.error);
                                             }
                                             catch (...) {
                                                 if (tp_ctx_) tp_ctx_->requestAbort();
                                                 if (cpu_preparation_gate)
                                                     cpu_preparation_gate->publishFailure("Original TP constructor failed");
                                                 throw;
                                             }
                                             if (!result.error.empty()) {
                                                 if (!result.failure) result.failure = std::make_exception_ptr(std::runtime_error(result.error));
                                                 if (tp_ctx_) tp_ctx_->requestAbort();
                                             }
                                             return result;
                                         }));
            }
            catch (...)
            {
                // A failed async launch must wake any already launched
                // dependent instead of leaving its future blocked forever.
                if (tp_ctx_) tp_ctx_->requestAbort();
                if (cpu_preparation_gate)
                    cpu_preparation_gate->publishFailure(
                        "RankOrchestrator could not launch all device runner constructors");
                constructor_failure = std::current_exception();
                break;
            }
        }

        // Consume ALL constructor futures before propagating the first error.
        // Abort is signalled at discovery, before waiting for any peer barrier.
        std::vector<int> constructed_indices;
        constructed_indices.reserve(futures.size());
        device_runners_.reserve(futures.size());
        constructor_failure = collectOriginalParticipantConstructors(futures,
            [&](RunnerResult result) noexcept {
                if (result.runner) {
                    constructed_indices.push_back(result.device_idx);
                    device_runners_.push_back(std::move(result.runner));
                }
            }, [&] { if (tp_ctx_) tp_ctx_->requestAbort(); }, constructor_failure);
        if (constructor_failure) std::rethrow_exception(constructor_failure);
        for (std::size_t i = 0; i < device_runners_.size(); ++i) {
            auto& runner = device_runners_[i];
            // Cast to DeviceGraphOrchestrator for setPPStageConfig (construction-time only)
            if (config_.nested_pp_stage_config.has_value())
            {
                auto *device_orchestrator = dynamic_cast<DeviceGraphOrchestrator *>(runner.get());
                if (device_orchestrator)
                {
                    device_orchestrator->setHostResidentReleaseEnabled(false);
                    device_orchestrator->setPPStageConfig(config_.nested_pp_stage_config.value());
                    LOG_DEBUG("RankOrchestrator: Set PP stage config on device " << constructed_indices[i]
                                                                                 << " (layers " << config_.nested_pp_stage_config->first_layer
                                                                                 << "-" << config_.nested_pp_stage_config->last_layer
                                                                                 << " has_lm_head=" << config_.nested_pp_stage_config->has_lm_head << ")");
                }
                else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                {
                    dgo->setHostResidentReleaseEnabled(false);
                }
            }
            else if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
            {
                dgo->setHostResidentReleaseEnabled(false);
            }

            LOG_DEBUG("RankOrchestrator: Successfully created runner for device " << constructed_indices[i]);
        }

        // Continuation banks are now published in the same original registry.
        // Prepare auxiliary experts before releasing any remaining source tensors;
        // these devices are genuine expert endpoints, not additional TP members.
        if (config_.rank_local_expert_endpoints)
        {
            const auto &auxiliary = *config_.rank_local_expert_endpoints;
            const auto same_original = [](const auto &left, const auto &right)
            {
                return left.get() == right.get() &&
                    !left.owner_before(right) && !right.owner_before(left);
            };
            if (auxiliary.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary ||
                !same_original(auxiliary.model_context, model_ctx_) ||
                !same_original(auxiliary.placement_plan, config_.moe_routed_expert_plan) ||
                !same_original(auxiliary.residency_authority, config_.moe_expert_overlay_residency_authority) ||
                !same_original(auxiliary.participant_residency, config_.moe_expert_overlay_participant_residency) ||
                !same_original(auxiliary.cpu_joint_context, config_.moe_expert_overlay_cpu_joint_context) ||
                !same_original(auxiliary.rank_batch_transport_registry, config_.moe_rank_batch_transport_registry) ||
                !same_original(auxiliary.device_controller_fabric, config_.moe_device_controller_fabric))
                throw std::invalid_argument("Rank auxiliary experts require the same original model, residency, controller and transport owners");
            rank_local_expert_endpoints_ = std::make_shared<MoEOverlayParticipantGraphRunner>(auxiliary);
        }

        // =====================================================================
        // SYNCHRONOUS HOST WEIGHT RELEASE (Phase 9 final)
        // =====================================================================
        // All expert GEMM preparation is resolved before host release: GPU paths
        // consume the unified ExpertGemmRegistry and CPU paths prepare inline.
        // There is no lazy graph building path.
        // Dynamic MoE rebalancing uses serialized transfer blobs, not raw host data.
        //
        // Mark graph_materialization_complete and release host data immediately.
        // =====================================================================
        {
            const bool is_nested_tp_in_pp = config_.nested_pp_stage_config.has_value();
            if (!is_nested_tp_in_pp)
            {
                auto weight_mgr = model_ctx_->weightManager();
                if (weight_mgr)
                {
                    weight_mgr->markGraphMaterializationComplete();

                    size_t released = weight_mgr->releaseAllHostWeightData();
                    if (released > 0)
                    {
                        LOG_DEBUG("RankOrchestrator: Host weight release after graph materialization: "
                                  << released << " tensors released");
                    }
                }
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: Retaining host weight data "
                          "(nested TP-in-PP; outer caller will release)");
            }
        }

        // Only a rank whose typed PP role owns the LM head may materialize a
        // host logits aggregate. Non-terminal nested TP ranks publish hidden
        // activations to the outer PP transfer boundary instead.
        if (ownsMainForwardLogits() && model_ctx_ && device_runners_.size() > 0)
        {
            int vocab = vocab_size();
            if (vocab > 0)
            {
                // logits() publishes one terminal row for both decode and
                // prefill. The explicitly requested all-position verifier has
                // its own gatherer; the KV horizon is never a host output axis.
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    vocab,
                    /*max_tokens=*/1,
                    logits_backend_resolver_);

                // Pin the logits buffer for faster D2H DMA
                if (device_runners_.size() > 1)
                {
                    DeviceId primary_dev = device_runners_[0]->primaryDeviceId();
                    if (external_device_backend_access_enabled_ &&
                        primary_dev.is_gpu())
                    {
                        logits_gatherer_->pinForDevice(primary_dev);
                    }
                }
                applyLogitsGatherSkipFlags();
            }
        }

        // =====================================================================
        // REGISTER COMPUTE STREAMS WITH COLLECTIVE BACKEND
        // =====================================================================
        // Enables event-based pre-synchronization in RCCL/NCCL coordinators instead
        // of hipDeviceSynchronize/cudaDeviceSynchronize. Each device's compute stream
        // is passed so the coordinator can do hipEventRecord(compute_stream) +
        // hipStreamWaitEvent(rccl_stream) before collectives — zero host stall.
        // =====================================================================
        if (tp_ctx_ && tp_ctx_->degree() > 1)
        {
            const auto &devices = tp_ctx_->devices();

            auto &pool = GPUDeviceContextPool::instance();
            std::vector<void *> compute_streams;
            compute_streams.reserve(devices.size());

            for (const auto &dev : devices)
            {
                if (dev.device_type == DeviceType::ROCm || dev.device_type == DeviceType::CUDA)
                {
                    compute_streams.push_back(
                        pool.getContext(DeviceId(dev.device_type, dev.device_ordinal)).defaultStream());
                }
                else
                {
                    LOG_WARN("RankOrchestrator: Skipping compute stream registration for non-GPU device "
                             << dev.toString());
                    compute_streams.clear();
                    break;
                }
            }

            if (!compute_streams.empty())
            {
                tp_ctx_->setComputeStreams(compute_streams);
                LOG_DEBUG("RankOrchestrator: Registered " << compute_streams.size()
                                                          << " compute streams for event-based collective sync");
            }
        }

        wireLocalTPMoERuntimeHistogramSyncs();
    }
    catch (...) {
        const auto failure = std::current_exception();
        if (tp_ctx_) tp_ctx_->requestAbort();
        // Every successfully created sibling is retained in device_runners_ by
        // the all-result collection below. No future destroys its arena first.
        try {
            retireChildTopologyBeforeSharedStorageDetach();
        } catch (...) {
            // A constructor exception otherwise unwinds child arenas even when
            // original native retirement is unproven. Preserve both diagnostics
            // and stop BEFORE C++ member cleanup can release those live roots.
            try { std::rethrow_exception(failure); }
            catch (const std::exception& error) { LOG_ERROR("Original TP construction failure: " << error.what()); }
            catch (...) { LOG_ERROR("Original TP construction raised a non-standard exception"); }
            LOG_ERROR("Original TP construction cleanup could not prove every native parent retired");
            std::terminate();
        }
        std::rethrow_exception(failure);
    }

    // =========================================================================
    // PP Mode Initialization
    // =========================================================================

    void RankOrchestrator::initializePPDeviceRunners()
    {
        if (config_.pp_stages.empty())
        {
            throw std::runtime_error("Cannot initialize PP device runners: no PP stages configured");
        }

        LOG_DEBUG("RankOrchestrator: Initializing " << config_.pp_stages.size() << " PP stage runners");

        // Cast to concrete ModelContext — needed by createPPStageRunner for concreteLoader() access
        auto concrete_model_ctx = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
        if (!concrete_model_ctx)
        {
            throw std::runtime_error("RankOrchestrator: model_ctx_ must be a concrete ModelContext for PP mode");
        }

        const size_t num_stages = config_.pp_stages.size();

        // =========================================================================
        // Cross-Vendor PP Detection
        // =========================================================================
        // Check if any PP transfer is cross-vendor (ROCm→CUDA or CUDA→ROCm).
        // If so, the source stage's hidden state tensor needs host-staged allocation
        // to enable cross-vendor transfers via HOST backend.
        // =========================================================================
        auto isCrossVendorTransfer = [](const PPStageConfig &src, const PPStageConfig &dst) -> bool
        {
            if (src.stage_devices.empty() || dst.stage_devices.empty())
            {
                return false;
            }
            DeviceId src_dev = src.stage_devices[0].toLocalDeviceId();
            DeviceId dst_dev = dst.stage_devices[0].toLocalDeviceId();

            bool src_cuda = src_dev.is_cuda();
            bool src_rocm = src_dev.is_rocm();
            bool dst_cuda = dst_dev.is_cuda();
            bool dst_rocm = dst_dev.is_rocm();

            return (src_cuda && dst_rocm) || (src_rocm && dst_cuda);
        };

        // Check for cross-vendor transfers for logging purposes
        for (size_t i = 0; i + 1 < num_stages; ++i)
        {
            if (isCrossVendorTransfer(config_.pp_stages[i], config_.pp_stages[i + 1]))
            {
                LOG_DEBUG("RankOrchestrator: PP stage " << i
                                                        << " outputs to cross-vendor stage " << (i + 1)
                                                        << " - will use host-staged transfer");
            }
        }

        // Validate every handoff before the first participant allocates weights.
        // A foreign terminal-stage binding must not leave an earlier GPU stage
        // partially constructed while setup unwinds.
        std::vector<Config> child_configs;
        child_configs.reserve(num_stages);
        for (size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx)
        {
            const auto &stage_config = config_.pp_stages[stage_idx];
            Config child_config = config_.forPipelineStage(stage_idx);
            const auto &factory_pp_config = *child_config.nested_pp_stage_config;
            factory_pp_config.requireValidForModel(*concrete_model_ctx);
            const auto routed_metadata = resolveMoERoutedExpertModelMetadataForModel(
                *concrete_model_ctx, child_config.mtp, factory_pp_config);
            if (routed_metadata.num_experts > 0 && !stage_config.moe_runtime)
                throw std::invalid_argument("Pipeline MoE stage has no admitted expert runtime binding");
            if (stage_config.moe_runtime)
            {
                const auto &bound = stage_config.moe_runtime->metadata();
                if (bound.first_model_layer != routed_metadata.first_model_layer ||
                    bound.num_layers != routed_metadata.num_layers ||
                    bound.num_experts != routed_metadata.num_experts ||
                    bound.main_inference_layer_count != routed_metadata.main_inference_layer_count)
                    throw std::invalid_argument("Pipeline expert binding differs from the loaded model/MTP geometry");
            }

            child_configs.push_back(std::move(child_config));
        }

        pp_stage_runners_.reserve(num_stages);

        for (size_t stage_idx = 0; stage_idx < num_stages; ++stage_idx)
        {
            const auto &stage_config = config_.pp_stages[stage_idx];

            LOG_DEBUG("RankOrchestrator: Creating PP stage " << stage_idx
                                                             << " [layers " << stage_config.first_layer
                                                             << "-" << stage_config.last_layer << ")"
                                                             << " has_embedding=" << stage_config.has_embedding
                                                             << " has_lm_head=" << stage_config.has_lm_head);

            // Validate stage has at least one device
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(stage_idx) + " has no devices configured");
            }

            // Get primary device for this stage
            DeviceId primary_device = stage_config.stage_devices[0].toLocalDeviceId();

            const auto &child_config = child_configs[stage_idx];
            const auto &factory_pp_config = *child_config.nested_pp_stage_config;

            // =====================================================================
            // Build InferenceRunnerConfig for this stage
            // =====================================================================
            InferenceRunnerConfig runner_config;
            runner_config.max_seq_len = static_cast<int>(child_config.max_seq_len);
            runner_config.activation_seq_len = child_config.resident_graph_rows;
            runner_config.attention_participants = child_config.attention_participants;
            runner_config.cpu_serial_expert_storage = child_config.cpu_serial_expert_storage;
            runner_config.batch_size = child_config.batch_size;
            runner_config.ordinary_terminal_projection = child_config.ordinary_terminal_projection;
            runner_config.activation_precision = child_config.activation_precision;
            runner_config.fused_attention_backend = child_config.fused_attention_backend;
            runner_config.kv_cache_scale_k = child_config.kv_cache_scale_k;
            runner_config.kv_cache_scale_v = child_config.kv_cache_scale_v;
            runner_config.kv_cache_precision = child_config.kv_cache_precision;
            runner_config.tp_allreduce_precision_override =
                child_config.tp_allreduce_precision_override;
            runner_config.prefix_cache = child_config.prefix_cache;
            runner_config.ngram_table_policy = child_config.ngram_table_policy;
            runner_config.ngram_lookup_schedule = child_config.ngram_lookup_schedule;
            runner_config.ngram_table_placement = child_config.ngram_table_placement;
            runner_config.retained_host_services = child_config.retained_host_services;
            runner_config.mtp = child_config.mtp;
            runner_config.routed_expert_compute_policy = child_config.routed_expert_compute_policy;
            runner_config.routed_expert_owner_order = child_config.routed_expert_owner_order;
            runner_config.moe_hot_expert_cache = child_config.moe_hot_expert_cache;
            runner_config.moe_routed_prefill = child_config.moe_routed_prefill;
            runner_config.moe_rebalance = child_config.moe_rebalance;
            runner_config.prepared_weight_store = child_config.prepared_weight_store;
            runner_config.prepared_weight_admission =
                child_config.prepared_weight_admission;
            runner_config.reusable_execution_workspaces =
                child_config.reusable_execution_workspaces;
            runner_config.moe_routed_expert_plan = child_config.moe_routed_expert_plan;
            runner_config.moe_expert_overlay_residency_authority =
                child_config.moe_expert_overlay_residency_authority;
            runner_config.moe_expert_overlay_participant_residency =
                child_config.moe_expert_overlay_participant_residency;
            runner_config.moe_expert_overlay_cpu_joint_context = child_config.moe_expert_overlay_cpu_joint_context;
            runner_config.moe_expert_overlay_decode_histogram =
                child_config.moe_expert_overlay_decode_histogram;
            runner_config.moe_expert_overlay_mpi_ctx = child_config.moe_expert_overlay_mpi_ctx;
            runner_config.moe_rank_batch_transport_registry =
                child_config.moe_rank_batch_transport_registry;
            runner_config.moe_device_controller_fabric =
                child_config.moe_device_controller_fabric;
            runner_config.moe_node_local_route_exchange =
                child_config.moe_node_local_route_exchange;
            runner_config.moe_node_local_route_transport_policy =
                child_config.moe_node_local_route_transport_policy;
            // =====================================================================
            // Handle single-device vs TP-domain stages
            // =====================================================================
            if (stage_config.isTPDomain())
            {
                // =====================================================================
                // TP Domain Stage: Create nested RankOrchestrator in TP mode
                // =====================================================================
                LOG_DEBUG("RankOrchestrator: PP stage " << stage_idx
                                                        << " is a TP domain with " << stage_config.stage_devices.size() << " devices");

                // TP siblings retain one stage authority; the PP parent owns
                // composition and cannot substitute a model-wide expert map.
                const Config &nested_config = child_config;

                // Create the nested RankOrchestrator
                // Note: model_ctx_ is the shared ModelContext — the nested MDO's
                // createPPStageRunner uses the shared WeightManager from ModelContext
                auto nested_mdo = std::make_unique<RankOrchestrator>(concrete_model_ctx, nested_config);

                if (!nested_mdo)
                {
                    throw std::runtime_error("Failed to create nested RankOrchestrator for PP stage " +
                                             std::to_string(stage_idx));
                }

                pp_stage_runners_.push_back(std::move(nested_mdo));

                LOG_DEBUG("RankOrchestrator: Created TP domain PP stage " << stage_idx
                                                                          << " with " << stage_config.stage_devices.size() << " devices"
                                                                          << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
            else
            {
                // =====================================================================
                // Single Device Stage: Use existing factory path
                // =====================================================================
                auto runner = createPPStageRunner(concrete_model_ctx, primary_device, factory_pp_config, runner_config);

                if (!runner)
                {
                    throw std::runtime_error("Failed to create PP stage runner for stage " +
                                             std::to_string(stage_idx) + " on device " +
                                             primary_device.to_string());
                }

                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                    dgo->setHostResidentReleaseEnabled(false);

                pp_stage_runners_.push_back(std::move(runner));

                LOG_DEBUG("RankOrchestrator: Created PP stage " << stage_idx
                                                                << " runner on device " << primary_device.to_string()
                                                                << " (layers " << stage_config.first_layer << "-" << stage_config.last_layer << ")");
            }
        }

        LOG_DEBUG("RankOrchestrator: Successfully initialized " << pp_stage_runners_.size() << " PP stage runners");

        // Source reclamation belongs to the complete serving family transition
        // below. Prepared children alone do not prove every graph has bound its
        // operands; nested TP deliberately leaves that shared gate to this PP.

        // Build PPActivationContract describing inter-stage data transfers
        if (config_.pp_stages.size() > 1)
        {
            pp_activation_contract_ = std::make_unique<PPActivationContract>();

            const auto *producer = pp_stage_runners_.front()->getHiddenState();
            if (!producer)
                throw std::logic_error("Pipeline transfer contract requires its admitted producer residual");
            const size_t embedding_dim = pipelineResidualWidth(producer->shape(), producer->native_type());
            const size_t max_seq_len = producer->shape().front();

            for (size_t i = 0; i + 1 < config_.pp_stages.size(); ++i)
            {
                const auto &src = config_.pp_stages[i];
                const auto &dst = config_.pp_stages[i + 1];

                PPStageTransferContract xfer;
                xfer.source_stage = static_cast<int>(i);
                xfer.target_stage = static_cast<int>(i + 1);
                xfer.source_device = src.stage_devices[0].toLocalDeviceId();
                xfer.target_device = dst.stage_devices[0].toLocalDeviceId();
                xfer.embedding_dim = embedding_dim;
                xfer.dtype = ActivationDType::FP32;
                xfer.max_seq_len = max_seq_len;

                pp_activation_contract_->addTransfer(std::move(xfer));
            }

            auto validation_err = pp_activation_contract_->validate();
            if (!validation_err.empty())
            {
                LOG_ERROR("RankOrchestrator: PPActivationContract validation failed: " << validation_err);
                pp_activation_contract_.reset();
            }
            else
            {
                LOG_DEBUG("RankOrchestrator: PPActivationContract built with "
                          << pp_activation_contract_->numTransfers() << " transfers"
                          << " (embedding=" << embedding_dim << ", max_seq=" << max_seq_len << ")");
            }
        }

        // Note: PP context initialization is done by the caller (constructor)
        // after this method returns, to avoid double-initialization
        initializePPGraphExecutionPlan();
    }

    const PipelineGraphExecutionPlan& RankOrchestrator::pipelineGraphExecutionPlan() const
    {
        if(!pp_graph_execution_plan_)
            throw std::logic_error("Rank has no declared pipeline graph execution plan");
        return *pp_graph_execution_plan_;
    }

    void RankOrchestrator::initializePPGraphExecutionPlan()
    {
        if (pp_stage_runners_.empty())
        {
            throw std::runtime_error(
                "Cannot initialize a pipeline graph execution plan without stages");
        }

        std::vector<PipelineGraphExecutionSegment> segments;
        segments.reserve(pp_stage_runners_.size());
        for (std::size_t stage = 0; stage < pp_stage_runners_.size(); ++stage)
        {
            const auto &runner = pp_stage_runners_[stage];
            if (!runner)
            {
                throw std::runtime_error(
                    "Pipeline graph execution plan found a missing runner at stage " +
                    std::to_string(stage));
            }

            const ServingGraphPreparationKind kind =
                runner->servingGraphPreparationKind();
            PipelineGraphSegmentExecution execution;
            switch (kind)
            {
            case ServingGraphPreparationKind::EagerHostGraph:
                execution = PipelineGraphSegmentExecution::HostDeclarative;
                break;
            case ServingGraphPreparationKind::NativeDeviceExecutableFamily:
                execution =
                    PipelineGraphSegmentExecution::NativeDeviceExecutable;
                break;
            case ServingGraphPreparationKind::Unresolved:
                throw std::runtime_error(
                    "Pipeline graph execution plan found an unresolved graph owner at stage " +
                    std::to_string(stage));
            }

            PipelineGraphExecutionSegment segment{.stage_index = stage, .execution = execution};
            if (const auto *rank = dynamic_cast<const IRankOrchestrator *>(runner.get()))
            {
                for (int index = 0; index < rank->device_count(); ++index)
                {
                    const auto *child = rank->deviceRunner(index);
                    if (!child)
                        throw std::runtime_error("Pipeline domain is missing a TP graph owner");
                    segment.participants.push_back(child->primaryDeviceId());
                }
            }
            else segment.participants.push_back(runner->primaryDeviceId());
            if (!segment.valid() || !pipelineStageMatches(*runner, segment))
                throw std::runtime_error("Pipeline domain runners do not match their ordered collective membership at stage " +
                    std::to_string(stage));
            segments.push_back(std::move(segment));
        }

        pp_graph_execution_plan_ =
            std::make_unique<PipelineGraphExecutionPlan>(
                std::move(segments));
    }

    bool RankOrchestrator::validatePPGraphTransactionReady(
        const char *operation) const noexcept
    {
        const char *const operation_name =
            operation && *operation ? operation : "LocalPP transaction";
        if (!pp_graph_execution_plan_)
        {
            LOG_ERROR(operation_name << " has no frozen pipeline graph plan");
            return false;
        }
        if (pp_graph_execution_plan_->segmentCount() !=
            pp_stage_runners_.size())
        {
            LOG_ERROR(
                operation_name
                << " runner cardinality changed after graph-plan construction"
                << " planned="
                << pp_graph_execution_plan_->segmentCount()
                << " live=" << pp_stage_runners_.size());
            return false;
        }

        for (const auto &segment : pp_graph_execution_plan_->segments())
        {
            if (segment.stage_index >= pp_stage_runners_.size() ||
                !pp_stage_runners_[segment.stage_index])
            {
                LOG_ERROR(
                    operation_name << " lost pipeline stage "
                                   << segment.stage_index);
                return false;
            }
            const auto &runner = pp_stage_runners_[segment.stage_index];
            if (!pipelineStageMatches(*runner, segment))
            {
                LOG_ERROR(
                    operation_name
                    << " detected pipeline graph identity drift at stage "
                    << segment.stage_index
                    << " planned_device="
                    << segment.primaryDevice().toString()
                    << " live_device="
                    << runner->primaryDeviceId().toString());
                return false;
            }
        }

        if (pp_graph_execution_plan_->nativeSegmentCount() != 0u &&
            !pp_graph_execution_plan_->materialized())
        {
            LOG_ERROR(
                operation_name
                << " preceded native pipeline graph-family materialization");
            return false;
        }
        return true;
    }

    void RankOrchestrator::recordCompletedPPGraphTransactions(
        const char *phase,
        std::size_t completed_segments,
        std::size_t transaction_count) const
    {
        if (!pp_graph_execution_plan_ ||
            !pp_graph_execution_plan_->hasHeterogeneousBoundary())
        {
            return;
        }

        std::string certification_error;
        if (transaction_count == 0u ||
            !pp_graph_execution_plan_->certifiesReplay(
                completed_segments, &certification_error))
        {
            LOG_ERROR(
                "Completed LocalPP transaction violated its frozen graph plan: "
                << (transaction_count == 0u
                        ? "transaction count is zero"
                        : certification_error));
            std::terminate();
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "segmented_replay_segments",
            static_cast<double>(completed_segments * transaction_count),
            phase && *phase ? phase : "inference",
            "pipeline_coordinator",
            {{"segments_per_transaction",
              std::to_string(completed_segments)},
             {"transactions", std::to_string(transaction_count)},
             {"heterogeneous_segmented", "true"},
             {"boundary_authority", "rank_pipeline_graph_plan"}});
    }

    void RankOrchestrator::initializePPContext()
    {
        if (pp_stage_runners_.empty())
        {
            // PP stage runners not initialized - this is expected for Phase 1
            LOG_DEBUG("RankOrchestrator::initializePPContext: No PP stage runners yet");
            return;
        }

        // Build LocalPPConfig for simple single-device stages
        // TODO: Phase 5 - use HierarchicalPPConfig for TP domain stages
        LocalPPConfig pp_config;
        pp_config.stage_devices.reserve(config_.pp_stages.size());
        pp_config.layer_boundaries.reserve(config_.pp_stages.size() + 1);

        // Add first boundary (start of first stage)
        pp_config.layer_boundaries.push_back(config_.pp_stages[0].first_layer);

        for (size_t i = 0; i < config_.pp_stages.size(); ++i)
        {
            const auto &stage_config = config_.pp_stages[i];
            // Use first device in stage_devices for single-device stages
            if (stage_config.stage_devices.empty())
            {
                throw std::runtime_error("PP stage " + std::to_string(i) + " has no devices");
            }
            pp_config.stage_devices.push_back(stage_config.stage_devices[0]);
            // Each boundary is the exclusive end = last_layer
            pp_config.layer_boundaries.push_back(stage_config.last_layer);
        }

        // Validate config
        if (!pp_config.isValid())
        {
            throw std::runtime_error("Generated LocalPPConfig is invalid");
        }

        LOG_DEBUG("RankOrchestrator::initializePPContext: Creating LocalPPContext with "
                  << pp_config.numStages() << " stages");

        // Create the LocalPPContext using the factory function
        pp_ctx_ = createLocalPPContext(pp_config);

        if (!pp_ctx_)
        {
            throw std::runtime_error("Failed to create LocalPPContext");
        }

        LOG_DEBUG("RankOrchestrator: Initialized LocalPPContext with " << pp_config.numStages() << " stages");
    }

    void RankOrchestrator::aggregateStats() const
    {
        if (!stats_dirty_ || device_runners_.empty())
        {
            return;
        }

        // Reset or create aggregated stats
        if (!aggregated_stats_)
        {
            aggregated_stats_ = std::make_unique<GraphExecutorStats>();
        }
        aggregated_stats_->reset();

        // Aggregate from all device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const auto *stats = runner->executorStats();
                if (stats)
                {
                    aggregated_stats_->total_stages_executed += stats->total_stages_executed;
                    aggregated_stats_->total_flops += stats->total_flops;
                    aggregated_stats_->total_time_ms += stats->total_time_ms;
                    aggregated_stats_->total_execute_ms += stats->total_execute_ms;
                    aggregated_stats_->total_collective_ms += stats->total_collective_ms;
                    aggregated_stats_->total_collective_calls += stats->total_collective_calls;

                    // Merge stage times
                    for (const auto &[stage_name, time_ms] : stats->stage_times_ms)
                    {
                        aggregated_stats_->stage_times_ms[stage_name] += time_ms;
                    }

                    // Merge stage type execution times and counts
                    for (const auto &[type_name, time_ms] : stats->stage_type_execute_ms)
                    {
                        aggregated_stats_->stage_type_execute_ms[type_name] += time_ms;
                    }
                    for (const auto &[type_name, count] : stats->stage_type_counts)
                    {
                        aggregated_stats_->stage_type_counts[type_name] += count;
                    }

                    // Merge overhead breakdown
                    aggregated_stats_->overhead += stats->overhead;

                    // Merge phase-split stats (prefill / decode)
                    auto mergePhase = [](PhaseStats &dst, const PhaseStats &src)
                    {
                        dst.total_execute_ms += src.total_execute_ms;
                        dst.total_stages_executed += src.total_stages_executed;
                        dst.total_collective_ms += src.total_collective_ms;
                        dst.total_collective_calls += src.total_collective_calls;
                        for (const auto &[type_name, time_ms] : src.stage_type_execute_ms)
                            dst.stage_type_execute_ms[type_name] += time_ms;
                        for (const auto &[type_name, cnt] : src.stage_type_counts)
                            dst.stage_type_counts[type_name] += cnt;
                    };
                    mergePhase(aggregated_stats_->prefill, stats->prefill);
                    mergePhase(aggregated_stats_->decode, stats->decode);
                }
            }
        }

        // Average the times (since devices run in parallel)
        if (!device_runners_.empty())
        {
            size_t count = device_runners_.size();
            double dcount = static_cast<double>(count);
            aggregated_stats_->total_time_ms /= dcount;
            aggregated_stats_->total_execute_ms /= dcount;
            aggregated_stats_->total_collective_ms /= dcount;
            aggregated_stats_->total_collective_calls /= count;
            for (auto &[stage_name, time_ms] : aggregated_stats_->stage_times_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, time_ms] : aggregated_stats_->stage_type_execute_ms)
            {
                time_ms /= dcount;
            }
            for (auto &[type_name, cnt] : aggregated_stats_->stage_type_counts)
            {
                cnt /= count;
            }

            // Average overhead (each device incurs its own overhead in parallel)
            aggregated_stats_->overhead.input_cohere_ms /= dcount;
            aggregated_stats_->overhead.weight_cohere_ms /= dcount;
            aggregated_stats_->overhead.output_alloc_ms /= dcount;
            aggregated_stats_->overhead.mark_dirty_ms /= dcount;
            aggregated_stats_->overhead.dump_input_ms /= dcount;
            aggregated_stats_->overhead.dump_output_ms /= dcount;
            aggregated_stats_->overhead.verify_ms /= dcount;
            aggregated_stats_->overhead.callback_ms /= dcount;
            aggregated_stats_->overhead.get_dump_info_ms /= dcount;

            // Average phase stats (devices run in parallel)
            auto avgPhase = [dcount, count](PhaseStats &ps)
            {
                ps.total_execute_ms /= dcount;
                ps.total_stages_executed /= count;
                ps.total_collective_ms /= dcount;
                ps.total_collective_calls /= count;
                for (auto &[_, ms] : ps.stage_type_execute_ms)
                    ms /= dcount;
                for (auto &[_, cnt] : ps.stage_type_counts)
                    cnt /= count;
            };
            avgPhase(aggregated_stats_->prefill);
            avgPhase(aggregated_stats_->decode);
        }

        stats_dirty_ = false;
    }

    // =========================================================================
    // IInferenceRunner Interface Implementation
    // =========================================================================

    bool RankOrchestrator::forward(const int *tokens, int seq_len)
    {
        bool success = false;

        // Dispatch to appropriate implementation based on parallelism mode
        switch (mode_)
        {
        case ParallelismMode::TP:
            success = forwardTP(
                tokens, seq_len, MainForwardDispatch::Automatic);
            break;
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            // PP and TP_PP both use sequential stage execution
            // The difference is that TP_PP stages may be nested MDOs (TP domains)
            // but forwardPP() works through IInferenceRunner interface regardless
            success = forwardPP(
                tokens, seq_len, MainForwardDispatch::Automatic);
            break;
        default:
            LOG_ERROR("RankOrchestrator::forward: Unknown parallelism mode");
            return false;
        }

        return success;
    }

    bool RankOrchestrator::forwardPrefill(const int *tokens, int seq_len)
    {
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(
                tokens, seq_len, MainForwardDispatch::Prefill);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPP(
                tokens, seq_len, MainForwardDispatch::Prefill);
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefill: Unknown parallelism mode");
            return false;
        }
    }

    bool RankOrchestrator::forwardRestoredPrefixMTPDecodeBridge(
        const RestoredPrefixMTPDecodeBridgeRequest &request)
    {
        if (!request.valid() || current_position_ != request.restored_prefix_tokens)
        {
            LOG_ERROR(
                "RankOrchestrator restored-prefix MTP bridge does not match "
                "the rank-local published history"
                << " restored=" << request.restored_prefix_tokens
                << " position=" << current_position_);
            return false;
        }
        const int token = request.token_id;
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(
                &token,
                /*seq_len=*/1,
                MainForwardDispatch::RestoredPrefixMTPDecodeBridge,
                request.restored_prefix_tokens);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPP(
                &token,
                /*seq_len=*/1,
                MainForwardDispatch::RestoredPrefixMTPDecodeBridge,
                request.restored_prefix_tokens);
        default:
            LOG_ERROR(
                "RankOrchestrator::forwardRestoredPrefixMTPDecodeBridge: "
                "Unknown parallelism mode");
            return false;
        }
    }

    ServingGraphPreparationKind
    RankOrchestrator::servingGraphPreparationKind() const noexcept
    {
        const auto composite_kind = [](
                                        const std::vector<
                                            std::unique_ptr<IInferenceRunner>>
                                            &runners)
        {
            if (runners.empty())
                return ServingGraphPreparationKind::Unresolved;
            bool native_family_required = false;
            for (const auto &runner : runners)
            {
                if (!runner)
                    return ServingGraphPreparationKind::Unresolved;
                const auto child_kind =
                    runner->servingGraphPreparationKind();
                if (child_kind ==
                    ServingGraphPreparationKind::Unresolved)
                {
                    return child_kind;
                }
                native_family_required =
                    native_family_required ||
                    child_kind == ServingGraphPreparationKind::
                                      NativeDeviceExecutableFamily;
            }
            return native_family_required
                       ? ServingGraphPreparationKind::
                             NativeDeviceExecutableFamily
                       : ServingGraphPreparationKind::EagerHostGraph;
        };

        if (mode_ == ParallelismMode::TP &&
            pp_stage_runners_.empty())
        {
            if (device_runners_.empty() || !device_runners_.front())
                return ServingGraphPreparationKind::Unresolved;
            const auto first =
                device_runners_.front()->servingGraphPreparationKind();
            if (first == ServingGraphPreparationKind::Unresolved)
                return first;
            for (const auto &runner : device_runners_)
            {
                if (!runner ||
                    runner->servingGraphPreparationKind() != first)
                {
                    // A mixed LocalTP tier needs a distinct symmetric graph
                    // protocol; it cannot inherit PP's sequential composite.
                    return ServingGraphPreparationKind::Unresolved;
                }
            }
            return first;
        }
        if ((mode_ == ParallelismMode::PP ||
             mode_ == ParallelismMode::TP_PP) &&
            device_runners_.empty())
        {
            /*
             * PP stages are sequential, so a mixed CPU/GPU pipeline reports
             * the strongest child transition. Materialization below binds and
             * seals every stage in pipeline order, including CPU workspace
             * families. Nested LocalTP stages remain responsible for their
             * own symmetric participant capture.
             */
            return composite_kind(pp_stage_runners_);
        }
        return ServingGraphPreparationKind::Unresolved;
    }

    bool RankOrchestrator::materializeServingGraphFamilyWithoutLaunch(
        const ServingGraphFamilyMaterializationPlan &plan)
    {
        if (!plan.valid())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup requires a valid frozen family plan");
            return false;
        }

        if (mode_ == ParallelismMode::PP ||
            mode_ == ParallelismMode::TP_PP)
        {
            if (!device_runners_.empty() || pp_stage_runners_.empty())
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup found an invalid composite runner shape");
                return false;
            }
            if (!pp_graph_execution_plan_ ||
                pp_graph_execution_plan_->segmentCount() !=
                    pp_stage_runners_.size())
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup has no exact frozen pipeline plan");
                return false;
            }

            std::size_t materialized_native_segments = 0u;
            if (PipelineDeviceGeneration::isGPUStageSet(pp_stage_runners_))
            {
                if (!pp_device_generation_)
                    pp_device_generation_ = std::make_unique<PipelineDeviceGeneration>(pp_stage_runners_, config_.original_pipeline_mpi_ctx);
                if (!pp_device_generation_->prepareServing(plan)) return false;
                materialized_native_segments = pp_stage_runners_.size();
            }
            else
            {
                for (const auto &segment : pp_graph_execution_plan_->segments())
                {
                    const std::size_t stage = segment.stage_index;
                    auto &runner = pp_stage_runners_[stage];
                    if (!runner)
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP serving graph setup found a missing stage at index " << stage);
                        return false;
                    }
                    if (!pipelineStageMatches(*runner, segment))
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP serving graph setup detected identity drift at stage "
                            << stage << " planned_device=" << segment.primaryDevice().toString()
                            << " live_device=" << runner->primaryDeviceId().toString());
                        return false;
                    }
                    ServingGraphFamilyMaterializationPlan stage_plan = plan;
                    stage_plan.pipeline_hidden_input = nullptr;
                    if (stage > 0u)
                    {
                        auto *previous = pp_stage_runners_[stage - 1u].get();
                        stage_plan.pipeline_hidden_input = previous ? previous->getHiddenState() : nullptr;
                        if (!stage_plan.pipeline_hidden_input)
                        {
                            LOG_ERROR(
                                "RankOrchestrator PP serving graph setup could not bind the stable activation owner from stage "
                                << (stage - 1u) << " to stage " << stage);
                            return false;
                        }
                    }
                    if (!runner->materializeServingGraphFamilyWithoutLaunch(stage_plan))
                    {
                        LOG_ERROR("RankOrchestrator PP serving graph setup failed at stage " << stage);
                        return false;
                    }
                    // CPU stages declare their workspace family, but do not
                    // contribute a native executable to this inventory.
                    if (segment.execution == PipelineGraphSegmentExecution::NativeDeviceExecutable)
                        ++materialized_native_segments;
                }
            }

            std::string materialization_error;
            if (!pp_graph_execution_plan_->markMaterialized(
                    materialized_native_segments,
                    &materialization_error))
            {
                LOG_ERROR(
                    "RankOrchestrator PP serving graph setup did not cover its frozen native inventory: "
                    << materialization_error);
                return false;
            }

            // Publish only after the complete, exact stage inventory succeeds.
            // WeightManager's typed CPU policy retains any CPU operands while
            // reclaiming GPU-only sources, even in a heterogeneous pipeline.
            if (model_ctx_)
            {
                if (const auto weights = model_ctx_->weightManager())
                {
                    weights->markGraphMaterializationComplete();
                    const auto released = weights->releaseAllHostWeightData();
                    LOG_DEBUG("RankOrchestrator: Released " << released
                        << " host weight copies after PP graph materialization");
                }
            }

            if (pp_graph_execution_plan_->hasHeterogeneousBoundary())
            {
                const std::string total_segments = std::to_string(
                    pp_graph_execution_plan_->segmentCount());
                const std::string native_segments = std::to_string(
                    pp_graph_execution_plan_->nativeSegmentCount());
                const std::string host_segments = std::to_string(
                    pp_graph_execution_plan_->hostSegmentCount());
                const PerfStatsCollector::Tags tags =
                    {{"scope", "pipeline_coordinator"},
                     {"total_segments", total_segments},
                     {"native_segments", native_segments},
                     {"host_segments", host_segments},
                     {"heterogeneous_segmented", "true"},
                     {"boundary_authority", "rank_pipeline_graph_plan"}};
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_plan_segments",
                    static_cast<double>(
                        pp_graph_execution_plan_->segmentCount()),
                    "setup",
                    "pipeline_coordinator",
                    tags);
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "segmented_graph_capture_segments",
                    static_cast<double>(materialized_native_segments),
                    "setup",
                    "pipeline_coordinator",
                    tags);
            }

            PerfStatsCollector::addCounter(
                "forward_graph",
                "serving_graph_family_pipeline_completions",
                1.0,
                "setup",
                "rank",
                {{"stages", std::to_string(pp_stage_runners_.size())},
                 {"request_state_mutations", "0"},
                 {"executable_launches", "0"}});
            return true;
        }

        if (mode_ != ParallelismMode::TP ||
            !pp_stage_runners_.empty() || device_runners_.empty())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup requires one flat, "
                "non-empty LocalTP graph and a valid frozen family plan");
            return false;
        }
        if (!tp_ctx_ || tp_ctx_->degree() != static_cast<int>(device_runners_.size()) ||
            tp_ctx_->devices().size() != device_runners_.size())
        {
            LOG_ERROR("RankOrchestrator serving graph setup lost its exact LocalTP participant inventory");
            return false;
        }
        const auto preparation_kind = servingGraphPreparationKind();
        if (preparation_kind != ServingGraphPreparationKind::EagerHostGraph &&
            preparation_kind != ServingGraphPreparationKind::NativeDeviceExecutableFamily)
        {
            LOG_ERROR("RankOrchestrator serving graph setup requires one declared homogeneous preparation lifecycle");
            return false;
        }
        // Validate the entire immutable participant set before visiting any
        // child. CPU endpoints share DeviceId::cpu(), so their original vector
        // index remains the worker/runner binding; never infer a NUMA ordinal
        // from storage identity or substitute a GPU capture lifecycle.
        for (std::size_t index = 0; index < device_runners_.size(); ++index)
        {
            const auto &runner = device_runners_[index];
            if (!runner)
            {
                LOG_ERROR("RankOrchestrator serving graph setup found a missing LocalTP participant at index " << index);
                return false;
            }
            const auto device = runner->primaryDeviceId();
            const bool supported_locus =
                (device.is_cpu() && preparation_kind == ServingGraphPreparationKind::EagerHostGraph) ||
                (device.is_gpu() && preparation_kind == ServingGraphPreparationKind::NativeDeviceExecutableFamily);
            if (!device.is_valid() || !supported_locus ||
                runner->servingGraphPreparationKind() != preparation_kind ||
                device != tp_ctx_->devices()[index].toLocalDeviceId())
            {
                LOG_ERROR("RankOrchestrator serving graph setup found incompatible declared lifecycle/device at LocalTP index "
                    << index << " device=" << device.toString());
                return false;
            }
        }

        if (device_runners_.size() == 1)
        {
            return device_runners_.front()
                ->materializeServingGraphFamilyWithoutLaunch(plan);
        }

        /*
         * Native graph capture can enter LocalTP collective nodes. All native
         * participants therefore build/capture on their persistent workers at
         * the same time. CPU participants use those same original worker slots
         * to declare workspace and seal their host engine, without executing
         * arithmetic, capturing a GPU graph or changing inherited CPU placement.
         * A serial native setup loop would deadlock at the first collective.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN(
                            "[TPWorkerPool] Serving graph setup failure "
                            "detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != device_runners_.size())
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup worker topology no "
                "longer matches the immutable LocalTP participant count");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             &plan,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](std::size_t index) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device =
                    device_runners_[index]->primaryDeviceId();
                if (device.is_rocm()) ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                if (device.is_cuda()) CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                const auto started = std::chrono::steady_clock::now();
                LOG_INFO(
                    "[ServingGraphMaterialization] LocalTP worker begin"
                    << " participant=" << index
                    << " device=" << device.toString());
                const bool success = device_runners_[index]
                                         ->materializeServingGraphFamilyWithoutLaunch(
                                             plan);
                const auto elapsed = std::chrono::duration_cast<
                    std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - started);
                LOG_INFO(
                    "[ServingGraphMaterialization] LocalTP worker complete"
                    << " participant=" << index
                    << " device=" << device.toString()
                    << " success=" << (success ? "true" : "false")
                    << " elapsed_ms=" << elapsed.count());
                return success;
            });

        bool all_success = true;
        std::exception_ptr first_exception;
        std::size_t first_exception_device = 0;
        const auto results = tp_worker_pool_->collectAll(
            /*timeout_ms=*/0);
        for (const auto &result : results)
        {
            if (!result.completed || !result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
            }
        }
        if (first_exception)
        {
            LOG_ERROR(
                "RankOrchestrator serving graph setup rethrowing failure from "
                "LocalTP participant "
                << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        if (rank_local_expert_endpoints_) {
            std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> original;
            std::vector<RetainedForwardSourceDefinition> original_definitions;
            for (const auto& child : device_runners_) {
                const auto* device = dynamic_cast<const DeviceGraphOrchestrator*>(child.get());
                if (!device) continue;
                auto definitions = device->retainedExpertSourceFamilyDefinitions();
                const auto source = std::find_if(definitions.begin(),definitions.end(),[](const auto& definition) {
                    return static_cast<bool>(definition.source);
                });
                if (source == definitions.end()) continue;
                if (original)
                    throw std::invalid_argument("Rank expert storage family has multiple original continuation source owners");
                auto model = std::dynamic_pointer_cast<ModelContext>(model_ctx_);
                if (!model)
                    throw std::invalid_argument("Rank expert storage family lost its actual original ModelContext");
                original = model->adaptiveImmutableLookupParticipants(source->source->originalSource().device());
                if (!original || !original->belongsToOriginalCohort(replicated_ple_cohort_))
                    throw std::invalid_argument("Rank expert storage family changed its original declared source cohort");
                original_definitions = std::move(definitions);
            }
            if (original)
                rank_local_expert_endpoints_->bindOriginalPooledSourceFamily(std::move(original),std::move(original_definitions));
            if (retained_source_admission_)
                rank_local_expert_endpoints_->bindOriginalPooledSourceCancellation(retained_source_admission_->cancel);
        }

        if (rank_local_expert_endpoints_ &&
            !rank_local_expert_endpoints_->materializeServingGraphFamilyWithoutLaunch(plan))
            return false;

        PerfStatsCollector::addCounter(
            "forward_graph",
            "serving_graph_family_rank_completions",
            1.0,
            "setup",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"prefill_buckets",
              std::to_string(plan.prefill_bucket_rows.size())},
             {"fanout", "concurrent_persistent_workers"},
             {"preparation_kind", preparation_kind == ServingGraphPreparationKind::EagerHostGraph
                 ? "eager_host_graph" : "native_device_executable_family"}});
        return true;
    }

    /**
     * @brief Forward one overlay request generation through the rank-local graph tree.
     *
     * A RankOrchestrator may own LocalTP children or nested PP-stage runners.
     * Each child that can enter a graph-native sparse boundary must see the
     * same root-published generation before any worker thread begins its graph
     * launch. Propagating during request setup keeps this out of the token hot
     * path and prevents a child graph cache from inventing its own epoch.
     */
    bool RankOrchestrator::setMoEOverlayCollectiveRequestGeneration(
        uint64_t generation_id)
    {
        if (generation_id == 0)
        {
            LOG_ERROR(
                "RankOrchestrator::setMoEOverlayCollectiveRequestGeneration "
                "received an invalid zero generation");
            return false;
        }

        const auto &children =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (children.empty())
        {
            LOG_ERROR(
                "RankOrchestrator::setMoEOverlayCollectiveRequestGeneration "
                "has no child graph runners");
            return false;
        }

        for (const auto &child : children)
        {
            if (!child ||
                !child->setMoEOverlayCollectiveRequestGeneration(
                    generation_id))
            {
                LOG_ERROR(
                    "RankOrchestrator could not publish graph-native MoE "
                    "collective generation to every child graph");
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::setMoEOverlayInferenceTransactionCoordinator(
        std::shared_ptr<MoEOverlayInferenceTransactionCoordinator>
            coordinator,
        int continuation_participant_index)
    {
        /*
         * A RankOrchestrator is itself one participant only when nested under
         * another composite runner.  The production heterogeneous overlay cell
         * is the outer LocalTP rank, so it owns all child participant indices
         * and requires callers to address that rank as participant zero.
         */
        if (!coordinator || continuation_participant_index != 0 ||
            mode_ != ParallelismMode::TP ||
            device_runners_.empty() || !pp_stage_runners_.empty() ||
            coordinator->continuationParticipantCount() !=
                static_cast<int>(device_runners_.size()))
        {
            LOG_ERROR(
                "RankOrchestrator requires a flat LocalTP graph whose child "
                "count exactly matches the ExpertOverlay coordinator");
            return false;
        }

        if (moe_overlay_inference_transaction_coordinator_ &&
            moe_overlay_inference_transaction_coordinator_ != coordinator)
        {
            LOG_ERROR(
                "RankOrchestrator cannot replace its ExpertOverlay transaction authority");
            return false;
        }
        if (moe_overlay_interference_probe_)
        {
            std::string probe_error;
            if (!coordinator->bindPrefillInterferenceProbe(
                    moe_overlay_interference_probe_, &probe_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind chunk-level ExpertOverlay "
                    "prefill calibration: " << probe_error);
                return false;
            }
        }

        for (std::size_t index = 0; index < device_runners_.size(); ++index)
        {
            auto &child = device_runners_[index];
            if (!child ||
                !child->setMoEOverlayInferenceTransactionCoordinator(
                    coordinator, static_cast<int>(index)))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind every LocalTP child to "
                    "the rank-wide ExpertOverlay transaction authority");
                return false;
            }
        }
        moe_overlay_inference_transaction_coordinator_ =
            std::move(coordinator);
        moe_overlay_coordinator_owns_prefill_probe_ =
            static_cast<bool>(moe_overlay_interference_probe_);
        return true;
    }

    bool RankOrchestrator::setMoEOverlayInferenceInterferenceProbe(
        std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe)
    {
        if (!probe || mode_ != ParallelismMode::TP ||
            device_runners_.empty() || !pp_stage_runners_.empty() ||
            (moe_overlay_interference_probe_ &&
             moe_overlay_interference_probe_ != probe))
        {
            LOG_ERROR(
                "RankOrchestrator requires one flat LocalTP owner and one stable "
                "ExpertOverlay interference probe");
            return false;
        }
        moe_overlay_interference_probe_ = std::move(probe);
        if (moe_overlay_inference_transaction_coordinator_)
        {
            std::string probe_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->bindPrefillInterferenceProbe(
                         moe_overlay_interference_probe_, &probe_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not bind chunk-level ExpertOverlay "
                    "prefill calibration: " << probe_error);
                moe_overlay_interference_probe_.reset();
                return false;
            }
            moe_overlay_coordinator_owns_prefill_probe_ = true;
        }
        return true;
    }

    bool RankOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds(
        const std::vector<std::vector<int>> &token_batches)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardGroupedMTPVerifierWithHostTokenIds(
                token_batches);
        }

        for (const auto &runner : device_runners_)
        {
            if (runner && !runner->primaryDeviceId().is_cpu())
            {
                LOG_ERROR(
                    "RankOrchestrator::forwardGroupedMTPVerifierWithHostTokenIds: "
                    "host-token verification is invalid for GPU participants");
                return false;
            }
        }

        return forwardHostTokenBatchAcrossDevices(
            token_batches,
            &IInferenceRunner::forwardGroupedMTPVerifierWithHostTokenIds,
            "forwardGroupedMTPVerifierWithHostTokenIds");
    }

    bool RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds(
        const int *token_shadow,
        const void *token_ids_device,
        int seq_len)
    {
        if (pp_device_generation_)
            return pp_device_generation_->forwardMTPVerifier(seq_len,
                [&](IInferenceRunner &tail) {
                    return tail.forwardGroupedMTPVerifierWithDeviceTokenIds(token_shadow, token_ids_device, seq_len);
                });
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardGroupedMTPVerifierWithDeviceTokenIds(
                token_shadow,
                token_ids_device,
                seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardGroupedMTPVerifierWithDeviceTokenIds(
                token_shadow,
                token_ids_device,
                seq_len);
        }
        if (device_runners_.size() < 2 ||
            !token_shadow ||
            seq_len <= 0 ||
            token_ids_device != rank_mtp_verifier_child_token_inputs_.data() ||
            rank_mtp_verifier_child_token_inputs_.size() !=
                device_runners_.size() ||
            rank_mtp_verifier_child_token_count_ != seq_len)
        {
            LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: "
                      "invalid LocalTP device-token input bundle");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !rank_mtp_verifier_child_token_inputs_[i])
            {
                LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: participant "
                          << i << " has no staged device-token input row");
                return false;
            }
        }

        // A restored prefix can make this the first main-model operation of
        // the request. Admit the complete source cohort before either worker
        // reaches its participant-local storage prelude.
        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] device-token verifier forward failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        tp_worker_pool_->dispatch(
            [this,
             token_shadow,
             seq_len,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_forward_enter"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " seq_len=" << seq_len);
                }

                const bool ok =
                    device_runners_[i]
                        ->forwardGroupedMTPVerifierWithDeviceTokenIds(
                        token_shadow,
                        rank_mtp_verifier_child_token_inputs_[i],
                        seq_len);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_forward_leave"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " success=" << (ok ? 1 : 0));
                }
                return ok;
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
                all_success = false;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardGroupedMTPVerifierWithDeviceTokenIds",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardGroupedMTPVerifierWithDeviceTokenIds: "
                      "Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        current_position_ += seq_len;
        current_padded_seq_len_ = seq_len;
        if (current_sequence_lengths_.empty())
        {
            current_sequence_lengths_.assign(
                static_cast<size_t>(std::max(1, config_.batch_size)),
                current_position_ - seq_len);
        }
        for (int &length : current_sequence_lengths_)
            length += seq_len;
        stats_dirty_ = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_device_token_forwards",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"seq_len", std::to_string(seq_len)}});
        source_attempt.keep=true;
        return true;
    }

    bool RankOrchestrator::
        advanceMTPMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            MTPConditionForwardPurpose purpose,
            int request_index)
    {
        if (pp_device_generation_)
            return pp_device_generation_->forwardMTPCondition(purpose,
                [&](IInferenceRunner &tail) {
                    return tail.advanceMTPMainConditionFromDeviceResidentLogicalState(
                        token_shadow, logical_state, purpose, request_index);
                });
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->advanceMTPMainConditionFromDeviceResidentLogicalState(
                    token_shadow,
                    logical_state,
                    purpose,
                    request_index);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->advanceMTPMainConditionFromDeviceResidentLogicalState(
                    token_shadow,
                    logical_state,
                    purpose,
                    request_index);
        }

        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (!logical_state.valid() ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            !logical_state.coversRequest(request_index) ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident main-condition advance received a "
                "stale, foreign, or incomplete LocalTP mailbox");
            return false;
        }

        const bool advanced = dispatchLocalTPMTPMainCondition(
            "advanceMTPMainConditionFromDeviceResidentLogicalState",
            LocalTPConditionEffect::ForwardMainRow,
            [this, token_shadow, purpose, request_index](
                IInferenceRunner &child,
                size_t participant)
            {
                return child
                    .advanceMTPMainConditionFromDeviceResidentLogicalState(
                        token_shadow,
                        rank_resident_child_logical_state_handles_[participant],
                        purpose,
                        request_index);
            });
        if (!advanced)
            return false;

        /*
         * Every child has consumed and retired its one-shot condition input.
         * The rank-level bundle is only a typed view over those child handles,
         * so retaining it would resurrect state that no participant owns.
         * Canonical child KV becomes the sole post-forward position authority.
         */
        invalidateRankResidentLogicalStateAggregate(
            "resident_main_condition_advance",
            "condition_input_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_)
            ++length;
        current_padded_seq_len_ = 1;
        // This public condition transaction completed one main decode row on
        // every participant. Token/mailbox publication alone does not own this
        // phase edge, and a failed fan-out must retain the preceding phase.
        last_logits_forward_phase_ = LogitsForwardPhase::Decode;
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::advanceMTPMainConditionFromDeviceTargetSample(
        int32_t token_shadow,
        int target_sample_slot,
        MTPConditionForwardPurpose purpose)
    {
        if (pp_device_generation_)
            return pp_device_generation_->forwardMTPCondition(purpose,
                [&](IInferenceRunner &tail) {
                    return tail.advanceMTPMainConditionFromDeviceTargetSample(token_shadow, target_sample_slot, purpose);
                });
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->advanceMTPMainConditionFromDeviceTargetSample(
                token_shadow,
                target_sample_slot,
                purpose);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->advanceMTPMainConditionFromDeviceTargetSample(
                    token_shadow,
                    target_sample_slot,
                    purpose);
        }
        if (target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            LOG_ERROR(
                "[RankOrchestrator] Device-target main-condition advance "
                "received an invalid target slot");
            return false;
        }

        const bool advanced = dispatchLocalTPMTPMainCondition(
            "advanceMTPMainConditionFromDeviceTargetSample",
            LocalTPConditionEffect::ForwardMainRow,
            [token_shadow, target_sample_slot, purpose](
                IInferenceRunner &child,
                size_t)
            {
                return child.advanceMTPMainConditionFromDeviceTargetSample(
                    token_shadow,
                    target_sample_slot,
                    purpose);
            });
        if (!advanced)
            return false;

        /*
         * Target-sample publication is the condition graph's input, not its
         * output. Child completion retires those mailboxes; rank orchestration
         * must likewise expose no aggregate until sampling or verifier
         * publication creates the next device-owned condition transaction.
         */
        invalidateRankResidentLogicalStateAggregate(
            "target_sample_main_condition_advance",
            "condition_input_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_)
            ++length;
        current_padded_seq_len_ = 1;
        // This public condition transaction completed one main decode row on
        // every participant. Token/mailbox publication alone does not own this
        // phase edge, and a failed fan-out must retain the preceding phase.
        last_logits_forward_phase_ = LogitsForwardPhase::Decode;
        stats_dirty_ = true;
        return true;
    }

    void RankOrchestrator::retirePipelineForwardOnlyDomain()
    {
        if (device_runners_.empty() || !pp_stage_runners_.empty())
            throw std::logic_error("Pipeline forward-only completion requires its original local TP domain");
        // The composed parent bypasses forwardTP(), including its host-result
        // lifecycle. Child tensors already carry publishForwardGraphOutputReady
        // on each exact parent stream. Retire the old prefill row even when
        // observation is disabled, then consume only those original producer
        // edges when this terminal domain explicitly requests a host result.
        if (logits_gatherer_)
            logits_gatherer_->invalidate();
        if (ownsMainForwardLogits() && !skip_logits_gather_decode_ &&
            (!logits_gatherer_ || !logits_gatherer_->isAllocated()))
            throw std::logic_error("Pipeline forward-only domain has no retained host logits observation owner");
        if (ownsMainForwardLogits() && logits_gatherer_ &&
            logits_gatherer_->needsGather(LogitsForwardPhase::Decode) &&
            !logits_gatherer_->gather(device_runners_, 1, vocab_size()))
            throw std::runtime_error("Pipeline forward-only domain failed to publish current host logits");
        // A rejected observation leaves the prior public geometry unchanged
        // and its host data invalid; storage capacity never certifies freshness.
        invalidateRankResidentLogicalStateAggregate("pipeline_forward_only_domain", "original_children_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_) ++length;
        current_padded_seq_len_ = 1;
        last_logits_forward_phase_ = LogitsForwardPhase::Decode;
        stats_dirty_ = true;
    }

    bool RankOrchestrator::
        advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
            int32_t token_shadow,
            const DeviceResidentLogicalSequenceStateHandle &logical_state,
            int request_index)
    {
        if (pp_device_generation_)
        {
            // A prior prefill buffer is storage, never proof of this new row.
            // Host observation remains optional and uses the original tail owner.
            if (logits_gatherer_) logits_gatherer_->invalidate();
            if (ownsMainForwardLogits() && !skip_logits_gather_decode_ &&
                (!logits_gatherer_ || !logits_gatherer_->isAllocated()))
            {
                LOG_ERROR("Pipeline ordinary condition requested host logits without its retained observation owner");
                return false;
            }
            if (!pp_device_generation_->advanceOrdinaryCondition(logical_state, request_index))
                return false;
            if (ownsMainForwardLogits() && !skip_logits_gather_decode_)
            {
                logits_gatherer_->copyFromStage(*pp_stage_runners_.back(), static_cast<size_t>(vocab_size()));
                if (!logits_gatherer_->data()) return false;
                PerfStatsCollector::addCounter("sampling", "rank_pipeline_condition_host_logits_publications",
                    1.0, "decode", "rank", {{"rows", "1"},
                        {"participants", std::to_string(pp_stage_runners_.size())},
                        {"authority", "original_pipeline_tail"}});
            }
            ++current_position_;
            for (int &length : current_sequence_lengths_) ++length;
            current_padded_seq_len_ = 1;
            last_logits_forward_phase_ = LogitsForwardPhase::Decode;
            stats_dirty_ = true;
            return true;
        }
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
                    token_shadow, logical_state, request_index);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
                    token_shadow, logical_state, request_index);
        }

        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (!logical_state.valid() || !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            !logical_state.coversRequest(request_index) ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR(
                "[RankOrchestrator] Ordinary resident condition received a stale, foreign, or incomplete LocalTP mailbox");
            return false;
        }

        const bool advanced = dispatchLocalTPMTPMainCondition(
            "advanceOrdinaryMainConditionFromDeviceResidentLogicalState",
            LocalTPConditionEffect::ForwardMainRow,
            [this, token_shadow, request_index](
                IInferenceRunner &child, size_t participant)
            {
                return child
                    .advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
                        token_shadow,
                        rank_resident_child_logical_state_handles_[participant],
                        request_index);
            });
        if (!advanced)
            return false;

        invalidateRankResidentLogicalStateAggregate(
            "resident_ordinary_condition_advance",
            "condition_input_consumed");
        ++current_position_;
        for (int &length : current_sequence_lengths_)
            ++length;
        current_padded_seq_len_ = 1;
        // This public condition transaction completed one main decode row on
        // every participant. Token/mailbox publication alone does not own this
        // phase edge, and a failed fan-out must retain the preceding phase.
        last_logits_forward_phase_ = LogitsForwardPhase::Decode;
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::forwardBatchWithDeviceTokenIds(
        const std::vector<std::vector<int>> &token_batches,
        const void *token_ids_device,
        int padded_seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardBatchWithDeviceTokenIds(
                token_batches,
                token_ids_device,
                padded_seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardBatchWithDeviceTokenIds(
                token_batches,
                token_ids_device,
                padded_seq_len);
        }
        if (device_runners_.size() < 2 ||
            token_batches.empty() ||
            padded_seq_len <= 0 ||
            token_ids_device != rank_mtp_verifier_child_token_inputs_.data() ||
            rank_mtp_verifier_child_token_inputs_.size() !=
                device_runners_.size() ||
            rank_mtp_verifier_child_token_count_ != padded_seq_len)
        {
            LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: invalid LocalTP verifier token bundle");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !rank_mtp_verifier_child_token_inputs_[i])
            {
                LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: participant "
                          << i << " has no staged verifier token matrix");
                return false;
            }
        }

        // Grouped rows share the same original request/source cohort as scalar
        // conditions; a verifier may be their first execution after restore.
        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] device-token batch verifier forward failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        tp_worker_pool_->dispatch(
            [this,
             &token_batches,
             padded_seq_len,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_batch_forward_enter"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " padded_seq_len=" << padded_seq_len
                              << " request_batch=" << token_batches.size());
                }

                const bool ok =
                    device_runners_[i]->forwardBatchWithDeviceTokenIds(
                        token_batches,
                        rank_mtp_verifier_child_token_inputs_[i],
                        padded_seq_len);

                if (debugEnv().tp_collective_contract_trace)
                {
                    LOG_DEBUG("[TP_WORKER_CONTRACT] event=device_token_batch_forward_leave"
                              << " worker=" << i
                              << " device=" << device_id.toString()
                              << " success=" << (ok ? 1 : 0));
                }
                return ok;
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
                all_success = false;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardBatchWithDeviceTokenIds",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardBatchWithDeviceTokenIds: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        current_position_ += padded_seq_len;
        current_padded_seq_len_ = padded_seq_len;
        current_sequence_lengths_.resize(std::max(
            current_sequence_lengths_.size(),
            token_batches.size()),
            0);
        for (size_t request = 0; request < token_batches.size(); ++request)
        {
            current_sequence_lengths_[request] +=
                static_cast<int>(token_batches[request].size());
        }
        stats_dirty_ = true;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_device_token_batch_forwards",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(token_batches.size())},
             {"padded_seq_len", std::to_string(padded_seq_len)}});
        source_attempt.keep=true;
        return true;
    }

    bool RankOrchestrator::supportsPrefillChunkSchedule(int seq_len) const
    {
        const auto *runners =
            mode_ == ParallelismMode::TP
                ? &device_runners_
                : (mode_ == ParallelismMode::PP ||
                           mode_ == ParallelismMode::TP_PP
                       ? &pp_stage_runners_
                       : nullptr);
        if (!runners || runners->empty())
            return false;
        if ((mode_ == ParallelismMode::PP ||
             mode_ == ParallelismMode::TP_PP) &&
            !pp_ctx_ && !(pp_device_generation_ && pp_device_generation_->prefillPrepared()))
        {
            return false;
        }

        for (const auto &runner : *runners)
        {
            if (!runner || !runner->supportsPrefillChunkSchedule(seq_len))
                return false;
        }
        return true;
    }

    bool RankOrchestrator::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        switch (mode_)
        {
        case ParallelismMode::TP:
            return forwardTP(
                tokens,
                seq_len,
                MainForwardDispatch::Automatic,
                /*restored_prefix_tokens=*/0,
                &policy,
                pad_token_id,
                allow_padded_execution);
        case ParallelismMode::PP:
        case ParallelismMode::TP_PP:
            return forwardPPPrefillChunkSchedule(
                tokens,
                seq_len,
                policy,
                pad_token_id,
                allow_padded_execution);
        default:
            LOG_ERROR("RankOrchestrator::forwardPrefillChunkSchedule: Unknown parallelism mode");
            return false;
        }
    }

    bool RankOrchestrator::forwardPPPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        if (!tokens || seq_len <= 0 ||
            pp_stage_runners_.empty() || (!pp_ctx_ && !(pp_device_generation_ && pp_device_generation_->prefillPrepared())))
        {
            LOG_ERROR(
                "RankOrchestrator PP prefill schedule requires tokens, "
                "pipeline stages, and a LocalPP transfer context");
            return false;
        }
        if (!validatePPGraphTransactionReady(
                "RankOrchestrator PP prefill schedule"))
        {
            return false;
        }
        if (policy.real_token_start != current_position_ ||
            policy.real_token_count != seq_len)
        {
            LOG_ERROR(
                "RankOrchestrator PP prefill schedule disagrees with the "
                "pipeline request cursor"
                << " schedule_start=" << policy.real_token_start
                << " position=" << current_position_
                << " schedule_rows=" << policy.real_token_count
                << " request_rows=" << seq_len);
            return false;
        }

        const PrefillChunkSchedule root_schedule =
            planPrefillChunkSchedule(policy);
        if (!root_schedule || root_schedule.chunks.empty())
        {
            LOG_ERROR(
                "RankOrchestrator could not plan its pipeline-wide prefill "
                "schedule: "
                << root_schedule.error);
            return false;
        }

        last_logits_forward_phase_ = LogitsForwardPhase::Prefill;
        if (logits_gatherer_)
            logits_gatherer_->invalidate();

        std::uint64_t transferred_rows = 0u;
        const std::size_t num_stages =
            pp_graph_execution_plan_->segmentCount();
        std::size_t completed_transactions = 0u;
        if (pp_device_generation_ && pp_device_generation_->prefillPrepared())
        {
            if (!pp_device_generation_->prefill(tokens, seq_len, policy, pad_token_id, allow_padded_execution))
                return false;
            if (!skip_logits_gather_prefill_)
            {
                if (!logits_gatherer_)
                {
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(0, 0, logits_backend_resolver_);
                    applyLogitsGatherSkipFlags();
                }
                logits_gatherer_->copyFromStage(*pp_stage_runners_.back());
            }
            recordCompletedPPGraphTransactions("prefill", num_stages, root_schedule.chunks.size());
            current_position_ += seq_len;
            PerfStatsCollector::addCounter("forward_graph", "pipeline_prefill_chunk_transactions",
                static_cast<double>(root_schedule.chunks.size()), "prefill", "rank",
                {{"stages", std::to_string(num_stages)}, {"logical_rows", std::to_string(seq_len)},
                 {"boundary_authority", "captured_native_pipeline_edges"}});
            return true;
        }
        for (const PrefillChunkPlan &chunk : root_schedule.chunks)
        {
            const int relative_offset =
                chunk.token_offset - policy.real_token_start;
            if (relative_offset < 0 || chunk.real_count <= 0 ||
                relative_offset > seq_len ||
                chunk.real_count > seq_len - relative_offset ||
                chunk.bucket_seq_len < chunk.real_count)
            {
                LOG_ERROR(
                    "RankOrchestrator PP prefill plan contains an invalid "
                    "chunk"
                    << " chunk=" << chunk.chunk_index
                    << " offset=" << relative_offset
                    << " real_rows=" << chunk.real_count
                    << " bucket_rows=" << chunk.bucket_seq_len);
                return false;
            }

            /*
             * A preceding transfer may leave TensorBase's current-device
             * metadata naming the destination. The retained stage-0 graph
             * still owns its original allocation, so make that allocation the
             * selected producer again before the next chunk writes it.
             */
            if (num_stages > 1u)
            {
                TensorBase *const stage0_hidden =
                    pp_stage_runners_.front()->getHiddenState();
                if (stage0_hidden)
                {
                    const DeviceId stage0_device =
                        pp_ctx_->deviceForStage(0).toLocalDeviceId();
                    const auto current_device =
                        stage0_hidden->current_device();
                    if (current_device && *current_device != stage0_device)
                        stage0_hidden->allocateOnDevice(stage0_device);
                }
            }

            std::size_t completed_segments = 0u;
            for (const auto &segment :
                 pp_graph_execution_plan_->segments())
            {
                const std::size_t stage = segment.stage_index;
                auto &runner = pp_stage_runners_[stage];
                if (!runner)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP prefill schedule found a missing "
                        "stage at index "
                        << stage);
                    return false;
                }

                if (stage > 0u)
                {
                    TensorBase *const hidden =
                        pp_stage_runners_[stage - 1u]->getHiddenState();
                    if (!hidden)
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP prefill stage "
                            << (stage - 1u)
                            << " produced no activation owner");
                        return false;
                    }

                    // Only live producer rows cross the declared boundary.
                    // Resident bucket capacity remains local allocation storage.
                    if (!pp_activation_contract_)
                        throw std::logic_error("Pipeline prefill lacks its admitted residual transfer contract");
                    const std::size_t active_bytes = pipelinePrefillLiveBytes(
                        pp_activation_contract_->transfer(static_cast<int>(stage - 1u)), chunk);
                    if (!pp_ctx_->transfer(
                            hidden,
                            static_cast<int>(stage - 1u),
                            static_cast<int>(stage),
                            active_bytes))
                    {
                        LOG_ERROR(
                            "RankOrchestrator PP prefill activation transfer "
                            "failed from stage "
                            << (stage - 1u) << " to " << stage
                            << " for " << chunk.bucket_seq_len
                            << " physical rows");
                        return false;
                    }
                    transferred_rows += static_cast<std::uint64_t>(
                        chunk.bucket_seq_len);
                    runner->setHiddenState(hidden);
                }

                /*
                 * The root owns chunk boundaries and maintenance decisions.
                 * Each child receives exactly one fixed-width transaction at
                 * its own canonical cursor, preventing independent child
                 * schedulers from splitting a pipeline chunk differently.
                 */
                PrefillChunkSchedulerPolicy child_policy;
                child_policy.bucket_sizes = {chunk.bucket_seq_len};
                child_policy.fixed_chunk_real_tokens =
                    chunk.bucket_seq_len;
                child_policy.real_token_start = runner->get_position();
                child_policy.real_token_count = chunk.real_count;

                const bool child_ok = runner->forwardPrefillChunkSchedule(
                    tokens + relative_offset,
                    chunk.real_count,
                    child_policy,
                    pad_token_id,
                    allow_padded_execution);
                if (stage > 0u)
                    runner->clearHiddenStateInput();
                if (!child_ok)
                {
                    LOG_ERROR(
                        "RankOrchestrator PP prefill transaction failed at "
                        "stage "
                        << stage << " chunk=" << chunk.chunk_index);
                    return false;
                }
                ++completed_segments;
            }
            if (completed_segments != num_stages)
            {
                LOG_ERROR(
                    "RankOrchestrator PP prefill transaction did not traverse its frozen pipeline plan"
                    << " completed=" << completed_segments
                    << " expected=" << num_stages);
                std::terminate();
            }
            ++completed_transactions;
        }

        if (!skip_logits_gather_prefill_)
        {
            if (!logits_gatherer_)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    0, 0, logits_backend_resolver_);
                applyLogitsGatherSkipFlags();
            }
            logits_gatherer_->copyFromStage(
                *pp_stage_runners_.back());
        }

        recordCompletedPPGraphTransactions(
            "prefill", num_stages, completed_transactions);
        current_position_ += seq_len;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "pipeline_prefill_chunk_transactions",
            static_cast<double>(root_schedule.chunks.size()),
            "prefill",
            "rank",
            {{"stages", std::to_string(num_stages)},
             {"logical_rows", std::to_string(seq_len)},
             {"transferred_physical_rows",
              std::to_string(transferred_rows)},
             {"boundary_authority", "pipeline_root"}});
        return true;
    }

    // =========================================================================
    // TP Mode Forward Implementation (existing parallel execution)
    // =========================================================================
    int RankOrchestrator::effectiveTPWorkerJoinTimeoutMs() const
    {
        return collective_timeout_policy::effectiveWorkerJoinTimeoutMs(
            debugEnv().tp_collect_timeout_ms);
    }

    bool RankOrchestrator::forwardTP(
        const int *tokens,
        int seq_len,
        MainForwardDispatch dispatch,
        int restored_prefix_tokens,
        const PrefillChunkSchedulerPolicy *chunk_schedule_policy,
        int chunk_schedule_pad_token_id,
        bool chunk_schedule_allow_padded_execution)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardTP: No device runners available");
            return false;
        }
        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        const bool restored_prefix_bridge =
            dispatch == MainForwardDispatch::RestoredPrefixMTPDecodeBridge;
        if ((restored_prefix_bridge &&
             (!tokens || seq_len != 1 || restored_prefix_tokens <= 0)) ||
            (!restored_prefix_bridge && restored_prefix_tokens != 0) ||
            (chunk_schedule_policy &&
             dispatch != MainForwardDispatch::Automatic))
        {
            LOG_ERROR(
                "RankOrchestrator::forwardTP received an invalid typed main-forward dispatch"
                << " dispatch=" << static_cast<int>(dispatch)
                << " seq_len=" << seq_len
                << " restored=" << restored_prefix_tokens
                << " chunked=" << (chunk_schedule_policy != nullptr));
            return false;
        }

        const LogitsForwardPhase logits_phase =
            (dispatch == MainForwardDispatch::Prefill || seq_len != 1)
                ? LogitsForwardPhase::Prefill
                : LogitsForwardPhase::Decode;
        last_logits_forward_phase_ = logits_phase;

        /*
         * The MPI root and every follower execute this exact rank-local
         * boundary. Keeping the one-shot probe here gives distributed economy
         * calibration symmetric evidence without putting timing or policy on
         * the sparse device graph. Schedule geometry is planned only for an
         * explicitly chunked prefill, where the production path already owns
         * the same bounded host plan.
         */
        int probe_execution_rows = seq_len;
        int probe_transaction_count = 1;
        std::uint64_t probe_schedule_fingerprint = 0u;
        if (moe_overlay_interference_probe_ && chunk_schedule_policy)
        {
            const auto schedule = planPrefillChunkSchedule(
                *chunk_schedule_policy);
            if (schedule)
            {
                probe_execution_rows = 0;
                probe_transaction_count = static_cast<int>(
                    schedule.chunks.size());
                probe_schedule_fingerprint = 14695981039346656037ull;
                for (const auto &chunk : schedule.chunks)
                {
                    probe_execution_rows += chunk.bucket_seq_len;
                    for (const auto value : {
                             static_cast<std::uint64_t>(chunk.real_count),
                             static_cast<std::uint64_t>(chunk.bucket_seq_len)})
                    {
                        constexpr std::uint64_t kPrime = 1099511628211ull;
                        std::uint64_t mixed = value;
                        for (std::size_t byte = 0u;
                             byte < sizeof(mixed);
                             ++byte)
                        {
                            probe_schedule_fingerprint ^= mixed & 0xffu;
                            probe_schedule_fingerprint *= kPrime;
                            mixed >>= 8u;
                        }
                    }
                }
            }
        }
        if (probe_schedule_fingerprint == 0u)
        {
            probe_schedule_fingerprint = 14695981039346656037ull;
            for (const auto value : {
                     static_cast<std::uint64_t>(
                         logits_phase == LogitsForwardPhase::Decode
                             ? ExpertHistogramSource::DecodeToken
                             : ExpertHistogramSource::PrefillChunk),
                     static_cast<std::uint64_t>(seq_len),
                     static_cast<std::uint64_t>(probe_execution_rows),
                     static_cast<std::uint64_t>(probe_transaction_count),
                     std::uint64_t{0}})
            {
                constexpr std::uint64_t kPrime = 1099511628211ull;
                std::uint64_t mixed = value;
                for (std::size_t byte = 0u; byte < sizeof(mixed); ++byte)
                {
                    probe_schedule_fingerprint ^= mixed & 0xffu;
                    probe_schedule_fingerprint *= kPrime;
                    mixed >>= 8u;
                }
            }
        }
        const MoEOverlayInferenceWorkloadIdentity probe_workload{
            .source = logits_phase == LogitsForwardPhase::Decode
                          ? ExpertHistogramSource::DecodeToken
                          : ExpertHistogramSource::PrefillChunk,
            .real_rows = seq_len,
            .execution_rows = probe_execution_rows,
            .transaction_count = probe_transaction_count,
            .speculative_depth = 0,
            .schedule_fingerprint = probe_schedule_fingerprint,
        };
        if (chunk_schedule_policy &&
            moe_overlay_coordinator_owns_prefill_probe_)
        {
            std::string declaration_error;
            if (!moe_overlay_inference_transaction_coordinator_ ||
                !moe_overlay_inference_transaction_coordinator_
                     ->declarePrefillInterferenceSchedule(
                         probe_workload, &declaration_error))
            {
                LOG_ERROR(
                    "RankOrchestrator could not declare its complete ExpertOverlay prefill calibration schedule: "
                    << declaration_error);
                return false;
            }
        }
        MoEOverlayInferenceInterferenceScope interference_scope(
            chunk_schedule_policy &&
                    moe_overlay_coordinator_owns_prefill_probe_
                ? nullptr
                : moe_overlay_interference_probe_.get(),
            probe_workload);
        if (logits_gatherer_)
        {
            logits_gatherer_->invalidate();
        }

        // TP timing diagnostic: explicit executor instrumentation only.
        const bool tp_timing = debugEnv().tp_timing;
        const bool tp_profiling = debugEnv().execution.executor_profiling;
        const bool collect_timing = tp_timing || tp_profiling;
        auto tp_t0 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        LOG_DEBUG("RankOrchestrator::forwardTP: seq_len=" << seq_len
                                                          << ", devices=" << device_runners_.size());

        // Decode latency breakdown timing
        const bool decode_breakdown = collect_timing && seq_len == 1;
        auto launch_t0 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                          : std::chrono::high_resolution_clock::time_point{};
        auto launch_t1 = launch_t0; // Set properly after dispatch in parallel mode

        // DIAGNOSTIC: Run forward passes SEQUENTIALLY to test if concurrency causes crash.
        // If sequential execution works but parallel crashes, it's a concurrent HIP issue.
        const bool serialize_devices = debugEnv().runtime_debug.serialize_tp_forward;

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;

        if (serialize_devices)
        {
            // ----- SERIAL MODE (diagnostic fallback) -----
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                auto &runner = device_runners_[i];
                if (!runner)
                    continue;
                // Cast to IInferenceRunner* to disambiguate forward() overloads
                IInferenceRunner *runner_iface = runner.get();
                LOG_WARN("RankOrchestrator::forwardTP: SERIAL mode - device "
                         << i << " running synchronously");
                try
                {
                    bool ok = false;
                    if (chunk_schedule_policy)
                    {
                        ok = runner_iface->forwardPrefillChunkSchedule(
                            tokens,
                            seq_len,
                            *chunk_schedule_policy,
                            chunk_schedule_pad_token_id,
                            chunk_schedule_allow_padded_execution);
                    }
                    else if (restored_prefix_bridge)
                    {
                        ok = runner_iface->forwardRestoredPrefixMTPDecodeBridge(
                            {.token_id = tokens[0],
                             .restored_prefix_tokens =
                                 restored_prefix_tokens});
                    }
                    else if (dispatch == MainForwardDispatch::Prefill)
                    {
                        ok = runner_iface->forwardPrefill(tokens, seq_len);
                    }
                    else
                    {
                        ok = runner_iface->forward(tokens, seq_len);
                    }
                    if (!ok)
                    {
                        LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                  << i << " forward failed");
                        all_success = false;
                    }
                }
                catch (...)
                {
                    all_success = false;
                    if (!first_exception)
                    {
                        first_exception = std::current_exception();
                        first_exception_device = i;
                    }
                }
            }
        }
        else
        {
            // ----- PARALLEL MODE: persistent thread pool -----
            // Lazy-initialize on first TP forward call. Workers persist for the
            // lifetime of the orchestrator, eliminating per-step thread overhead.
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);

                // Wire abort callback: when one worker fails (exception or false return),
                // abort the collective backend to unblock any workers stuck in NCCL/RCCL
                // collective calls. Without this, a failed worker exits the forward pass
                // while other workers block forever in ncclAllReduce waiting for all ranks.
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback([this]()
                                                        {
                        LOG_WARN("[TPWorkerPool] Worker failure detected — aborting collective backend");
                        tp_ctx_->requestAbort(); });
                }

                LOG_DEBUG("[TPWorkerPool] Created " << device_runners_.size()
                                                    << " persistent worker threads");
            }

            // Dispatch parallel forward passes to persistent worker threads.
            // dispatch() wakes all workers via condition_variable and returns
            // immediately — no thread creation, no pthread_create syscall.
            //
            // Capture profiler phases from the caller thread and propagate to
            // workers. All profilers use thread-local phase storage, so worker
            // threads must explicitly set their phase to match the caller.
            auto kernel_phase = KernelProfiler::getCurrentPhase();
            auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            auto kv_phase = KVCacheProfiler::getCurrentPhase();
            auto executor_phase = GraphExecutorStats::currentPhase();

            std::vector<DeviceGraphOrchestrator *> pre_execution_rendezvous_runners;
            std::shared_ptr<ForwardGraphExecutionRendezvous> pre_execution_rendezvous;
            if (!tp_first_forward_completed_ && device_runners_.size() > 1)
            {
                pre_execution_rendezvous_runners.reserve(device_runners_.size());
                for (auto &runner : device_runners_)
                {
                    auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get());
                    if (dgo)
                        pre_execution_rendezvous_runners.push_back(dgo);
                }

                if (pre_execution_rendezvous_runners.size() == device_runners_.size())
                {
                    pre_execution_rendezvous =
                        std::make_shared<ForwardGraphExecutionRendezvous>(
                            device_runners_.size(),
                            "rank_tp_first_forward");
                    for (auto *dgo : pre_execution_rendezvous_runners)
                        dgo->setForwardGraphExecutionRendezvous(pre_execution_rendezvous);
                    LOG_DEBUG("RankOrchestrator::forwardTP: armed first-forward pre-execution rendezvous for "
                              << device_runners_.size() << " TP child graph(s)");
                }
                else if (!pre_execution_rendezvous_runners.empty())
                {
                    LOG_WARN("RankOrchestrator::forwardTP: skipping first-forward pre-execution rendezvous; "
                             << pre_execution_rendezvous_runners.size() << "/"
                             << device_runners_.size()
                             << " child runners expose DeviceGraphOrchestrator");
                    pre_execution_rendezvous_runners.clear();
                }
            }

            struct ScopedForwardExecutionRendezvousClear
            {
                std::vector<DeviceGraphOrchestrator *> runners;
                ~ScopedForwardExecutionRendezvousClear()
                {
                    for (auto *runner : runners)
                    {
                        if (runner)
                            runner->setForwardGraphExecutionRendezvous(nullptr);
                    }
                }
            } rendezvous_clear_guard{pre_execution_rendezvous_runners};

            tp_worker_pool_->dispatch(
            [this,
             tokens,
             seq_len,
             dispatch,
             restored_prefix_tokens,
             restored_prefix_bridge,
             chunk_schedule_policy,
             chunk_schedule_pad_token_id,
             chunk_schedule_allow_padded_execution,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
                {
                    // Propagate profiler phases from caller thread
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    // Set per-device profiler context so ROCm/CUDA profilers
                    // can attribute kernel times to specific GPUs
                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    IInferenceRunner *runner_iface = device_runners_[i].get();
                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " runner=" << static_cast<void *>(runner_iface)
                                  << " seq_len=" << seq_len);
                    }
                    try
                    {
                        bool ok = false;
                        if (chunk_schedule_policy)
                        {
                            ok = runner_iface->forwardPrefillChunkSchedule(
                                tokens,
                                seq_len,
                                *chunk_schedule_policy,
                                chunk_schedule_pad_token_id,
                                chunk_schedule_allow_padded_execution);
                        }
                        else if (restored_prefix_bridge)
                        {
                            ok = runner_iface
                                     ->forwardRestoredPrefixMTPDecodeBridge(
                                         {.token_id = tokens[0],
                                          .restored_prefix_tokens =
                                              restored_prefix_tokens});
                        }
                        else if (dispatch == MainForwardDispatch::Prefill)
                        {
                            ok = runner_iface->forwardPrefill(
                                tokens, seq_len);
                        }
                        else
                        {
                            ok = runner_iface->forward(tokens, seq_len);
                        }
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_leave"
                                      << " worker=" << i
                                      << " device=" << device_id.toString()
                                      << " success=" << (ok ? 1 : 0));
                        }
                        return ok;
                    }
                    catch (...)
                    {
                        if (debugEnv().tp_collective_contract_trace)
                        {
                            LOG_ERROR("[TP_WORKER_CONTRACT] event=forward_exception"
                                      << " worker=" << i
                                      << " device=" << device_id.toString());
                        }
                        throw;
                    }
                });

            launch_t1 = decode_breakdown ? std::chrono::high_resolution_clock::now()
                                         : std::chrono::high_resolution_clock::time_point{};

            // Collect results from all workers. LLAMINAR_TP_COLLECT_TIMEOUT_MS
            // is enforced inside LocalTP/NCCL/RCCL rendezvous calls; it must not
            // be treated as a wall-clock limit for the whole runner forward.
            const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            auto results = tp_worker_pool_->collectAll(collect_timeout_ms);

            // Process results with fault-tolerant exception handling.
            // IMPORTANT: Store the FIRST substantive exception. When one device
            // throws (e.g., VerificationFailure), it can cause CUDA/HIP context
            // destruction, making other devices fail with misleading "context is
            // destroyed" errors. We want to surface the original root cause.
            bool worker_timeout = false;
            for (auto &r : results)
            {
                if (!r.completed)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " did not complete (stuck)");
                    worker_timeout = true;
                    all_success = false;
                    if (collect_timeout_ms > 0)
                    {
                        abortAfterTPWorkerTimeout(
                            "forwardTP",
                            collect_timeout_ms,
                            tp_worker_pool_->completedCount(),
                            tp_worker_pool_->numWorkers());
                    }
                    if (tp_ctx_)
                    {
                        LOG_WARN("RankOrchestrator::forwardTP: requesting TP context abort after worker timeout");
                        tp_ctx_->requestAbort();
                    }
                    continue;
                }

                if (r.exception)
                {
                    all_success = false;
                    try
                    {
                        std::rethrow_exception(r.exception);
                    }
                    catch (const std::exception &e)
                    {
                        std::string error_msg = e.what();
                        bool is_context_destroyed =
                            (error_msg.find("context is destroyed") != std::string::npos ||
                             error_msg.find("context destroyed") != std::string::npos ||
                             error_msg.find("error 709") != std::string::npos);

                        if (!first_exception)
                        {
                            first_exception = r.exception;
                            first_exception_device = r.worker_index;
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw PRIMARY exception: " << error_msg);
                        }
                        else if (!is_context_destroyed)
                        {
                            // Substantive error — replace if first was a context error
                            try
                            {
                                std::rethrow_exception(first_exception);
                            }
                            catch (const std::exception &first_e)
                            {
                                std::string first_msg = first_e.what();
                                bool first_is_ctx =
                                    (first_msg.find("context is destroyed") != std::string::npos ||
                                     first_msg.find("context destroyed") != std::string::npos ||
                                     first_msg.find("error 709") != std::string::npos);
                                if (first_is_ctx)
                                {
                                    LOG_WARN("RankOrchestrator::forwardTP: Replacing "
                                             "secondary context error with primary error from device "
                                             << r.worker_index);
                                    first_exception = r.exception;
                                    first_exception_device = r.worker_index;
                                }
                            }
                            LOG_ERROR("RankOrchestrator::forwardTP: Device "
                                      << r.worker_index
                                      << " threw exception: " << error_msg);
                        }
                        else
                        {
                            LOG_WARN("RankOrchestrator::forwardTP: Device "
                                     << r.worker_index
                                     << " threw SECONDARY exception (likely due to primary failure): "
                                     << error_msg);
                        }
                    }
                }
                else if (!r.success)
                {
                    LOG_ERROR("RankOrchestrator::forwardTP: Device "
                              << r.worker_index << " forward failed");
                    all_success = false;
                }
            }
            if (worker_timeout && collect_timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "forwardTP",
                    collect_timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }

            if (!all_success && config_.moe_device_controller_fabric)
            {
                LOG_ERROR(
                    "[RankOrchestrator] Fatal LocalTP ExpertOverlay epoch evidence "
                    << config_.moe_device_controller_fabric
                           ->describeInferenceEpochBarrier());
            }

            // Capture timing for decode breakdown (only in parallel mode)
            if (decode_breakdown)
            {
                auto collect_t1 = std::chrono::high_resolution_clock::now();
                double launch_us = std::chrono::duration<double, std::micro>(
                                       launch_t1 - launch_t0)
                                       .count();
                double wait_us = std::chrono::duration<double, std::micro>(
                                     collect_t1 - launch_t1)
                                     .count();
                double total_us = std::chrono::duration<double, std::micro>(
                                      collect_t1 - launch_t0)
                                      .count();
                if (tp_timing)
                {
                    LOG_DEBUG("[DECODE_BREAKDOWN] launch=" << std::fixed << std::setprecision(1)
                                                           << launch_us << "us"
                                                           << " wait=" << wait_us << "us"
                                                           << " total=" << total_us << "us");
                }
            }
        }

        // If we captured an exception, re-throw it with full context
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        auto tp_t1 = collect_timing ? std::chrono::high_resolution_clock::now()
                                    : std::chrono::high_resolution_clock::time_point{};

        if (all_success)
        {
            // Gather logits from all devices (delegates to LogitsGatherer)
            const bool need_gather =
                ownsMainForwardLogits() &&
                logits_gatherer_ && logits_gatherer_->needsGather(logits_phase);

            if (!ownsMainForwardLogits())
            {
                PerfStatsCollector::addCounter(
                    "execution",
                    "rank_tp_hidden_only_forward_completions",
                    1.0,
                    {},
                    primaryDeviceId().toString(),
                    {{"boundary", "nested_non_head_pp_stage"},
                     {"phase",
                      logits_phase == LogitsForwardPhase::Decode
                          ? "decode"
                          : "prefill"},
                     {"host_logits_gathers", "0"}});
            }

            if (!need_gather && seq_len > 1)
            {
                static bool logged_prefill_skip = false;
                if (!logged_prefill_skip)
                {
                    LOG_DEBUG("[forwardTP] Skipping prefill logits gather (seq_len="
                              << seq_len << ") — prefill logits are not consumed");
                    logged_prefill_skip = true;
                }
            }

            // During prefill the LM head computes only 1 row of logits (the
            // last-token position) written to row 0 of logits_local.  We must
            // gather exactly 1 row; gathering seq_len rows would include
            // uninitialised data in rows 1..seq_len-1.
            const size_t gather_rows =
                logits_phase == LogitsForwardPhase::Prefill
                    ? 1
                    : static_cast<size_t>(seq_len);
            if (need_gather && !logits_gatherer_->gather(device_runners_, gather_rows, vocab_size()))
            {
                LOG_ERROR("RankOrchestrator::forwardTP: Failed to gather logits");
                all_success = false;
            }

            // Update position tracking
            current_position_ += seq_len;
            current_padded_seq_len_ = seq_len;
            if (current_sequence_lengths_.empty())
            {
                current_sequence_lengths_.assign(
                    static_cast<size_t>(std::max(1, config_.batch_size)),
                    current_position_ - seq_len);
            }
            for (int &length : current_sequence_lengths_)
            {
                length += seq_len;
            }
            stats_dirty_ = true;

            // After first prefill, release host-resident weight data.
            // GPU kernels (e.g., embedding repack) have now uploaded their own
            // device copies, and later MoE migration must use prepared/packed
            // weight transfer rather than falling back to raw GGUF bytes.
            if (!host_resident_released_ && seq_len > 1 && model_ctx_)
            {
                host_resident_released_ = true;
                if (auto wm = model_ctx_->weightManager())
                {
                    /*
                     * The worker owns residual host-buffer release, mmap host
                     * unregistration, backing-aware advice, and allocator
                     * trimming as one ordered operation.  Publishing this edge
                     * performs none of that maintenance on the inference
                     * authority thread.
                     */
                    const auto submission = wm->scheduleMmapReclaim();
                    if (debugEnv().vram_trace)
                    {
                        LOG_TRACE("[VRAM_TRACE] rank_mmap_reclaim phase=after_first_prefill submission="
                                  << MmapReclaimLifecycle::toString(submission));
                    }
                }
            }
        }

        if (collect_timing)
        {
            auto tp_t2 = std::chrono::high_resolution_clock::now();
            double forward_ms = std::chrono::duration<double, std::milli>(tp_t1 - tp_t0).count();
            double gather_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t1).count();
            double total_ms = std::chrono::duration<double, std::milli>(tp_t2 - tp_t0).count();

            if (tp_timing)
            {
                LOG_DEBUG("[TP_TIMING] seq_len=" << seq_len
                                                 << " forward=" << std::fixed << std::setprecision(3) << forward_ms << "ms"
                                                 << " gather=" << std::fixed << std::setprecision(3) << gather_ms << "ms"
                                                 << " total=" << std::fixed << std::setprecision(3) << total_ms << "ms");
            }

            // Accumulate TP decode stats for profiling summary at benchmark end.
            // dispatch_ms = time to wake workers, wait_ms = time blocked on collect.
            // These are computed from DECODE_BREAKDOWN timestamps which are only
            // available in parallel (non-serialized) mode for decode (seq_len==1).
            if (tp_profiling && seq_len == 1)
            {
                double dispatch_ms = 0, wait_ms = 0;
                if (decode_breakdown && !serialize_devices)
                {
                    dispatch_ms = std::chrono::duration<double, std::milli>(launch_t1 - launch_t0).count();
                    wait_ms = std::chrono::duration<double, std::milli>(tp_t1 - launch_t1).count();
                }
                else
                {
                    // Serial mode or timing unavailable: entire forward time is wait
                    wait_ms = forward_ms;
                }
                tp_decode_stats_.record(total_ms, dispatch_ms, wait_ms, gather_ms);
            }
        }

        if (all_success)
            tp_first_forward_completed_ = true;

        source_attempt.keep=all_success;
        return all_success;
    }

    // =========================================================================
    // PP Mode Forward Implementation (sequential pipeline execution)
    // =========================================================================
    bool RankOrchestrator::forwardPP(
        const int *tokens,
        int seq_len,
        MainForwardDispatch dispatch,
        int restored_prefix_tokens)
    {
        if (pp_stage_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No PP stage runners available");
            return false;
        }

        if (!pp_ctx_)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: No LocalPPContext available for transfers");
            return false;
        }
        if (!validatePPGraphTransactionReady(
                "RankOrchestrator PP forward"))
        {
            return false;
        }

        const bool restored_prefix_bridge =
            dispatch == MainForwardDispatch::RestoredPrefixMTPDecodeBridge;
        if ((restored_prefix_bridge &&
             (!tokens || seq_len != 1 || restored_prefix_tokens <= 0)) ||
            (!restored_prefix_bridge && restored_prefix_tokens != 0))
        {
            LOG_ERROR(
                "RankOrchestrator::forwardPP received an invalid typed main-forward dispatch"
                << " dispatch=" << static_cast<int>(dispatch)
                << " seq_len=" << seq_len
                << " restored=" << restored_prefix_tokens);
            return false;
        }

        const LogitsForwardPhase logits_phase =
            (dispatch == MainForwardDispatch::Prefill || seq_len != 1)
                ? LogitsForwardPhase::Prefill
                : LogitsForwardPhase::Decode;
        last_logits_forward_phase_ = logits_phase;
        if (logits_gatherer_)
        {
            logits_gatherer_->invalidate();
        }

        const size_t num_stages =
            pp_graph_execution_plan_->segmentCount();
        std::size_t completed_segments = 0u;

        if (pp_device_generation_ && seq_len == 1 && dispatch != MainForwardDispatch::Prefill)
        {
            // Frozen GPU pipeline edges already own activation movement.
            // Enter the same concurrent domain authority as prefill/verifiers;
            // the CPU stage loop cannot drive these captured channels.
            if (!tokens)
            {
                LOG_ERROR("RankOrchestrator::forwardPP scalar input requires its request token");
                return false;
            }
            const bool submitted = restored_prefix_bridge
                ? pp_device_generation_->restoredPrefixMTPDecodeBridge(
                      {.token_id = tokens[0], .restored_prefix_tokens = restored_prefix_tokens})
                : pp_device_generation_->forwardDecodeInput(tokens[0]);
            if (!submitted)
                return false;
            if (!skip_logits_gather_decode_)
            {
                if (!logits_gatherer_)
                {
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(0, 0, logits_backend_resolver_);
                    applyLogitsGatherSkipFlags();
                }
                logits_gatherer_->copyFromStage(*pp_stage_runners_.back());
            }
            recordCompletedPPGraphTransactions("decode", num_stages, 1);
            ++current_position_;
            stats_dirty_ = true;
            PerfStatsCollector::addCounter("forward_graph", "pipeline_scalar_input_transactions", 1.0,
                "decode", "rank", {{"boundary_authority", "captured_native_pipeline_edges"},
                    {"state_transition", restored_prefix_bridge ? "main_and_shifted_mtp" : "main_only"}});
            return true;
        }

        LOG_DEBUG("RankOrchestrator::forwardPP: seq_len=" << seq_len
                                                          << " num_stages=" << num_stages);

        // =====================================================================
        // Stage 0: Embedding + first layers (receives tokens as input)
        // =====================================================================
        auto &stage0_runner = pp_stage_runners_[0];
        if (!stage0_runner)
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 runner is null");
            return false;
        }

        // PP device coherence: After PP transfer in the previous iteration, the
        // hidden state tensor's gpu_device_ may point to the *destination* device
        // (e.g., ROCm:1) instead of this stage's device (ROCm:0). Promote the
        // secondary buffer back to primary so kernels and transitionToWithEvent
        // operate on the correct device.
        if (pp_ctx_ && num_stages > 1)
        {
            TensorBase *hidden = stage0_runner->getHiddenState();
            if (hidden)
            {
                DeviceId stage0_device = pp_ctx_->deviceForStage(0).toLocalDeviceId();
                auto cur = hidden->current_device();
                if (cur.has_value() && *cur != stage0_device)
                {
                    LOG_DEBUG("RankOrchestrator::forwardPP: Re-asserting hidden state device from "
                              << cur->toString() << " to " << stage0_device.toString());
                    hidden->allocateOnDevice(stage0_device);
                }
            }
        }

        LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage 0 (embedding)");
        const auto execute_stage =
            [&](IInferenceRunner &runner, const int *stage_tokens)
        {
            if (restored_prefix_bridge)
            {
                return runner.forwardRestoredPrefixMTPDecodeBridge(
                    {.token_id = tokens[0],
                     .restored_prefix_tokens = restored_prefix_tokens});
            }
            if (dispatch == MainForwardDispatch::Prefill)
                return runner.forwardPrefill(stage_tokens, seq_len);
            return runner.forward(stage_tokens, seq_len);
        };
        if (!execute_stage(*stage0_runner, tokens))
        {
            LOG_ERROR("RankOrchestrator::forwardPP: Stage 0 forward failed");
            return false;
        }
        ++completed_segments;

        // =====================================================================
        // Intermediate stages: Transfer activations and continue execution
        // =====================================================================
        for (std::size_t plan_index = 1u;
             plan_index < pp_graph_execution_plan_->segments().size();
             ++plan_index)
        {
            const size_t stage_idx =
                pp_graph_execution_plan_->segments()[plan_index].stage_index;
            auto &prev_runner = pp_stage_runners_[stage_idx - 1];
            auto &curr_runner = pp_stage_runners_[stage_idx];

            if (!curr_runner)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " runner is null");
                return false;
            }

            // Get hidden state from previous stage
            // DeviceGraphOrchestrator stores hidden state in InferenceState
            TensorBase *hidden_state = prev_runner->getHiddenState();
            if (!hidden_state)
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << (stage_idx - 1)
                                                                << " has no hidden state to transfer");
                return false;
            }

            // Transfer activations from previous stage to current stage
            // Only transfer the active region (seq_len tokens), not the full buffer
            if (!pp_activation_contract_)
                throw std::logic_error("Pipeline forward lacks its admitted residual transfer contract");
            const size_t active_bytes = pp_activation_contract_
                ->transfer(static_cast<int>(stage_idx - 1)).activeBytes(seq_len);

            LOG_DEBUG("RankOrchestrator::forwardPP: Transferring hidden state from stage "
                      << (stage_idx - 1) << " to stage " << stage_idx
                      << " (" << active_bytes << " bytes for " << seq_len << " tokens)");

            if (!pp_ctx_->transfer(hidden_state, static_cast<int>(stage_idx - 1),
                                   static_cast<int>(stage_idx), active_bytes))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Transfer from stage "
                          << (stage_idx - 1) << " to stage " << stage_idx << " failed");
                return false;
            }

            // Set the hidden state as input for current stage
            curr_runner->setHiddenState(hidden_state);

            LOG_DEBUG("RankOrchestrator::forwardPP: Executing stage " << stage_idx);

            /*
             * This explicitly host-driven transaction owns immutable token
             * ingress for every participant, just like the prefill scheduler.
             * Intermediate GPUs must admit that live row into their own request
             * bank for native token consumers; a hidden activation does not
             * replace the token authority. has_embedding=false still selects
             * the transferred activation and never repeats embedding. Native
             * parent-owned decode takes the separate device-generation branch
             * above and does not enter this host-I/O boundary.
             */
            if (!execute_stage(*curr_runner, tokens))
            {
                LOG_ERROR("RankOrchestrator::forwardPP: Stage " << stage_idx << " forward failed");
                return false;
            }
            ++completed_segments;

            // Clear hidden state input for clean state on next forward
            curr_runner->clearHiddenStateInput();
        }

        // =====================================================================
        // Copy logits from last stage to combined buffer
        // =====================================================================
        const bool gather_host_logits =
            logits_phase == LogitsForwardPhase::Decode
                ? !skip_logits_gather_decode_
                : !skip_logits_gather_prefill_;
        if (gather_host_logits)
        {
            int last_stage = static_cast<int>(num_stages - 1);
            if (last_stage >= 0 && static_cast<size_t>(last_stage) < pp_stage_runners_.size() && pp_stage_runners_[last_stage])
            {
                if (!logits_gatherer_)
                {
                    logits_gatherer_ = std::make_unique<LogitsGatherer>(
                        0,
                        0,
                        logits_backend_resolver_);
                    applyLogitsGatherSkipFlags();
                }
                logits_gatherer_->copyFromStage(*pp_stage_runners_[last_stage]);
            }
        }

        // =====================================================================
        // Update position tracking for PP mode
        // Each PP stage runner updates its own position internally, but we also
        // need to update current_position_ for consistency with TP mode and
        // get_position() queries.
        // =====================================================================
        recordCompletedPPGraphTransactions(
            logits_phase == LogitsForwardPhase::Decode
                ? "decode"
                : "prefill",
            completed_segments);
        current_position_ += seq_len;

        LOG_DEBUG("RankOrchestrator::forwardPP: Complete, all " << num_stages << " stages executed"
                                                                << ", position now " << current_position_);
        return true;
    }

    IInferenceRunner *RankOrchestrator::finalPPSidecarRunner()
    {
        if (pp_stage_runners_.empty())
            return nullptr;
        return pp_stage_runners_.back().get();
    }

    const IInferenceRunner *RankOrchestrator::finalPPSidecarRunner() const
    {
        if (pp_stage_runners_.empty())
            return nullptr;
        return pp_stage_runners_.back().get();
    }

    void RankOrchestrator::refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication()
    {
        /*
         * The child runners own the actual published state.  RankOrchestrator
         * mirrors a small public bookkeeping surface for callers that cannot
         * delegate to a single child, so after MTP publication we adopt the
         * primary child's accepted-prefix boundary instead of keeping the
         * verifier graph's temporary row count.
         */
        const IInferenceRunner *primary = nullptr;
        if (!pp_stage_runners_.empty())
        {
            primary = pp_stage_runners_.front().get();
        }
        else if (!device_runners_.empty())
        {
            primary = device_runners_.front().get();
        }

        if (!primary)
        {
            current_position_ = 0;
            current_batch_size_ = std::max(1, config_.batch_size);
            current_padded_seq_len_ = 0;
            current_sequence_lengths_.assign(
                static_cast<size_t>(current_batch_size_),
                0);
            stats_dirty_ = true;
            return;
        }

        current_position_ = primary->get_position();
        current_batch_size_ = std::max(1, primary->batch_size());
        current_padded_seq_len_ = primary->padded_seq_len();

        const std::vector<int> &child_lengths = primary->sequence_lengths();
        if (!child_lengths.empty())
        {
            current_sequence_lengths_ = child_lengths;
        }
        else
        {
            current_sequence_lengths_.assign(
                static_cast<size_t>(current_batch_size_),
                current_position_);
        }
        if (current_sequence_lengths_.size() <
            static_cast<size_t>(current_batch_size_))
        {
            current_sequence_lengths_.resize(
                static_cast<size_t>(current_batch_size_),
                current_position_);
        }
        stats_dirty_ = true;
    }

    bool RankOrchestrator::consumeUnusedReplicatedMainLogitsPublications(
        const char *boundary)
    {
        if (device_runners_.size() < 2)
            return true;

        size_t consumed_participants = 0;
        for (size_t participant = 1;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *runner = device_runners_[participant].get();
            if (!runner)
            {
                LOG_ERROR("[RankOrchestrator] Replicated main-logits publication "
                          "has no participant " << participant
                          << " at boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }

            /*
             * CPU execution is synchronous and owns no graph-stream token.
             * A GPU participant must explicitly implement the retirement API;
             * the interface default fails so an unrecognized nested runner
             * cannot silently leak or discard an ordering edge.
             */
            if (!runner->primaryDeviceId().is_gpu())
                continue;
            if (runner->hasLogitsLocal())
            {
                LOG_ERROR("[RankOrchestrator] Cannot retire a sharded main-logits "
                          "publication as an unused replica participant="
                          << participant << " boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }
            if (!runner->consumeUnusedReplicatedMainLogitsPublication())
            {
                LOG_ERROR("[RankOrchestrator] Failed to retire unused replicated "
                          "main-logits publication participant="
                          << participant << " boundary="
                          << (boundary ? boundary : "unknown"));
                return false;
            }
            ++consumed_participants;
        }

        if (consumed_participants > 0)
        {
            PerfStatsCollector::addCounter(
                "sampling",
                "rank_unused_replicated_main_logits_publications_consumed",
                static_cast<double>(consumed_participants),
                "decode",
                "rank",
                {{"boundary", boundary ? boundary : "unknown"},
                 {"participants", std::to_string(device_runners_.size())},
                 {"sampler_participant", "0"}});
        }
        return true;
    }

    int RankOrchestrator::sampleGreedyOnDevice()
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyOnDevice();
        }
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (all_local_logits)
        {
            /*
             * CPU LocalTP has no graph-stream handoff to consume: the current
             * LOGITS_LOCAL tensors are ordinary host-resident shard rows.  Use
             * the non-consuming local views directly so ready prefill logits and
             * MTP condition logits are reduced by the same grouped argmax used
             * for verifier rows.  GPU children still route through
             * DeviceSampler::sampleGreedy(), whose consuming view enforces the
             * producer stream contract before launching per-shard argmax.
             */
            std::vector<LogitsLocalInfo> host_local_infos;
            host_local_infos.reserve(device_runners_.size());
            bool all_host_local_infos = true;
            for (const auto &runner : device_runners_)
            {
                LogitsLocalInfo info = runner ? runner->getLogitsLocalInfo() : LogitsLocalInfo{};
                if (!info || info.vocab_local == 0)
                {
                    all_host_local_infos = false;
                    break;
                }
                if (info.device.has_value() && info.device->is_gpu())
                {
                    all_host_local_infos = false;
                    break;
                }
                host_local_infos.push_back(info);
            }
            if (all_host_local_infos)
            {
                return DeviceSampler::sampleGreedyFromLocalInfos(
                    host_local_infos,
                    /*row=*/0);
            }
            return DeviceSampler::sampleGreedy(device_runners_);
        }
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
        {
            const int token = device_runners_[0]->sampleGreedyOnDevice();
            if (token < 0 ||
                !consumeUnusedReplicatedMainLogitsPublications(
                    "sampleGreedyOnDevice"))
            {
                return -1;
            }
            return token;
        }
        return -1;
    }

    int RankOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        return sampleOnDeviceAtLogicalPosition(params, get_position());
    }

    int RankOrchestrator::sampleOnDeviceAtLogicalPosition(
        const SamplingParams &params,
        int logical_position)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleOnDeviceAtLogicalPosition(
                params,
                logical_position);
        }
        if (params.is_greedy())
            return sampleGreedyOnDevice();
        if (mode_ != ParallelismMode::TP || device_runners_.size() < 2)
            return -1;

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (all_local_logits)
        {
            const float threshold =
                params.seed != 0
                    ? sampling_math::mtp_spec_threshold_from_seed(
                          static_cast<uint64_t>(params.seed),
                          logical_position,
                          /*draw_purpose=*/0)
                    : sampling_math::uniform01(
                          0xD1B54A32D192ED03ull,
                          static_cast<uint64_t>(std::max(0, logical_position)));
            return DeviceSampler::sample(device_runners_, params, threshold);
        }
        if (!any_local_logits && !device_runners_.empty() && device_runners_[0])
        {
            const int token =
                device_runners_[0]->sampleOnDeviceAtLogicalPosition(
                    params,
                    logical_position);
            if (token < 0 ||
                !consumeUnusedReplicatedMainLogitsPublications(
                    "sampleOnDeviceAtLogicalPosition"))
            {
                return -1;
            }
            return token;
        }
        return -1;
    }
} // namespace llaminar2
