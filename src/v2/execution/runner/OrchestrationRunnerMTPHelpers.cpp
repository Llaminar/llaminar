/**
 * @file OrchestrationRunnerMTPHelpers.cpp
 * @brief OrchestrationRunner MTP planning/completion helpers, replay oracle, ordinary decode, and rebalance.
 */

#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{
    MTPRankParticipation OrchestrationRunner::mtpRankParticipation() const
    {
        // The actual installed outer pipeline service owns stage scheduling.
        // Its follower can also host local expert machinery; that does not make
        // the original layer stage a vocabulary or dynamic-depth participant.
        if (const auto *original = dynamic_cast<const GlobalOrchestrator *>(runner_.get()))
        {
            if (original->originalRankGenerationFollower())
                return MTPRankParticipation::OriginalPipelineFollower;
            if (original->originalRankGenerationSource())
                return MTPRankParticipation::OriginalPipelineContinuation;
        }
        if (moe_overlay_inference_transaction_coordinator_)
            return MTPRankParticipation::ExpertTransactionContinuation;
        if (moe_overlay_inference_transaction_follower_)
            return MTPRankParticipation::ExpertTransactionFollower;
        return mpi_ctx_ && mpi_ctx_->world_size() > 1 ?
            MTPRankParticipation::ModelCollectivePeer : MTPRankParticipation::SingleRank;
    }

    bool OrchestrationRunner::shouldUseMTPDecode() const
    {
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        return mtp.enabled &&
               mtpDecodeHardFailureReason().empty() &&
               mtpDecodeBypassReason().empty();
    }

    std::string OrchestrationRunner::mtpDecodeHardFailureReason() const
    {
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (!mtp.enabled || !runner_)
            return {};

        if (!active_sampling_params_.is_greedy() &&
            mtp.verify_mode == MTPVerifyMode::Greedy)
        {
            return "MTP greedy verification requires greedy sampling; use --mtp-verify-mode speculative-sampling for non-greedy requests";
        }

        const int effective_max_draft_tokens = effectiveMTPMaxDraftDepth(mtp);
        if (effective_max_draft_tokens < 1)
        {
            return "MTP decode requires --mtp-draft-tokens >= 1";
        }
        if (effective_max_draft_tokens > 1 && !runner_->supportsChainedMTPDrafts())
        {
            return "MTP decode with --mtp-draft-tokens > 1 requires runner support for chained MTP sidecars";
        }
        if (mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy())
        {
            // Actual original pipeline/expert transaction owners distinguish
            // layer/expert followers from full-model vocabulary peers.
            if (const auto failure = mtpStochasticVerificationOwnerFailure(
                    plan_, *runner_, mtpRankParticipation()); !failure.empty())
                return std::string(failure);
        }
        if (runner_->primaryDeviceId().is_rocm() && debugEnv().rocm.concurrent_decode)
        {
            return "ROCm MTP decode is incompatible with LLAMINAR_ROCM_CONCURRENT_DECODE";
        }
        return {};
    }

    std::string OrchestrationRunner::mtpDecodeBypassReason() const
    {
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (!mtp.enabled)
        {
            return "feature disabled";
        }
        if (!runner_)
        {
            return "runner unavailable";
        }
        const std::string runner_reason = runner_->mtpDecodeUnsupportedReason();
        if (!runner_reason.empty())
        {
            return runner_reason;
        }
        /*
         * Generic MPI worlds still need an explicit token-coordination
         * contract. Original pipeline and ExpertOverlay transaction services
         * retain one continuation token/MTP authority. Followers execute their
         * immutable retained-graph tickets without a token collective.
         */
        if (mtpRankRequiresModelCollectives(mtpRankParticipation()) &&
            !runner_->supportsMTPTokenCoordination())
        {
            return "MTP decode is not enabled for MPI world_size > 1";
        }
        return {};
    }

    void OrchestrationRunner::recordMTPBypass(const std::string &reason)
    {
        if (reason.empty() || reason == "feature disabled")
        {
            return;
        }
        mtp_bypassed_ = true;
        mtp_bypass_reason_ = reason;
        if (!mtp_bypass_recorded_for_request_)
        {
            ++mtp_stats_.bypasses;
            mtp_bypass_recorded_for_request_ = true;
            LOG_DEBUG("[OrchestrationRunner] MTP bypassed: " << reason);
        }
    }

    int OrchestrationRunner::effectiveMTPMaxDraftDepth(const MTPRuntimeConfig &mtp) const
    {
        /*
         * This helper governs executable policy width, not retained graph
         * capacity. A fixed-depth request admitted into a wider reusable graph
         * family must still propose exactly its configured number of drafts.
         */
        return resolveMTPMaximumExecutionDraftDepth(mtp);
    }

    bool OrchestrationRunner::ensureMTPDepthController(const MTPRuntimeConfig &mtp)
    {
        try
        {
            if (!mtp_depth_controller_)
            {
                MTPDepthPolicyConfig depth_policy = resolveMTPDepthPolicyConfig(mtp);
                mtp_depth_controller_ =
                    std::make_unique<MTPDepthController>(
                        depth_policy,
                        mtp.draft_tokens,
                        mtp.verify_mode);
            }
            return true;
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Invalid MTP depth policy: ") + e.what());
        }
    }

    int OrchestrationRunner::currentMTPDraftDepth(const MTPRuntimeConfig &mtp)
    {
        if (!ensureMTPDepthController(mtp) || !mtp_depth_controller_)
        {
            return std::max(1, mtp.draft_tokens);
        }

        int depth = mtp_depth_controller_->requestedDepthForStep();

        /*
         * Dynamic depth is a request-level scheduling decision.  In NodeTP /
         * GlobalTP every rank must execute the same sidecar/verifier shape in the
         * same order, so the coordinated root's controller decision is the scalar
         * source of truth and broadcast before the step begins.  This mirrors the
         * vLLM-style contract: the speculative batch shape is coordinated once,
         * while tensor data still moves through the graph and collective layers.
         */
        if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
            mtpRankRequiresModelCollectives(mtpRankParticipation()))
        {
            int32_t coordinated_depth =
                mpi_ctx_->rank() == mpi_coordinated_root_rank_
                    ? static_cast<int32_t>(depth)
                    : 0;
            mpi_ctx_->broadcast_int32(
                &coordinated_depth, 1, mpi_coordinated_root_rank_);
            depth = static_cast<int>(coordinated_depth);

            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_mpi_depth_broadcasts",
                1.0,
                "decode",
                {},
                {{"depth", std::to_string(depth)},
                 {"rank", std::to_string(mpi_ctx_->rank())},
                 {"world_size", std::to_string(mpi_ctx_->world_size())}});
        }

        return depth;
    }

    bool OrchestrationRunner::isHostedDynamicMTPContinuation(
        const MTPRuntimeConfig &mtp) const
    {
        return runner_ && runner_->primaryDeviceId().is_gpu() &&
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
            device_generation_admission_.awaitsAdmission() &&
            device_generation_admission_.admissionKind() ==
                sampling_math::DeviceGenerationAdmissionKind::ContinueResponse &&
            runner_->deviceGenerationExecutionPolicy(
                DeviceGenerationLoopTopology::DynamicDepth) ==
                DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
    }

    bool OrchestrationRunner::initializeDecodeTransactionPlanningPositionAfterPrefill(
        int committed_tokens,
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
        {
            decode_transaction_planning_position_.reset();
            device_generation_admission_.reset();
            return true;
        }
        if (committed_tokens < 0)
        {
            decode_transaction_planning_position_.reset();
            return setError(
                "GPU MTP prefill produced a negative transaction planning position");
        }

        /*
         * The request scheduler already owns the exact number of prompt tokens
         * committed by prefill. This is control-plane transaction metadata,
         * not a mirror copied back from GPU KV state. Establishing it at the
         * successful prefill boundary gives the first speculative graph the
         * same position that later device publications advance from compact
         * accepted-count metadata.
        */
        decode_transaction_planning_position_ = committed_tokens;
        /*
         * Admission is a property of the successful GPU prefill boundary, not
         * of whichever decode policy happens to consume that boundary. The
         * first decode path closes or consumes it; stochastic MTP additionally
         * publishes the resident generation budget. Keeping MTP policy out of
         * this initializer preserves one transaction-position lifecycle for
         * ordinary, greedy-MTP, and stochastic-MTP decode.
         */
        device_generation_admission_.reset();
        if (!device_generation_admission_.openAfterPrefill())
        {
            return setError(
                "GPU prefill could not open its unique device-generation admission boundary");
        }
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "gpu_decode_transaction_position_initializations",
            1.0,
            {static_cast<uint64_t>(committed_tokens)},
            "prefill",
            {},
            {{"source", source && source[0] != '\0' ? source : "unknown"},
             {"position_owner", "request_scheduler"}});
        return true;
    }

    bool OrchestrationRunner::admitScalarDeviceResidentGeneration(
        bool grouped_device_verify,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition)
    {
        if (!grouped_device_verify)
            return true;
        if (!device_generation_admission_.awaitsAdmission())
        {
            return setError(
                "GPU grouped MTP generation reached a decode boundary without an admission-ready controller lifecycle");
        }
        if (!runner_ || !decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU grouped MTP generation admission has no initialized prefill boundary");
        }
        int response_budget = decode_step_token_budget_;
        if (response_budget <= 0)
        {
            const int configured_context =
                plan_.runtime.max_seq_len > 0
                    ? plan_.runtime.max_seq_len
                    : config_.max_seq_len;
            response_budget =
                configured_context - *decode_transaction_planning_position_;
        }
        if (response_budget <= 0)
        {
            return setError(
                "GPU grouped MTP generation admission has no positive response capacity");
        }

        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1,
            .max_new_tokens = response_budget,
            .kind = device_generation_admission_.admissionKind(),
            .depth_policy = resolveMTPDeviceGenerationDepthPolicy(
                activeMTPRequestConfig()),
            .initial_leading_row_disposition =
                initial_leading_row_disposition,
            .sampling_seeds = generation_request_seeds_,
            .initial_target_counts = DeviceGenerationInitialTargetCounts(
                std::vector<int>{*decode_transaction_planning_position_}),
        };
        if (!runner_->beginDeviceResidentGeneration(admission))
        {
            return setError(
                "Failed to admit the device-resident GPU generation response ledger");
        }

        if (!device_generation_admission_.admitController(response_budget))
        {
            return setError(
                "GPU grouped MTP generation could not bind its response budget to the admitted controller");
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_admission_boundaries",
            1.0,
            "decode",
            {},
            {{"request_count", "1"},
             {"response_budget", std::to_string(response_budget)},
             {"initial_leading_committed_output_count",
              std::to_string(
                  sampling_math::
                      device_generation_leading_committed_output_count(
                          initial_leading_row_disposition))},
             {"boundary", "first_scalar_grouped_mtp_decode"}});
        return true;
    }

    bool OrchestrationRunner::admitRequestBatchDeviceResidentGeneration(
        int request_count)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu() ||
            request_count <= 1 ||
            static_cast<int>(batched_request_states_.size()) != request_count)
        {
            return setError(
                "Request-batched device generation admission requires one complete GPU request set");
        }
        if (!device_generation_admission_.awaitsAdmission())
        {
            return setError(
                "Request-batched device generation reached sampling without an open prefill admission boundary");
        }

        int maximum_committed_tokens = 0;
        // These completed-prefill origins include every request's restored
        // history; no live device counter is copied into admission metadata.
        std::vector<int> initial_target_counts;
        initial_target_counts.reserve(static_cast<size_t>(request_count));
        std::vector<uint64_t> position_seeds;
        if (!active_sampling_params_.is_greedy())
            position_seeds.reserve(static_cast<size_t>(request_count));
        for (const BatchedDecodeRequestState &state : batched_request_states_)
        {
            if (state.logical_tokens <= 0 || state.is_complete ||
                !state.prefill_logits_ready)
            {
                return setError(
                    "Request-batched device generation admission requires every prefill row to be live and unconsumed");
            }
            maximum_committed_tokens =
                std::max(maximum_committed_tokens, state.logical_tokens);
            initial_target_counts.push_back(state.logical_tokens);
            if (!active_sampling_params_.is_greedy())
                position_seeds.push_back(state.stochastic_position_seed);
        }

        int response_budget = decode_step_token_budget_;
        if (response_budget <= 0)
        {
            const int configured_context =
                plan_.runtime.max_seq_len > 0
                    ? plan_.runtime.max_seq_len
                    : config_.max_seq_len;
            response_budget = configured_context - maximum_committed_tokens;
        }
        if (response_budget <= 0)
        {
            return setError(
                "Request-batched device generation admission has no common positive response capacity");
        }

        // Prefill already resolved each row's entropy. Reuse those exact seeds
        // in the admitted bank; generating fresh values here would give the
        // retained verifier a different sampling law from the leading token.
        std::optional<GenerationRequestSeeds> sampling_seeds;
        if (!position_seeds.empty())
            sampling_seeds.emplace(std::move(position_seeds));
        if (!runner_->beginDeviceResidentGeneration(
                DeviceGenerationAdmissionRequest{
                    .request_count = request_count,
                    .max_new_tokens = response_budget,
                    .kind = device_generation_admission_.admissionKind(),
                    .depth_policy = resolveMTPDeviceGenerationDepthPolicy(
                        activeMTPRequestConfig()),
                    .initial_leading_row_disposition =
                        sampling_math::
                            DeviceGenerationLeadingRowDisposition::
                                PendingResponse,
                    .sampling_seeds = std::move(sampling_seeds),
                    .initial_target_counts = DeviceGenerationInitialTargetCounts(
                        std::move(initial_target_counts)),
                }))
        {
            return setError(
                "Failed to admit the request-batched device generation response ledger");
        }

        if (!device_generation_admission_.admitController(response_budget))
        {
            return setError(
                "Request-batched device generation could not bind its response budget to the admitted controller");
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_admission_boundaries",
            1.0,
            "decode",
            {},
            {{"request_count", std::to_string(request_count)},
             {"response_budget", std::to_string(response_budget)},
             {"boundary", "request_batch_prefill_to_grouped_mtp"}});
        return true;
    }

    bool OrchestrationRunner::advanceDecodeTransactionPlanningPositionAfterForward(
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
            return true;
        if (!decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU decode forward has no initialized scheduler-owned transaction position");
        }

        ++(*decode_transaction_planning_position_);
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "gpu_decode_transaction_position_forward_advances",
            1.0,
            {static_cast<uint64_t>(*decode_transaction_planning_position_)},
            "decode",
            {},
            {{"source", source && source[0] != '\0' ? source : "unknown"},
             {"position_owner", "orchestration_transaction"}});
        return true;
    }

    bool OrchestrationRunner::publishDecodeTransactionPlanningPositionAfterMTPCommit(
        int transaction_base_tokens,
        int committed_rows,
        const char *source)
    {
        if (!runner_ || !runner_->primaryDeviceId().is_gpu())
        {
            decode_transaction_planning_position_.reset();
            return true;
        }
        if (transaction_base_tokens < 0 || committed_rows < 0)
        {
            return setError(
                "GPU MTP commit produced a negative scheduler transaction coordinate");
        }
        if (!decode_transaction_planning_position_.has_value() ||
            *decode_transaction_planning_position_ < 0)
        {
            return setError(
                "GPU MTP commit has no initialized scheduler-owned transaction position");
        }
        if (*decode_transaction_planning_position_ != transaction_base_tokens)
        {
            std::ostringstream error;
            error << "GPU MTP commit transaction base does not match the "
                     "scheduler-owned position: scheduler="
                  << *decode_transaction_planning_position_
                  << " transaction_base=" << transaction_base_tokens;
            return setError(error.str());
        }

        /*
         * The transaction validator has already proved that these rows are the
         * decode-equivalent continuation of transaction_base_tokens.  This
         * scalar is scheduler metadata used to plan the next captured launch;
         * it is not a host mirror of KV, GDN, terminal-hidden, or outcome
         * storage.  In particular, do not condition this publication on the
         * lifetime of a transient device outcome mailbox.
         */
        decode_transaction_planning_position_ =
            transaction_base_tokens + committed_rows;
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "gpu_decode_transaction_position_publications",
            1.0,
            {static_cast<uint64_t>(*decode_transaction_planning_position_)},
            "decode",
            {},
            {{"path", source && source[0] != '\0' ? source : "unknown"},
             {"advanced_tokens", std::to_string(committed_rows)},
             {"position_owner", "orchestration_transaction"}});
        return true;
    }

    std::optional<int> OrchestrationRunner::currentDecodeTransactionPositionForPlanning(
        const char *context,
        std::string *error) const
    {
        if (!runner_)
        {
            if (error)
                *error = "MTP sidecar position planning requires an initialized runner";
            return std::nullopt;
        }

        if (runner_->primaryDeviceId().is_gpu())
        {
            if (decode_transaction_planning_position_.has_value() &&
                *decode_transaction_planning_position_ >= 0)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "mtp",
                    "gpu_decode_transaction_position_reads",
                    1.0,
                    {static_cast<uint64_t>(*decode_transaction_planning_position_)},
                    "decode",
                    {},
                    {{"context", context ? context : "unknown"},
                     {"position_owner", "orchestration_transaction"}});
                return *decode_transaction_planning_position_;
            }

            std::string message =
                "GPU decode planning has no scheduler-owned transaction position";
            if (context && context[0] != '\0')
                message += std::string(" for ") + context;
            if (error)
                *error = std::move(message);
            return std::nullopt;
        }

        const int position = runner_->get_position();
        if (position < 0)
        {
            if (error)
                *error = "MTP sidecar position planning received a negative host position";
            return std::nullopt;
        }

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "cpu_decode_position_planning_reads",
            1.0,
            {static_cast<uint64_t>(position)},
            "decode",
            {},
            {{"context", context ? context : "unknown"},
             {"position_owner", "cpu_runner"}});
        return position;
    }

    void OrchestrationRunner::recordMTPDepthZeroBypass()
    {
        if (!mtp_depth_controller_)
        {
            return;
        }
        const MTPDepthDecision decision = mtp_depth_controller_->recordBypassStep();
        mtp_stats_.current_depth = mtp_depth_controller_->currentDepth();
        mtp_stats_.min_depth = mtp_depth_controller_->minDepth();
        mtp_stats_.max_depth = mtp_depth_controller_->maxDepth();

        PerfStatsCollector::addCounter(
            "mtp",
            "depth_policy_zero_depth_bypasses",
            1.0,
            "decode",
            {},
            {{"current_depth", std::to_string(decision.new_depth)},
             {"next_requested_depth", std::to_string(mtp_depth_controller_->requestedDepthForStep())},
             {"reason", toString(decision.reason)}});
    }

    void OrchestrationRunner::recordMTPDepthObservation(
        int requested_depth,
        int effective_depth,
        int accepted_speculative_prefix,
        bool budget_limited,
        bool rollback,
        const MTPPublicationWitness &publication)
    {
        if (!mtp_depth_controller_)
        {
            return;
        }
        const auto before = mtp_depth_controller_->stats();
        const bool trace = debugEnv().runtime_debug.mtp_depth_transaction_trace;
        const MTPDepthObservation observation{
            .requested_depth = requested_depth,
            .effective_depth = effective_depth,
            .accepted_speculative_prefix = accepted_speculative_prefix,
            .budget_limited = budget_limited,
            .rollback = rollback,
            .newly_emitted_tokens = publication.newlyEmittedTokens(),
            .publication = publication,
        };
        if (trace && mtp_stats_.verifier_runs == 1)
            publishCPUDepthTransactionTraceInitial(*mtp_depth_controller_,
                activeMTPRequestConfig().draft_tokens, publication, runner_->primaryDeviceId().toString(),
                "benchmark_single_request");
        const MTPDepthDecision decision = mtp_depth_controller_->recordStep(observation);
        const auto after = mtp_depth_controller_->stats();

        mtp_stats_.depth_policy_windows += after.windows - before.windows;
        mtp_stats_.depth_policy_updates += after.updates - before.updates;
        mtp_stats_.depth_policy_promotions += after.promotions - before.promotions;
        mtp_stats_.depth_policy_demotions += after.demotions - before.demotions;
        mtp_stats_.depth_policy_observe_recommendations +=
            after.observe_recommendations - before.observe_recommendations;
        mtp_stats_.depth_policy_utility_probes += after.utility_probes - before.utility_probes;
        mtp_stats_.depth_policy_utility_keeps += after.utility_keeps - before.utility_keeps;
        mtp_stats_.depth_policy_utility_returns += after.utility_returns - before.utility_returns;
        mtp_stats_.current_depth = mtp_depth_controller_->currentDepth();
        mtp_stats_.min_depth = mtp_depth_controller_->minDepth();
        mtp_stats_.max_depth = mtp_depth_controller_->maxDepth();

        if (trace)
        {
            const auto record = makeCPUDepthTransactionTraceRecord(observation, decision,
                mtp_stats_.verifier_runs, after.windows);
            // Physical verifier count is the sole ordinal. A benchmark case
            // owns exactly one fresh measured request before collector reset.
            publishMTPDepthTransactionRecord(record, static_cast<int>(mtp_stats_.verifier_runs),
                runner_->primaryDeviceId().toString(), "benchmark_single_request", "cpu_host");
        }
        if (decision.evaluated)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_windows",
                1.0,
                "decode",
                {},
                {{"old_depth", std::to_string(decision.old_depth)},
                 {"new_depth", std::to_string(decision.new_depth)},
                 {"recommended_depth", std::to_string(decision.recommended_depth)},
                 {"reason", toString(decision.reason)},
                 {"changed", decision.changed ? "true" : "false"},
                 {"observe_recommendation", decision.observe_recommendation ? "true" : "false"},
                 {"acceptance_rate", std::to_string(decision.acceptance_rate)},
                 {"zero_accept_rate", std::to_string(decision.zero_accept_rate)},
                 {"full_accept_rate", std::to_string(decision.full_accept_rate)},
                 {"window_size", std::to_string(decision.window.verifier_runs)}});
        }
        if (decision.changed)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                decision.new_depth > decision.old_depth
                    ? "depth_policy_promotions"
                    : "depth_policy_demotions",
                1.0,
                "decode",
                {},
                {{"old_depth", std::to_string(decision.old_depth)},
                 {"new_depth", std::to_string(decision.new_depth)},
                 {"reason", toString(decision.reason)}});
        }
        else if (decision.observe_recommendation)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_policy_observe_recommendations",
                1.0,
                "decode",
                {},
                {{"current_depth", std::to_string(decision.old_depth)},
                 {"recommended_depth", std::to_string(decision.recommended_depth)},
                 {"reason", toString(decision.reason)}});
        }
    }

    bool OrchestrationRunner::noteEmbeddedDeviceMoEOverlayMaintenance(
        uint64_t transaction_count,
        DeviceGenerationExecutionPolicy execution_policy)
    {
        if (!runner_ ||
            runner_->moeOverlayAuthorityExecution() !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident ||
            !runner_->deviceResidentMoEOverlayMaintenanceReady())
        {
            return true;
        }
        if (transaction_count == 0)
        {
            return setError(
                "Device-resident ExpertOverlay generation reported no committed maintenance boundary");
        }
        if (device_generation_embedded_moe_maintenance_pending_ack_)
        {
            return setError(
                "Device-resident ExpertOverlay generation crossed a second outer step before acknowledging embedded maintenance");
        }

        device_generation_embedded_moe_maintenance_pending_ack_ = true;
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_generation_embedded_maintenance_boundaries",
            static_cast<double>(transaction_count),
            "maintenance",
            runner_->primaryDeviceId().toString(),
            {{"authority_execution", "device_resident"},
             {"execution_policy",
              deviceGenerationExecutionPolicyName(execution_policy)},
             {"outer_host_launch", "false"},
             {"acknowledgement_pending", "true"}});
        return true;
    }

    GenerationBatchResult
    OrchestrationRunner::completeDeviceResidentBatchGeneration(
        const DeviceSpeculativePublicationRequest &publication_request,
        DeviceGenerationSamplingMode sampling_mode,
        int requested_draft_depth,
        int capture_draft_depth,
        GenerationBatchResult result)
    {
        const auto fail = [&](std::string message) -> GenerationBatchResult
        {
            result.error = std::move(message);
            return result;
        };
        if (!runner_ ||
            !isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !publication_request.valid() ||
            !publication_request.outcome.device_generation_controller_owned ||
            publication_request.requestCount() <= 1)
        {
            return fail(
                "Request-batched device generation requires one valid controller-owned compact outcome per request");
        }

        const int request_count = publication_request.requestCount();
        const int admitted_token_budget =
            device_generation_admission_.tokenBudget();
        if (admitted_token_budget <= 0 ||
            static_cast<int>(batched_request_states_.size()) != request_count)
        {
            return fail(
                "Request-batched device generation has no matching admitted response budget and request set");
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        const DeviceGenerationLoopTopology loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            runner_->deviceGenerationExecutionPolicy(loop_topology);
        if (execution_policy == DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail(
                "Request-batched device generation has no complete captured execution policy");
        }
        const int parent_draft_depth =
            publication_request.logicalVerifierRowsPerRequest() - 1;
        if (parent_draft_depth <= 0 || capture_draft_depth <= 0 ||
            parent_draft_depth != capture_draft_depth ||
            !runner_->materializeDeviceResidentGeneration(
                request_count,
                parent_draft_depth,
                loop_topology,
                sampling_mode))
        {
            return fail(
                "Request-batched device generation could not materialize its complete captured graph family");
        }
        if (!runner_->launchDeviceResidentGeneration())
        {
            return fail(
                "Request-batched device generation could not launch its captured parent");
        }

        DeviceGenerationTerminalResult terminal;
        if (!runner_->finishDeviceResidentGeneration(&terminal) ||
            !terminal.valid() ||
            static_cast<int>(terminal.requests.size()) != request_count)
        {
            return fail(
                "Request-batched device generation did not produce one valid terminal ledger per request");
        }

        result.requests.resize(static_cast<size_t>(request_count));
        uint64_t total_output_tokens = 0;
        uint64_t total_transactions = 0;
        uint64_t total_accepted = 0;
        uint64_t total_rejected = 0;
        uint64_t total_consumed_rows = 0;
        uint64_t total_attempted_drafts = 0;
        uint64_t total_verifier_tokens = 0;
        uint64_t total_depth_windows = 0;
        uint64_t total_depth_updates = 0;
        uint64_t total_depth_promotions = 0;
        uint64_t total_depth_utility_probes = 0;
        uint64_t total_depth_utility_keeps = 0;
        uint64_t total_depth_utility_returns = 0;
        uint64_t total_depth_demotions = 0;
        int final_depth = 0;

        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const DeviceGenerationTerminalRequestResult &request =
                terminal.requests[static_cast<size_t>(request_index)];
            const int response_count =
                static_cast<int>(request.tokens.size());
            if (response_count <= 0 ||
                request.remaining_token_count < 0 ||
                response_count + request.remaining_token_count !=
                    admitted_token_budget ||
                (request.remaining_token_count > 0 && !request.model_stopped) ||
                request.transaction_count <= 0 ||
                request.published_state_commit_count < 0 ||
                request.published_state_commit_count > response_count + 1 ||
                request.attempted_draft_token_count <= 0 ||
                request.verifier_token_count !=
                    request.attempted_draft_token_count +
                        request.transaction_count)
            {
                return fail(
                    "Request-batched terminal ledger disagrees with its admitted response, publication, or depth budget for request " +
                    std::to_string(request_index));
            }

            GenerationResult &request_result =
                result.requests[static_cast<size_t>(request_index)];
            request_result.tokens = request.tokens;
            request_result.is_complete = request.model_stopped;

            total_output_tokens += static_cast<uint64_t>(response_count);
            total_transactions +=
                static_cast<uint64_t>(request.transaction_count);
            total_accepted += static_cast<uint64_t>(
                request.accepted_speculative_token_count);
            total_rejected += static_cast<uint64_t>(
                request.rejected_transaction_count);
            total_consumed_rows += static_cast<uint64_t>(
                request.consumed_verifier_row_count);
            total_attempted_drafts += static_cast<uint64_t>(
                request.attempted_draft_token_count);
            total_verifier_tokens += static_cast<uint64_t>(
                request.verifier_token_count);
            total_depth_windows += static_cast<uint64_t>(
                request.depth_evaluated_window_count);
            total_depth_updates += static_cast<uint64_t>(
                request.depth_update_count);
            total_depth_promotions += static_cast<uint64_t>(
                request.depth_promotion_count);
            total_depth_demotions += static_cast<uint64_t>(
                request.depth_demotion_count);
            total_depth_utility_returns += static_cast<uint64_t>(request.depth_utility_return_count);
            total_depth_utility_keeps += static_cast<uint64_t>(request.depth_utility_keep_count);
            total_depth_utility_probes += static_cast<uint64_t>(request.depth_utility_probe_count);
            final_depth = std::max(final_depth, request.final_draft_depth);
        }

        const bool stochastic =
            sampling_mode == DeviceGenerationSamplingMode::Stochastic;
        mtp_stats_.draft_steps += total_attempted_drafts;
        mtp_stats_.verifier_runs += total_transactions;
        mtp_stats_.verifier_token_count += total_verifier_tokens;
        mtp_stats_.accepted_tokens += total_accepted;
        mtp_stats_.rejected_tokens += total_rejected;
        mtp_stats_.transaction_commits += total_transactions;
        mtp_stats_.depth_policy_windows += total_depth_windows;
        mtp_stats_.depth_policy_updates += total_depth_updates;
        mtp_stats_.depth_policy_promotions += total_depth_promotions;
        mtp_stats_.depth_policy_demotions += total_depth_demotions;
        mtp_stats_.depth_policy_utility_returns += total_depth_utility_returns;
        mtp_stats_.depth_policy_utility_keeps += total_depth_utility_keeps;
        mtp_stats_.depth_policy_utility_probes += total_depth_utility_probes;
        mtp_stats_.current_depth = final_depth;
        if (stochastic)
        {
            mtp_stats_.rollbacks += total_rejected;
            mtp_stats_.transaction_rollbacks += total_rejected;
            mtp_stats_.stochastic_accept_tests += total_consumed_rows;
            mtp_stats_.stochastic_accepts += total_accepted;
            mtp_stats_.stochastic_residual_samples += total_rejected;
        }

        const PerfStatsCollector::Tags terminal_tags{
            {"path", "request_batch_device_resident_generation_loop"},
            {"execution_policy",
             deviceGenerationExecutionPolicyName(execution_policy)},
            {"sampling", deviceGenerationSamplingModeName(sampling_mode)},
            {"requests", std::to_string(request_count)},
            {"requested_depth", std::to_string(requested_draft_depth)},
            {"capture_depth", std::to_string(capture_draft_depth)},
            {"final_depth", std::to_string(final_depth)},
            {"host_transaction_materializations", "0"}};
        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_generation_requests",
            static_cast<double>(request_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "output_tokens",
            static_cast<double>(total_output_tokens),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            stochastic
                ? "grouped_decode_equivalent_stochastic_verifier_runs"
                : "grouped_decode_equivalent_greedy_verifier_runs",
            static_cast<double>(total_transactions),
            "decode",
            {},
            terminal_tags);

        if (!device_generation_admission_.controllerActive())
        {
            return fail(
                "Request-batched terminal ledger lost its active controller lifecycle before retirement");
        }
        if (!noteEmbeddedDeviceMoEOverlayMaintenance(
                total_transactions, execution_policy))
        {
            return fail(
                last_error_.empty()
                    ? "Request-batched device generation could not publish embedded ExpertOverlay maintenance ownership"
                    : last_error_);
        }
        device_generation_admission_.reset();
        device_generation_terminal_ledger_authoritative_ = true;
        clearBatchedDecodeState();
        return result;
    }

    GenerationResult OrchestrationRunner::completeDeviceResidentGeneration(
        const DeviceSpeculativePublicationRequest &publication_request,
        DeviceGenerationSamplingMode sampling_mode,
        int transaction_base_cached_tokens,
        int requested_draft_depth,
        int initial_transaction_draft_depth,
        GenerationResult result)
    {
        const auto fail = [&](const std::string &message) -> GenerationResult
        {
            result.error = message;
            return result;
        };
        if (!isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !publication_request.valid() ||
            !publication_request.outcome.device_generation_controller_owned)
        {
            return fail(
                "Device-resident MTP generation requires one valid controller-owned compact outcome and sampling topology");
        }

        const char *const sampling_name =
            deviceGenerationSamplingModeName(sampling_mode);
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        const DeviceGenerationLoopTopology loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy execution_policy =
            runner_->deviceGenerationExecutionPolicy(loop_topology);
        if (execution_policy ==
            DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP has no complete generation-loop execution policy");
        }

        const int first_transaction_depth = publication_request.logicalVerifierRowsPerRequest() - 1;
        const int parent_draft_depth = mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
            ? effectiveMTPMaxDraftDepth(mtp) : initial_transaction_draft_depth;
        const bool hosted_exact = execution_policy == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
        if (first_transaction_depth <= 0 || initial_transaction_draft_depth <= 0 ||
            first_transaction_depth != initial_transaction_draft_depth ||
            first_transaction_depth > parent_draft_depth ||
            (hosted_exact && first_transaction_depth != requested_draft_depth) ||
            (!hosted_exact && first_transaction_depth != parent_draft_depth) ||
            !runner_->materializeDeviceResidentGeneration(
                publication_request.requestCount(),
                parent_draft_depth,
                loop_topology,
                sampling_mode))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not materialize its complete device-generation graph family");
        }

        if (!runner_->launchDeviceResidentGeneration())
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not launch its device-generation graph");
        }

        return finishScalarDeviceResidentGeneration(
            sampling_mode, transaction_base_cached_tokens,
            DeviceGenerationFirstTransaction::External,
            requested_draft_depth, initial_transaction_draft_depth,
            std::move(result));
    }

    GenerationResult OrchestrationRunner::finishScalarDeviceResidentGeneration(
        DeviceGenerationSamplingMode sampling_mode,
        int transaction_base_cached_tokens,
        DeviceGenerationFirstTransaction first_transaction,
        int requested_draft_depth,
        int initial_transaction_draft_depth,
        GenerationResult result)
    {
        const auto fail = [&](const std::string &message) -> GenerationResult
        {
            result.error = message;
            return result;
        };
        const bool external_first =
            first_transaction == DeviceGenerationFirstTransaction::External;
        const bool retained_first =
            first_transaction == DeviceGenerationFirstTransaction::AdmissionTicket;
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        const auto execution_policy = runner_->deviceGenerationExecutionPolicy(
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth);
        if (!isValidDeviceGenerationSamplingMode(sampling_mode) ||
            !device_generation_admission_.controllerActive() ||
            (!external_first && !retained_first) ||
            (external_first && (requested_draft_depth <= 0 ||
                                initial_transaction_draft_depth <= 0)) ||
            (retained_first && (requested_draft_depth != 0 ||
                                initial_transaction_draft_depth != 0 ||
                mtp.depth_policy.mode != MTPDepthPolicyMode::Dynamic ||
                execution_policy != DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)))
            return fail("Device-resident MTP terminal completion has an invalid first-transaction authority");
        const char *const sampling_name = deviceGenerationSamplingModeName(sampling_mode);
        const bool stochastic = sampling_mode == DeviceGenerationSamplingMode::Stochastic;
        const int parent_draft_depth = mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
            ? effectiveMTPMaxDraftDepth(mtp) : initial_transaction_draft_depth;

        DeviceGenerationTerminalResult terminal;
        if (!runner_->finishDeviceResidentGeneration(&terminal) ||
            !terminal.valid() || terminal.requests.size() != 1u)
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not materialize one valid terminal device ledger");
        }

        const DeviceGenerationTerminalRequestResult &request_result =
            terminal.requests.front();
        const int response_count =
            static_cast<int>(request_result.tokens.size());
        const int admitted_token_budget =
            device_generation_admission_.tokenBudget();
        if (admitted_token_budget <= 0 || response_count <= 0 ||
            request_result.remaining_token_count < 0 ||
            !sampling_math::
                valid_device_generation_leading_row_disposition(
                    request_result.next_leading_row_disposition) ||
            (request_result.model_stopped &&
             request_result.next_leading_row_disposition !=
                 sampling_math::DeviceGenerationLeadingRowDisposition::
                     PendingResponse) ||
            response_count + request_result.remaining_token_count !=
                admitted_token_budget ||
            (request_result.remaining_token_count > 0 &&
             !request_result.model_stopped) ||
            request_result.transaction_count <= 0 ||
            request_result.published_state_commit_count < 0 ||
            request_result.published_state_commit_count > response_count + 1 ||
            (!request_result.model_stopped &&
             request_result.published_state_commit_count <= 0))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP terminal ledger disagrees with the admitted response/state budget: response_count=" +
                std::to_string(response_count) +
                " remaining_token_count=" +
                std::to_string(request_result.remaining_token_count) +
                " admitted_token_budget=" +
                std::to_string(admitted_token_budget) +
                " model_stopped=" +
                (request_result.model_stopped ? "true" : "false") +
                " next_leading_committed_output_count=" +
                std::to_string(
                    sampling_math::
                        device_generation_leading_committed_output_count(
                            request_result.next_leading_row_disposition)) +
                " transaction_count=" +
                std::to_string(request_result.transaction_count) +
                " published_state_commit_count=" +
                std::to_string(
                    request_result.published_state_commit_count));
        }
        if (!publishDecodeTransactionPlanningPositionAfterMTPCommit(
                transaction_base_cached_tokens,
                request_result.published_state_commit_count,
                "device_resident_generation_loop"))
        {
            return fail(
                last_error_.empty()
                    ? std::string("Device-resident ") + sampling_name +
                          " MTP could not publish its terminal scheduler position"
                    : last_error_);
        }

        std::optional<ReadyMTPCondition> terminal_ready_condition;
        if (!request_result.model_stopped)
        {
            /*
             * Exhausting the caller's response budget is not a model-state
             * boundary. The final compact publication has already committed
             * the exact verifier prefix and sampled the token that belongs at
             * the next logical position. finishDeviceResidentGeneration()
             * leaves that mailbox live by contract; preserve the authenticated
             * handle itself and never infer the continuation from the last
             * host-visible response token.
             */
            DeviceResidentLogicalSequenceStateHandle resident_condition =
                runner_->deviceResidentLogicalSequenceState();
            if (!resident_condition.coversRequest(0) ||
                resident_condition.device != terminal.device ||
                resident_condition.device != runner_->primaryDeviceId())
            {
                return fail(
                    std::string("Device-resident ") + sampling_name +
                    " MTP terminal ledger has no matching live continuation mailbox");
            }
            terminal_ready_condition =
                ReadyMTPCondition::deviceResident(
                    active_sampling_params_,
                    std::move(resident_condition),
                    request_result.next_leading_row_disposition);
            if (!terminal_ready_condition->valid())
            {
                return fail(
                    std::string("Device-resident ") + sampling_name +
                    " MTP terminal continuation is incomplete");
            }
        }

        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        prefill_logits_ready_ = terminal_ready_condition.has_value();
        ready_mtp_condition_ = std::move(terminal_ready_condition);

        for (const int32_t token : request_result.tokens)
        {
            sampler_.record_token(token);
            result.tokens.push_back(token);
        }
        last_token_ = request_result.tokens.back();
        result.is_complete = request_result.model_stopped;

        const uint64_t transactions =
            static_cast<uint64_t>(request_result.transaction_count);
        const uint64_t accepted = static_cast<uint64_t>(
            request_result.accepted_speculative_token_count);
        const uint64_t rejected = static_cast<uint64_t>(
            request_result.rejected_transaction_count);
        const uint64_t consumed_rows = static_cast<uint64_t>(
            request_result.consumed_verifier_row_count);
        if (request_result.attempted_draft_token_count <= 0 ||
            request_result.verifier_token_count <= 0 ||
            request_result.verifier_token_count !=
                request_result.attempted_draft_token_count +
                    request_result.transaction_count ||
            (external_first && mtp_stats_.draft_steps <
                static_cast<uint64_t>(initial_transaction_draft_depth)))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP terminal depth ledger is internally inconsistent");
        }

        /*
         * Only a fresh externally submitted first transaction contributed host
         * draft submissions. A retained continuation starts from native ticket
         * zero and contributes no such work to subtract. In both cases the sole
         * controller's terminal totals account for every useful draft once.
         */
        if (external_first)
            mtp_stats_.draft_steps -=
                static_cast<uint64_t>(initial_transaction_draft_depth);
        mtp_stats_.draft_steps += static_cast<uint64_t>(
            request_result.attempted_draft_token_count);
        mtp_stats_.verifier_runs += transactions;
        mtp_stats_.verifier_token_count += static_cast<uint64_t>(
            request_result.verifier_token_count);
        mtp_stats_.last_transaction_draft_depth =
            request_result.last_transaction_draft_depth;
        mtp_stats_.last_transaction_emitted_token_count =
            request_result.last_transaction_emitted_token_count;
        mtp_stats_.accepted_tokens += accepted;
        mtp_stats_.rejected_tokens += rejected;
        if (stochastic)
        {
            mtp_stats_.rollbacks += rejected;
            mtp_stats_.transaction_rollbacks += rejected;
            mtp_stats_.stochastic_accept_tests += consumed_rows;
            mtp_stats_.stochastic_accepts += accepted;
            mtp_stats_.stochastic_residual_samples += rejected;
        }
        mtp_stats_.transaction_commits += transactions;
        mtp_stats_.depth_policy_windows += static_cast<uint64_t>(
            request_result.depth_evaluated_window_count);
        mtp_stats_.depth_policy_updates += static_cast<uint64_t>(
            request_result.depth_update_count);
        mtp_stats_.depth_policy_promotions += static_cast<uint64_t>(
            request_result.depth_promotion_count);
        mtp_stats_.depth_policy_demotions += static_cast<uint64_t>(
            request_result.depth_demotion_count);
        mtp_stats_.depth_policy_utility_returns += static_cast<uint64_t>(request_result.depth_utility_return_count);
        mtp_stats_.depth_policy_utility_keeps += static_cast<uint64_t>(request_result.depth_utility_keep_count);
        mtp_stats_.depth_policy_utility_probes += static_cast<uint64_t>(request_result.depth_utility_probe_count);
        mtp_stats_.current_depth = request_result.final_draft_depth;
        if (!device_generation_admission_.retireController(
                request_result.model_stopped))
        {
            return fail(
                std::string("Device-resident ") + sampling_name +
                " MTP could not retire its controller into the next explicit request phase");
        }

        PerfStatsCollector::Tags terminal_tags{
            {"path", "device_resident_generation_loop"},
            {"execution_policy",
             deviceGenerationExecutionPolicyName(execution_policy)},
            {"sampling", sampling_name},
            {"first_transaction_source", external_first ? "external" : "admission_ticket"},
            {"capture_depth", std::to_string(parent_draft_depth)},
            {"final_depth",
             std::to_string(request_result.final_draft_depth)},
            {"depth_evaluated_windows",
             std::to_string(request_result.depth_evaluated_window_count)},
            {"depth_updates",
             std::to_string(request_result.depth_update_count)},
            {"depth_promotions",
             std::to_string(request_result.depth_promotion_count)},
            {"depth_demotions",
             std::to_string(request_result.depth_demotion_count)},
            {"depth_policy_utility_returns", std::to_string(request_result.depth_utility_return_count)},
            {"depth_policy_utility_keeps", std::to_string(request_result.depth_utility_keep_count)},
            {"depth_policy_utility_probes", std::to_string(request_result.depth_utility_probe_count)},
            {"transactions",
             std::to_string(request_result.transaction_count)},
            {"state_commits",
             std::to_string(request_result.published_state_commit_count)}};
        if (external_first)
        {
            terminal_tags.emplace("depth", std::to_string(requested_draft_depth));
            terminal_tags.emplace("initial_transaction_depth", std::to_string(initial_transaction_draft_depth));
        }
        if (retained_first)
            PerfStatsCollector::addCounter(
                "mtp", "device_generation_retained_continuation_requests", 1.0,
                "decode", {}, terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_resident_generation_requests",
            1.0,
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "output_tokens",
            static_cast<double>(response_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_runs",
            static_cast<double>(transactions),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_tokens",
            static_cast<double>(request_result.verifier_token_count),
            "decode",
            {},
            terminal_tags);

        const std::string grouped_route_counter =
            stochastic
                ? "grouped_decode_equivalent_stochastic_verifier_runs"
                : "grouped_decode_equivalent_greedy_verifier_runs";
        const std::string verifier_path =
            stochastic
                ? "grouped_decode_equivalent_stochastic"
                : "grouped_decode_equivalent_greedy";
        PerfStatsCollector::Tags grouped_tags{
            {"execution", deviceGenerationExecutionPolicyName(execution_policy)},
            {"sampling", sampling_name},
            {"first_transaction_source", external_first ? "external" : "admission_ticket"},
            {"verifier_forward_tokens", std::to_string(request_result.verifier_token_count)},
            {"replay_forward_tokens", "0"},
            {"accepted_tokens", std::to_string(accepted)},
            {"state_publication", "device_resident"}};
        if (external_first)
            grouped_tags.emplace("verifier_rows", std::to_string(initial_transaction_draft_depth + 1));
        PerfStatsCollector::addCounter(
            "mtp", grouped_route_counter, static_cast<double>(transactions),
            "decode", {}, grouped_tags);

        if (stochastic)
        {
            PerfStatsCollector::Tags stochastic_tags = terminal_tags;
            stochastic_tags.emplace("device_resident", "true");
            stochastic_tags.emplace("verifier_path", verifier_path);
            stochastic_tags.emplace(
                "implementation",
                "device_resident_generation_terminal_ledger");
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_accept_tests",
                static_cast<double>(consumed_rows),
                "decode",
                {},
                stochastic_tags);
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_accepts",
                static_cast<double>(accepted),
                "decode",
                {},
                stochastic_tags);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "depth_policy_windows",
            static_cast<double>(request_result.depth_evaluated_window_count),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "accepted_tokens",
            static_cast<double>(accepted),
            "decode",
            {},
            terminal_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "rejected_tokens",
            static_cast<double>(rejected),
            "decode",
            {},
            terminal_tags);
        if (!noteEmbeddedDeviceMoEOverlayMaintenance(
                transactions, execution_policy))
        {
            return fail(
                last_error_.empty()
                    ? std::string("Device-resident ") + sampling_name +
                          " MTP could not publish embedded ExpertOverlay maintenance ownership"
                    : last_error_);
        }
        device_generation_terminal_ledger_authoritative_ = true;
        return result;
    }

    std::optional<std::string> OrchestrationRunner::verifyCommittedPrefixReplay(
        bool verify_commit_replay_check,
        bool use_ready_logits,
        const MTPRuntimeConfig &mtp,
        int32_t condition_token,
        int base_sidecar_position,
        const std::optional<PrefixStateSnapshot> &verifier_replay_base_checkpoint,
        const char *path,
        const std::vector<int32_t> &tokens_to_replay,
        int32_t expected_next_token,
        int state_advanced_token_count,
        const std::string &debug_context)
    {
        auto join_tokens = [](const std::vector<int32_t> &tokens) -> std::string
        {
            std::string out;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i > 0)
                    out += ",";
                out += std::to_string(tokens[i]);
            }
            return out;
        };
            if (!verify_commit_replay_check)
            {
                return std::nullopt;
            }
            if (tokens_to_replay.empty() ||
                !verifier_replay_base_checkpoint.has_value())
            {
                return std::nullopt;
            }

            const bool ready_token_was_provided = expected_next_token >= 0;
            const int state_advanced_count =
                state_advanced_token_count >= 0
                    ? std::clamp(
                          state_advanced_token_count,
                          0,
                          static_cast<int>(tokens_to_replay.size()))
                    : static_cast<int>(tokens_to_replay.size());
            /*
             * The compact all-position verifier publishes state in verifier-row
             * space, not in outer response-token space.  When the verifier also
             * produces a ready token, the correction suffix has already been
             * accounted for by the ready-token contract and the continuation
             * oracle should replay the full committed output stream.  When no
             * ready token exists, the suffix is still a pending condition token
             * and must be fed before comparing continuations.
             */
            const int replay_prefix_count =
                ready_token_was_provided
                    ? static_cast<int>(tokens_to_replay.size())
                    : state_advanced_count;
            const std::vector<int32_t> state_replay_tokens(
                tokens_to_replay.begin(),
                tokens_to_replay.begin() + replay_prefix_count);
            const std::vector<int32_t> pending_condition_inputs(
                tokens_to_replay.begin() + replay_prefix_count,
                tokens_to_replay.end());

            if (!runner_->ensureMTPCheckpointTerminalHidden())
            {
                return std::string("MTP commit replay check could not materialize committed terminal hidden");
            }
            PrefixStateSnapshot committed_checkpoint =
                runner_->captureLivePrefixState();
            if (!committed_checkpoint.valid)
            {
                return std::string("MTP commit replay check could not capture committed state");
            }
            auto summarize_probe = [](const PrefixRuntimeStateSnapshot &probe)
            {
                auto summarize_cache = [](const std::vector<PrefixKVCacheProbe> &caches)
                {
                    std::string out;
                    for (const auto &cache : caches)
                    {
                        if (!out.empty())
                            out += ";";
                        out += cache.owner + ":";
                        const size_t limit = std::min<size_t>(cache.layers.size(), 6);
                        for (size_t i = 0; i < limit; ++i)
                        {
                            if (i > 0)
                                out += ",";
                            const auto &layer = cache.layers[i];
                            out += "L" + std::to_string(layer.global_layer) +
                                   "/S" + std::to_string(layer.seq_idx) +
                                   "=" + std::to_string(layer.cached_tokens) +
                                   "@" + std::to_string(layer.ring_head);
                            if (layer.payload_hash_available)
                            {
                                out += "/kb=" + std::to_string(layer.k_payload_bytes) +
                                       "/vb=" + std::to_string(layer.v_payload_bytes) +
                                       "/kh=" + std::to_string(layer.k_payload_hash) +
                                       "/vh=" + std::to_string(layer.v_payload_hash);
                            }
                        }
                        if (cache.layers.size() > limit)
                            out += ",...";
                    }
                    return out.empty() ? std::string("none") : out;
                };

                std::string positions;
                for (size_t i = 0; i < probe.positions.size(); ++i)
                {
                    if (i > 0)
                        positions += ",";
                    positions += std::to_string(probe.positions[i]);
                }
                std::string seqs;
                for (size_t i = 0; i < probe.sequence_lengths.size(); ++i)
                {
                    if (i > 0)
                        seqs += ",";
                    seqs += std::to_string(probe.sequence_lengths[i]);
                }
                std::string gdn;
                const size_t gdn_limit = std::min<size_t>(probe.gdn_layers.size(), 4);
                for (size_t i = 0; i < gdn_limit; ++i)
                {
                    if (i > 0)
                        gdn += ",";
                    const auto &layer = probe.gdn_layers[i];
                    gdn += "L" + std::to_string(layer.global_layer) +
                           gdnDiagnosticStorageSuffix(layer) +
                           "/r=" + std::to_string(layer.recurrence_hash) +
                           "/c=" + std::to_string(layer.conv_hash);
                    if (layer.device_state_hash_available)
                    {
                        gdn += "/dr=" +
                               std::to_string(layer.recurrence_device_hash) +
                               "/dc=" +
                               std::to_string(layer.conv_device_hash);
                    }
                }
                if (probe.gdn_layers.size() > gdn_limit)
                    gdn += ",...";
                if (gdn.empty())
                    gdn = "none";

                return std::string("pos=[") + positions +
                       "] seq=[" + seqs +
                       "] kv={" + summarize_cache(probe.kv_caches) +
                       "} mtp={" + summarize_cache(probe.mtp_kv_caches) +
                       "} gdn={" + gdn + "}";
            };
            auto summarize_snapshot = [](const PrefixStateSnapshot &snapshot)
            {
                auto summarize_blocks = [](const std::vector<PrefixBlockHandle> &blocks)
                {
                    std::ostringstream out;
                    out << "count=" << blocks.size();
                    const size_t limit = std::min<size_t>(blocks.size(), 4);
                    for (size_t i = 0; i < limit; ++i)
                    {
                        const PrefixBlockHandle &block = blocks[i];
                        out << "|i=" << i
                            << "/tokens=" << block.key.token_count
                            << "/start=" << block.key.token_start
                            << "/idx=" << block.key.block_index
                            << "/kv=" << block.layout.faKVBytes()
                            << "/hybrid=" << block.layout.hybrid_state_bytes
                            << "/hybrid_host=" << block.layout.hybrid_host_state_bytes
                            << "/hybrid_device=" << block.layout.hybrid_device_state_bytes
                            << "/term_h=" << block.layout.terminal_hidden_bytes
                            << "/has_hybrid=" << (block.has_hybrid_state ? "1" : "0")
                            << "/has_term_h=" << (block.has_terminal_hidden ? "1" : "0")
                            << "/dev_hybrid="
                            << ((block.device_hybrid_storage ||
                                 block.device_hybrid_allocation)
                                    ? "1"
                                    : "0");
                    }
                    if (blocks.size() > limit)
                        out << "|...";
                    return out.str();
                };

                std::ostringstream out;
                out << "valid=" << (snapshot.valid ? "1" : "0")
                    << " logical=" << (snapshot.logical_checkpoint ? "1" : "0")
                    << " provenance=" << toString(snapshot.provenance)
                    << " cached=" << snapshot.cached_tokens
                    << " ready=" << (snapshot.ready_event_valid ? "1" : "0")
                    << " participants=" << snapshot.participant_snapshots.size()
                    << " blocks={" << summarize_blocks(snapshot.blocks) << "}"
                    << " mtp_blocks={" << summarize_blocks(snapshot.mtp_blocks) << "}";
                if (!snapshot.mtp_cached_tokens.empty())
                {
                    out << " mtp_cached=[";
                    for (size_t i = 0; i < snapshot.mtp_cached_tokens.size(); ++i)
                    {
                        if (i > 0)
                            out << ",";
                        out << snapshot.mtp_cached_tokens[i];
                    }
                    out << "]";
                }
                const size_t participant_limit =
                    std::min<size_t>(snapshot.participant_snapshots.size(), 4);
                for (size_t i = 0; i < participant_limit; ++i)
                {
                    const PrefixStateSnapshot &child =
                        snapshot.participant_snapshots[i];
                    out << " child" << i
                        << "{prov=" << toString(child.provenance)
                        << " logical=" << (child.logical_checkpoint ? "1" : "0")
                        << " cached=" << child.cached_tokens
                        << " blocks=" << child.blocks.size()
                        << " mtp_blocks=" << child.mtp_blocks.size()
                        << " ready=" << (child.ready_event_valid ? "1" : "0")
                        << "}";
                }
                if (snapshot.participant_snapshots.size() > participant_limit)
                    out << " child...";
                return out.str();
            };
            const PrefixRuntimeStateSnapshot committed_probe_before =
                runner_->prefixStateProbe();
            PrefixRuntimeStateSnapshot serial_probe_after_state_prefix;
            bool have_serial_probe_after_state_prefix = false;
            auto first_gdn_mismatch = [](
                                          const PrefixRuntimeStateSnapshot &lhs,
                                          const PrefixRuntimeStateSnapshot &rhs,
                                          const char *lhs_name,
                                          const char *rhs_name) -> std::string
            {
                const size_t count =
                    std::min(lhs.gdn_layers.size(), rhs.gdn_layers.size());
                if (lhs.gdn_layers.size() != rhs.gdn_layers.size())
                {
                    return std::string(lhs_name) + "_gdn_layers=" +
                           std::to_string(lhs.gdn_layers.size()) + " " +
                           rhs_name + "_gdn_layers=" +
                           std::to_string(rhs.gdn_layers.size());
                }
                for (size_t i = 0; i < count; ++i)
                {
                    const PrefixGDNLayerProbe &a = lhs.gdn_layers[i];
                    const PrefixGDNLayerProbe &b = rhs.gdn_layers[i];
                    const bool both_have_device_state =
                        a.device_state_hash_available &&
                        b.device_state_hash_available;
                    const bool mismatch =
                        a.global_layer != b.global_layer ||
                        a.recurrence_layout != b.recurrence_layout ||
                        a.recurrence_storage != b.recurrence_storage ||
                        a.recurrence_epoch_phase_words != b.recurrence_epoch_phase_words ||
                        (both_have_device_state
                             ? (a.recurrence_device_bytes != b.recurrence_device_bytes ||
                                a.conv_device_bytes != b.conv_device_bytes ||
                                a.recurrence_device_hash != b.recurrence_device_hash ||
                                a.conv_device_hash != b.conv_device_hash)
                             : (a.recurrence_hash != b.recurrence_hash ||
                                a.conv_hash != b.conv_hash ||
                                a.device_state_hash_available != b.device_state_hash_available ||
                                a.recurrence_all_zero != b.recurrence_all_zero ||
                                a.conv_all_zero != b.conv_all_zero));
                    if (mismatch)
                    {
                        std::ostringstream oss;
                        oss << "layer=" << a.global_layer << " " << lhs_name
                            << "_layout=" << gdnRecurrenceStateLayoutName(a.recurrence_layout)
                            << gdnDiagnosticStorageSuffix(a)
                            << " " << rhs_name
                            << "_layout=" << gdnRecurrenceStateLayoutName(b.recurrence_layout)
                            << gdnDiagnosticStorageSuffix(b)
                            << " " << lhs_name << "_rec=" << a.recurrence_hash << " " << rhs_name
                            << "_rec=" << b.recurrence_hash << " " << lhs_name
                            << "_conv=" << a.conv_hash << " " << rhs_name << "_conv=" << b.conv_hash
                            << " " << lhs_name << "_dev_rec=" << a.recurrence_device_hash << " "
                            << rhs_name << "_dev_rec=" << b.recurrence_device_hash << " "
                            << lhs_name << "_dev_conv=" << a.conv_device_hash << " " << rhs_name
                            << "_dev_conv=" << b.conv_device_hash << " " << lhs_name
                            << "_rec_zero=" << (a.recurrence_all_zero ? "true" : "false") << " "
                            << rhs_name
                            << "_rec_zero=" << (b.recurrence_all_zero ? "true" : "false") << " "
                            << lhs_name << "_conv_zero=" << (a.conv_all_zero ? "true" : "false")
                            << " " << rhs_name
                            << "_conv_zero=" << (b.conv_all_zero ? "true" : "false");
                        return oss.str();
                    }
                }
                return "none";
            };

            int continuation_check_depth = 1;
            if (const char *depth_env =
                    DebugEnv::envValue("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_DEPTH"))
            {
                char *end = nullptr;
                const long parsed = std::strtol(depth_env, &end, 10);
                if (end != depth_env && parsed > 0)
                {
                    continuation_check_depth =
                        static_cast<int>(std::min<long>(parsed, 16));
                }
            }

            std::string continuation_failure_detail;
            PrefixRuntimeStateSnapshot first_committed_restore_probe;
            PrefixRuntimeStateSnapshot second_committed_restore_probe;
            auto continuation_failure_suffix = [&]() -> std::string
            {
                if (continuation_failure_detail.empty())
                    return {};
                return std::string(": ") + continuation_failure_detail +
                       " current_probe={" + summarize_probe(runner_->prefixStateProbe()) + "}" +
                       " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                       " first_committed_restore_probe={" + summarize_probe(first_committed_restore_probe) + "}" +
                       " second_committed_restore_probe={" + summarize_probe(second_committed_restore_probe) + "}" +
                       " committed_restore_gdn_delta={" +
                       first_gdn_mismatch(first_committed_restore_probe,
                                          second_committed_restore_probe,
                                          "first",
                                          "second") +
                       "}" +
                       " committed_snapshot={" + summarize_snapshot(committed_checkpoint) + "}";
            };

            auto commit_shifted_replay_row = [&](
                                                  int32_t token,
                                                  int token_index,
                                                  const char *context) -> bool
            {
                struct ReplayCacheCounts
                {
                    int main = -1;
                    int shifted = -1;
                };
                auto current_replay_cache_counts = [&]() -> ReplayCacheCounts
                {
                    ReplayCacheCounts counts;
                    const PrefixRuntimeStateSnapshot probe =
                        runner_->prefixStateProbe();
                    for (const PrefixKVCacheProbe &cache : probe.kv_caches)
                    {
                        for (const PrefixKVLayerProbe &layer : cache.layers)
                        {
                            if (layer.seq_idx == 0)
                            {
                                counts.main =
                                    std::max(
                                        counts.main,
                                        layer.cached_tokens);
                            }
                        }
                    }
                    for (const PrefixKVCacheProbe &cache : probe.mtp_kv_caches)
                    {
                        for (const PrefixKVLayerProbe &layer : cache.layers)
                        {
                            if (layer.seq_idx == 0)
                            {
                                counts.shifted =
                                    std::max(
                                        counts.shifted,
                                        layer.cached_tokens);
                            }
                        }
                    }
                    return counts;
                };

                /*
                 * A verifier base can already contain the speculative shifted
                 * row for its first replay input.  Re-appending that row advances
                 * shifted KV one token ahead of grouped publication and changes
                 * the next MTP condition.  Decide from both canonical cache
                 * counts: reuse when shifted equals main, append only when it
                 * trails main by exactly one, and reject every other lifecycle
                 * shape as an oracle failure.
                 */
                const ReplayCacheCounts cache_counts =
                    current_replay_cache_counts();
                const MTPShiftedReplayRowPlan shifted_plan =
                    planMTPShiftedReplayRow(
                        cache_counts.main,
                        cache_counts.shifted,
                        architecture()=="qwen4exp" ? MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero : MTPShiftedCacheOrigin::SuccessorPairsOnly);
                if (!shifted_plan)
                {
                    continuation_failure_detail =
                        std::string("shifted MTP replay lifecycle invalid in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " main_before=" +
                        std::to_string(cache_counts.main) +
                        " shifted_before=" +
                        std::to_string(cache_counts.shifted) +
                        " reason=" + shifted_plan.reason;
                    return false;
                }
                if (shifted_plan.action ==
                    MTPShiftedReplayRowAction::ReuseResidentRow)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "commit_replay_check_shifted_row_reuses",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"context", context},
                         {"token_index", std::to_string(token_index)}});
                    return true;
                }

                constexpr int shifted_rows_committed_in_this_replay = 0;
                const int replay_position_offset =
                    shifted_plan.append_position_offset;
                bool ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "commit_replay_check_shifted_commit",
                        "decode",
                        {},
                        {{"path", path},
                         {"context", context},
                         {"token_index", std::to_string(token_index)}});
                    /*
                     * The replay oracle must match the shared stepwise verifier
                     * boundary exactly: every token forwarded from the verifier
                     * base first publishes the shifted MTP row derived from the
                     * current terminal hidden.  Otherwise the main KV/GDN state
                     * can look serial-equivalent while the sidecar cache remains
                     * at the base position, which makes the next MTP step
                     * compare a live committed state with a stale replay.
                     */
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        constexpr int kDiagnosticTargetSampleSlot = 0;
                        ok = runner_
                                 ->stageStochasticTargetTokenForDeviceSampling(
                                     token,
                                     kDiagnosticTargetSampleSlot) &&
                             runner_
                                 ->commitMTPShiftedRowFromDeviceTargetSample(
                                     kDiagnosticTargetSampleSlot,
                                     shifted_rows_committed_in_this_replay,
                                     /*allow_speculative_discard=*/true);
                    }
                    else
                    {
                        ok = runner_
                                 ->commitMTPShiftedRowFromCurrentTerminalHidden(
                                     token,
                                     shifted_rows_committed_in_this_replay,
                                     /*allow_speculative_discard=*/true,
                                     replay_position_offset);
                    }
                }
                if (!ok)
                {
                    continuation_failure_detail =
                        std::string("shifted MTP replay commit failed in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " token=" + std::to_string(token) +
                        " base_sidecar_position=" +
                        std::to_string(base_sidecar_position) +
                        " main_before=" +
                        std::to_string(cache_counts.main) +
                        " shifted_before=" +
                        std::to_string(cache_counts.shifted) +
                        " replay_position_offset=" +
                        std::to_string(replay_position_offset);
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_shifted_commits",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"context", context}});
                return true;
            };

            auto forward_replay_token = [&](
                                            int32_t token,
                                            int token_index,
                                            const char *context) -> bool
            {
                if (!commit_shifted_replay_row(token, token_index, context))
                    return false;
                int forward_token = static_cast<int>(token);
                if (!runner_->forward(&forward_token, 1))
                {
                    continuation_failure_detail =
                        std::string("replay forward failed in ") +
                        context +
                        " token_index=" + std::to_string(token_index) +
                        " token=" + std::to_string(token);
                    return false;
                }
                return true;
            };

            auto sample_continuation = [&](int32_t first_input,
                                           int first_token_index)
                -> std::optional<std::vector<int32_t>>
            {
                continuation_failure_detail.clear();
                std::vector<int32_t> tokens;
                tokens.reserve(static_cast<size_t>(continuation_check_depth));
                int32_t input = first_input;
                for (int i = 0; i < continuation_check_depth; ++i)
                {
                    const int token_index = first_token_index + i;
                    if (!forward_replay_token(
                            input,
                            token_index,
                            "continuation"))
                    {
                        continuation_failure_detail +=
                            " depth=" + std::to_string(i);
                        return std::nullopt;
                    }
                    const int32_t sampled = runner_->sampleGreedyOnDevice();
                    if (sampled < 0)
                    {
                        continuation_failure_detail =
                            std::string("main continuation greedy sampling failed at depth=") +
                            std::to_string(i) +
                            " input=" + std::to_string(input);
                        return std::nullopt;
                    }
                    tokens.push_back(sampled);
                    input = sampled;
                }
                return tokens;
            };

            /*
             * vLLM-style publication can make tokens host-visible before every
             * one of those tokens has been consumed by the live verifier state.
             * A first-row rejection is the common case: the correction token is
             * emitted as output, but it remains the next condition token.  The
             * replay oracle must therefore feed the unadvanced emitted suffix
             * first, prove it produces the expected ready token, and only then
             * continue past that ready token.
             */
            auto sample_continuation_after_pending_inputs = [&](
                                                                const std::vector<int32_t> &pending_inputs,
                                                                int32_t ready_token,
                                                                int pending_token_index_base)
                -> std::optional<std::vector<int32_t>>
            {
                if (ready_token < 0)
                    return std::nullopt;

                const PrefixRuntimeStateSnapshot pending_start_probe =
                    runner_->prefixStateProbe();
                for (size_t i = 0; i < pending_inputs.size(); ++i)
                {
                    const int32_t pending = pending_inputs[i];
                    const int token_index =
                        pending_token_index_base +
                        static_cast<int>(i);
                    if (!forward_replay_token(
                            pending,
                            token_index,
                            "pending_condition"))
                    {
                        continuation_failure_detail +=
                            " pending_index=" + std::to_string(i);
                        return std::nullopt;
                    }
                    const int32_t sampled = runner_->sampleGreedyOnDevice();
                    if (sampled < 0)
                    {
                        continuation_failure_detail =
                            std::string("pending-condition greedy sampling failed at index=") +
                            std::to_string(i) +
                            " input=" + std::to_string(pending);
                        return std::nullopt;
                    }

                    const int32_t expected =
                        (i + 1 < pending_inputs.size())
                            ? pending_inputs[i + 1]
                            : ready_token;
                    if (sampled != expected)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "commit_replay_check_pending_condition_mismatches",
                            1.0,
                            "decode",
                            {},
                            {{"path", path},
                             {"pending_index", std::to_string(i)},
                             {"pending_token", std::to_string(pending)},
                             {"sampled", std::to_string(sampled)},
                             {"expected", std::to_string(expected)}});
                        continuation_failure_detail =
                            std::string("pending-condition token mismatch at index=") +
                            std::to_string(i) +
                            " input=" + std::to_string(pending) +
                            " sampled=" + std::to_string(sampled) +
                            " expected=" + std::to_string(expected) +
                            " pending_start_probe={" +
                            summarize_probe(pending_start_probe) + "}";
                        return std::nullopt;
                    }
                }

                return sample_continuation(
                    ready_token,
                    pending_token_index_base +
                        static_cast<int>(pending_inputs.size()));
            };

            const bool derived_next_token_from_deferred_condition =
                expected_next_token < 0;
            auto prepare_committed_ready_state = [&]()
                -> std::optional<int32_t>
            {
                if (!derived_next_token_from_deferred_condition)
                    return expected_next_token;

                /*
                 * A forced reject has no ready token yet: publication advances
                 * only the accepted verifier prefix, while the rejected
                 * correction is the next ordinary condition token.  The debug
                 * oracle therefore has to run that one condition forward before
                 * comparing the committed state against a full replay.
                 */
                const int32_t deferred_condition_token = tokens_to_replay.back();
                const int deferred_token_index =
                    std::max(
                        0,
                        static_cast<int>(tokens_to_replay.size()) - 1);
                if (!forward_replay_token(
                        deferred_condition_token,
                        deferred_token_index,
                        "deferred_condition_ready"))
                    return std::nullopt;
                const int32_t sampled = runner_->sampleGreedyOnDevice();
                if (sampled < 0)
                    return std::nullopt;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_derived_next_tokens",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"deferred_condition_token",
                      std::to_string(deferred_condition_token)},
                     {"next_token", std::to_string(sampled)}});
                return sampled;
            };

            std::optional<int32_t> prepared_next_token =
                prepare_committed_ready_state();
            if (!prepared_next_token)
            {
                const std::string summary =
                    derived_next_token_from_deferred_condition
                        ? std::string("MTP commit replay check deferred condition forward failed")
                        : std::string("MTP commit replay check missing expected next token");
                return summary + continuation_failure_suffix();
            }
            expected_next_token = *prepared_next_token;

            const PrefixStateSnapshot &base = *verifier_replay_base_checkpoint;
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base state");
            }
            bool sequential_replay_ok = true;
            int32_t replay_next_token = -1;
            for (size_t i = 0; i < tokens_to_replay.size(); ++i)
            {
                const int32_t replay_token = tokens_to_replay[i];
                if (!forward_replay_token(
                        replay_token,
                        static_cast<int>(i),
                        "full_replay"))
                {
                    sequential_replay_ok = false;
                    break;
                }

                /*
                 * Validate every output transition, not merely the state after
                 * the complete transaction. A grouped verifier can emit a wrong
                 * intermediate correction token whose later continuation happens
                 * to reconverge. Feeding that wrong token into the replay oracle
                 * and checking only the final sample would bless the corruption.
                 * Sampling after each serial row proves byte-for-byte token-stream
                 * equivalence at the earliest observable boundary.
                 */
                replay_next_token = runner_->sampleGreedyOnDevice();
                if (replay_next_token < 0)
                {
                    return std::string(
                        "MTP commit replay check row sampling failed at replay index=") +
                           std::to_string(i);
                }
                const int32_t expected_row_next =
                    i + 1 < tokens_to_replay.size()
                        ? tokens_to_replay[i + 1]
                        : expected_next_token;
                if (replay_next_token != expected_row_next)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "commit_replay_check_serial_output_token_mismatches",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"replay_index", std::to_string(i)},
                         {"replay_token", std::to_string(replay_token)},
                         {"serial_next", std::to_string(replay_next_token)},
                         {"grouped_next",
                          std::to_string(expected_row_next)}});
                    return std::string(
                               "MTP grouped output token mismatch against serial row replay: path=") +
                           path +
                           " replay_index=" + std::to_string(i) +
                           " replay_token=" + std::to_string(replay_token) +
                           " serial_next=" +
                           std::to_string(replay_next_token) +
                           " grouped_next=" +
                           std::to_string(expected_row_next) +
                           " condition_token=" +
                           std::to_string(condition_token) +
                           " committed_tokens=" +
                           join_tokens(tokens_to_replay) +
                           " committed_probe_before={" +
                           summarize_probe(committed_probe_before) + "}" +
                           " replay_probe={" +
                           summarize_probe(runner_->prefixStateProbe()) + "}" +
                           (debug_context.empty()
                                ? std::string{}
                                : " " + debug_context);
                }
                if (!have_serial_probe_after_state_prefix &&
                    static_cast<int>(i + 1) == replay_prefix_count)
                {
                    /*
                     * Forced-reject grouped MTP publishes only the accepted
                     * verifier prefix, then carries the rejected correction as
                     * the next condition token.  Capturing the serial replay
                     * state at that exact prefix boundary lets the failure
                     * message distinguish a bad accepted-row publication from
                     * a bad one-token continuation after an otherwise correct
                     * publication.
                     */
                    serial_probe_after_state_prefix =
                        runner_->prefixStateProbe();
                    have_serial_probe_after_state_prefix = true;
                }
            }
            if (!sequential_replay_ok)
            {
                return std::string("MTP commit replay check sequential replay failed") +
                       continuation_failure_suffix();
            }
            const PrefixRuntimeStateSnapshot serial_probe_after_full_replay =
                runner_->prefixStateProbe();
            if (replay_next_token < 0)
            {
                return std::string("MTP commit replay check full replay sampling failed");
            }
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string("MTP commit replay check could not restore committed state");
            }
            first_committed_restore_probe = runner_->prefixStateProbe();
            if (replay_next_token != expected_next_token)
            {
                return std::string("MTP committed state mismatch against full replay: path=") +
                       path +
                       " condition_token=" + std::to_string(condition_token) +
                       " accepted_tokens=" + join_tokens(tokens_to_replay) +
                       " committed_next=" + std::to_string(expected_next_token) +
                       " replay_next=" + std::to_string(replay_next_token) +
                       " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                       " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                       " committed_restore_probe={" + summarize_probe(first_committed_restore_probe) + "}" +
                       (have_serial_probe_after_state_prefix
                            ? " serial_probe_after_state_prefix={" +
                                  summarize_probe(serial_probe_after_state_prefix) +
                                  "} state_prefix_gdn_first_mismatch={" +
                                  first_gdn_mismatch(
                                      committed_probe_before,
                                      serial_probe_after_state_prefix,
                                      "committed",
                                      "serial_prefix") +
                                  "}"
                            : std::string{}) +
                       " serial_probe_after_full_replay={" + summarize_probe(serial_probe_after_full_replay) + "}" +
                       " gdn_first_mismatch={" +
                       first_gdn_mismatch(
                           first_committed_restore_probe,
                           serial_probe_after_full_replay,
                           "committed",
                           "serial") +
                       "}" +
                       " committed_snapshot={" + summarize_snapshot(committed_checkpoint) + "}" +
                       (debug_context.empty() ? std::string{} : " " + debug_context);
            }

            /*
             * Keep the first replay assertion side-effect-free with respect to
             * future continuation work.  The continuation probe below is useful,
             * but it intentionally mutates live decode state; running it before
             * the base replay can turn a restore-ordering bug into a misleading
             * ready-token mismatch.
             */
            std::optional<std::vector<int32_t>> live_committed_continuation =
                sample_continuation_after_pending_inputs(
                    pending_condition_inputs,
                    expected_next_token,
                    replay_prefix_count);
            if (!live_committed_continuation)
            {
                return std::string("MTP commit replay check live committed continuation forward failed") +
                       continuation_failure_suffix();
            }
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string("MTP commit replay check could not restore committed state after live continuation check");
            }
            second_committed_restore_probe = runner_->prefixStateProbe();

            std::optional<int32_t> committed_ready_token =
                prepare_committed_ready_state();
            if (!committed_ready_token ||
                *committed_ready_token != expected_next_token)
            {
                std::ostringstream mismatch;
                mismatch
                    << "MTP commit replay check committed ready-token derivation mismatch"
                    << ": expected=" << expected_next_token
                    << " actual="
                    << (committed_ready_token
                            ? std::to_string(*committed_ready_token)
                            : std::string("none"))
                    << " current_probe={" << summarize_probe(runner_->prefixStateProbe()) << "}"
                    << " committed_probe_before={" << summarize_probe(committed_probe_before) << "}"
                    << " first_committed_restore_probe={"
                    << summarize_probe(first_committed_restore_probe) << "}"
                    << " second_committed_restore_probe={"
                    << summarize_probe(second_committed_restore_probe) << "}"
                    << " committed_snapshot={" << summarize_snapshot(committed_checkpoint) << "}";
                return mismatch.str();
            }
            if (derived_next_token_from_deferred_condition)
            {
                /*
                 * Deriving the ready token for a forced reject is itself an
                 * ordinary one-token forward from the committed checkpoint.
                 * The continuation oracle below intentionally feeds the same
                 * pending condition token to prove the restored state, so put
                 * the runner back at the committed checkpoint before that
                 * second probe.
                 */
                if (!runner_->restoreLivePrefixState(committed_checkpoint))
                {
                    return std::string("MTP commit replay check could not restore committed state after ready-token derivation");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "commit_replay_check_restores_after_ready_derivation",
                    1.0,
                    "decode",
                    {},
                    {{"path", path}});
            }
            std::optional<std::vector<int32_t>> committed_continuation =
                sample_continuation_after_pending_inputs(
                    pending_condition_inputs,
                    expected_next_token,
                    replay_prefix_count);
            if (!committed_continuation)
            {
                return std::string("MTP commit replay check committed continuation forward failed") +
                       continuation_failure_suffix();
            }
            if (!runner_->restoreLivePrefixState(base))
            {
                return std::string("MTP commit replay check could not restore verifier base for continuation replay");
            }
            bool sequential_continuation_ok = true;
            for (size_t i = 0; i < state_replay_tokens.size(); ++i)
            {
                const int32_t replay_token = state_replay_tokens[i];
                if (!forward_replay_token(
                        replay_token,
                        static_cast<int>(i),
                        "state_replay"))
                {
                    sequential_continuation_ok = false;
                    break;
                }
            }
            PrefixRuntimeStateSnapshot serial_probe_after_replay;
            bool have_serial_probe_after_replay = false;
            if (sequential_continuation_ok)
            {
                serial_probe_after_replay = runner_->prefixStateProbe();
                have_serial_probe_after_replay = true;
                std::optional<std::vector<int32_t>> replay_continuation =
                    sample_continuation_after_pending_inputs(
                        pending_condition_inputs,
                        expected_next_token,
                        replay_prefix_count);
                if (!replay_continuation)
                {
                    return std::string("MTP commit replay check continuation replay forward failed") +
                           continuation_failure_suffix();
                }
                if (!runner_->restoreLivePrefixState(committed_checkpoint))
                {
                    return std::string("MTP commit replay check could not restore committed state after continuation check");
                }
                auto summarize_prefix_replays = [&]() -> std::string
                {
                    std::string summary;
                    for (size_t prefix_len = 0;
                         prefix_len <= tokens_to_replay.size();
                         ++prefix_len)
                    {
                        if (!runner_->restoreLivePrefixState(base))
                        {
                            summary += " len" + std::to_string(prefix_len) + "=restore_failed";
                            continue;
                        }
                        bool prefix_ok = true;
                        for (size_t i = 0; i < prefix_len; ++i)
                        {
                            const int32_t replay_token = tokens_to_replay[i];
                            if (!forward_replay_token(
                                    replay_token,
                                    static_cast<int>(i),
                                    "prefix_replay_summary"))
                            {
                                prefix_ok = false;
                                break;
                            }
                        }
                        if (!prefix_ok)
                        {
                            summary += " len" + std::to_string(prefix_len) + "=forward_failed";
                            continue;
                        }
                        std::optional<std::vector<int32_t>> prefix_continuation =
                            sample_continuation_after_pending_inputs(
                                pending_condition_inputs,
                                expected_next_token,
                                replay_prefix_count);
                        summary += " len" + std::to_string(prefix_len) + "=";
                        summary += prefix_continuation
                                       ? join_tokens(*prefix_continuation)
                                       : std::string("sample_failed");
                    }
                    (void)runner_->restoreLivePrefixState(committed_checkpoint);
                    return summary;
                };
                if (*live_committed_continuation != *replay_continuation)
                {
                    const PrefixRuntimeStateSnapshot mismatch_probe =
                        runner_->prefixStateProbe();
                    return std::string("MTP live committed state continuation mismatch against full replay: path=") +
                           path +
                           " condition_token=" + std::to_string(condition_token) +
                           " accepted_tokens=" + join_tokens(tokens_to_replay) +
                           " next_token=" + std::to_string(expected_next_token) +
                           " live_committed_continuation=" + join_tokens(*live_committed_continuation) +
                           " committed_continuation=" + join_tokens(*committed_continuation) +
                           " replay_continuation=" + join_tokens(*replay_continuation) +
                           " prefix_replay_continuations=" + summarize_prefix_replays() +
                           " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                           (have_serial_probe_after_replay
                                ? " serial_probe_after_replay={" +
                                      summarize_probe(serial_probe_after_replay) +
                                      "} gdn_first_mismatch={" +
                                      first_gdn_mismatch(
                                          committed_probe_before,
                                          serial_probe_after_replay,
                                          "committed",
                                          "serial") +
                                      "}"
                                : std::string{}) +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                           (debug_context.empty() ? std::string{} : " " + debug_context);
                }
                if (*committed_continuation != *replay_continuation)
                {
                    const PrefixRuntimeStateSnapshot mismatch_probe =
                        runner_->prefixStateProbe();
                    return std::string("MTP committed state continuation mismatch against full replay: path=") +
                           path +
                           " condition_token=" + std::to_string(condition_token) +
                           " accepted_tokens=" + join_tokens(tokens_to_replay) +
                           " next_token=" + std::to_string(expected_next_token) +
                           " live_committed_continuation=" + join_tokens(*live_committed_continuation) +
                           " committed_continuation=" + join_tokens(*committed_continuation) +
                           " replay_continuation=" + join_tokens(*replay_continuation) +
                           " prefix_replay_continuations=" + summarize_prefix_replays() +
                           " committed_probe_before={" + summarize_probe(committed_probe_before) + "}" +
                           (have_serial_probe_after_replay
                                ? " serial_probe_after_replay={" +
                                      summarize_probe(serial_probe_after_replay) +
                                      "} gdn_first_mismatch={" +
                                      first_gdn_mismatch(
                                          committed_probe_before,
                                          serial_probe_after_replay,
                                          "committed",
                                          "serial") +
                                      "}"
                                : std::string{}) +
                           " mismatch_probe={" + summarize_probe(mismatch_probe) + "}" +
                           " continuation_depth=" + std::to_string(continuation_check_depth) +
                           " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")) +
                           (debug_context.empty() ? std::string{} : " " + debug_context);
                }
            }
            if (!sequential_continuation_ok)
            {
                return std::string("MTP commit replay check continuation replay forward failed");
            }

            /*
             * Replay diagnostics intentionally replace the CPU live timeline
             * several times. Restore the exact committed checkpoint once more
             * before the host oracle returns to the real decode transaction.
             * GPU grouped publication never enters this host replay diagnostic.
             */
            if (!runner_->restoreLivePrefixState(committed_checkpoint))
            {
                return std::string(
                    "MTP commit replay check could not perform final committed-state restore");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "commit_replay_check_matches",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"accepted_tokens", join_tokens(tokens_to_replay)},
                 {"next_token", std::to_string(expected_next_token)},
                 {"continuation_depth", std::to_string(continuation_check_depth)},
                 {"state_advanced_tokens", std::to_string(state_advanced_count)},
                 {"pending_condition_inputs", join_tokens(pending_condition_inputs)},
                 {"derived_next_token",
                  derived_next_token_from_deferred_condition ? "true" : "false"},
                 {"used_ready_logits", use_ready_logits ? "true" : "false"}});
            return std::nullopt;
    }

    GenerationResult OrchestrationRunner::decodeOrdinaryDeviceResidentGeneration()
    {
        using Leading = sampling_math::DeviceGenerationLeadingRowDisposition;
        GenerationResult result;
        const auto fail = [&](const std::string &reason) {
            result.error = "Ordinary device generation: " + reason;
            return result;
        };
        if (!runner_ || !runner_->primaryDeviceId().is_gpu() ||
            !device_generation_admission_.awaitsAdmission() ||
            !prefill_logits_ready_ || !decode_transaction_planning_position_)
            return fail("requires an admission-ready prefill or resident continuation");

        // A prior budget's last response has not entered KV. The existing
        // mailbox remains the token authority; returning it to the caller did
        // not make the host copy a valid input for another forward.
        Leading leading = Leading::PendingResponse;
        if (ready_mtp_condition_)
        {
            if (!ready_mtp_condition_->valid() ||
                !ready_mtp_condition_->isDeviceResidentOnly() ||
                !ready_mtp_condition_->wasAlreadyEmitted() ||
                !samplingParamsEqual(ready_mtp_condition_->sampling_params, active_sampling_params_))
                return fail("continuation is not the previously emitted resident condition under the same sampling law");
            leading = Leading::AlreadyEmitted;
        }
        const int base = *decode_transaction_planning_position_;
        const int capacity = plan_.runtime.max_seq_len > 0
            ? plan_.runtime.max_seq_len : config_.max_seq_len;
        const int budget = decode_step_token_budget_ > 0
            ? decode_step_token_budget_ : capacity - base;
        if (base < 0 || budget <= 0 || base > capacity || budget > capacity - base)
            return fail("response budget exceeds the admitted context capacity");

        if (!generation_request_seeds_)
            generation_request_seeds_.emplace(std::vector<uint64_t>{
                resolveRequestBatchedStochasticSeed(active_sampling_params_, 0)});
        const DeviceGenerationAdmissionRequest admission{
            .request_count = 1,
            .max_new_tokens = budget,
            .kind = device_generation_admission_.admissionKind(),
            .depth_policy = sampling_math::DeviceGenerationPolicy::ordinary(),
            .initial_leading_row_disposition = leading,
            .sampling_seeds = generation_request_seeds_,
            .ordinary_sampling = active_sampling_params_,
            .initial_target_counts = DeviceGenerationInitialTargetCounts(std::vector<int>{base}),
        };
        if (!admission.valid())
            return fail("sampling law is not implemented by the captured ordinary sampler (bounded top-k and presence/frequency penalties are supported; DRY is not)");
        if (!runner_->beginDeviceResidentGeneration(admission) ||
            !device_generation_admission_.admitController(budget))
            return fail("could not admit the complete response budget");
        const auto sampling = active_sampling_params_.is_greedy()
            ? DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic;
        if (!runner_->materializeDeviceResidentGeneration(1, 0,
                DeviceGenerationLoopTopology::FixedDepth, sampling))
            return fail("could not compose the complete captured model and sampler; see the participant diagnostic");
        if (!runner_->launchDeviceResidentGeneration())
            return fail("could not launch the captured generation policy");
        DeviceGenerationTerminalResult terminal;
        if (!runner_->finishDeviceResidentGeneration(&terminal) ||
            !terminal.valid() || terminal.requests.size() != 1 ||
            terminal.device != runner_->primaryDeviceId())
            return fail("did not return one valid terminal ledger from its admitted device");
        const auto &row = terminal.requests.front();
        const auto count = static_cast<int64_t>(row.tokens.size());
        const int initial_committed = sampling_math::device_generation_leading_committed_output_count(leading);
        // The device runner authenticates its ABI. This public boundary also
        // checks response/state semantics before exposing a continuation or
        // changing scheduler metadata. Ordinary generation has no verifier.
        if (count <= 0 || count > budget || row.remaining_token_count < 0 ||
            count + row.remaining_token_count != budget ||
            (!row.model_stopped && row.remaining_token_count != 0) ||
            row.next_leading_row_disposition != Leading::AlreadyEmitted ||
            row.transaction_count != count ||
            row.published_state_commit_count != count + initial_committed - 1 ||
            row.attempted_draft_token_count != 0 || row.verifier_token_count != 0 ||
            row.accepted_speculative_token_count != 0 || row.rejected_transaction_count != 0)
            return fail("terminal response or model-row accounting differs from the immutable admission");

        std::optional<ReadyMTPCondition> continuation;
        if (!row.model_stopped)
        {
            auto state = runner_->deviceResidentLogicalSequenceState();
            if (!state.coversRequest(0) || state.device != terminal.device)
                return fail("terminal response has no live resident continuation mailbox");
            continuation = ReadyMTPCondition::deviceResident(
                active_sampling_params_, std::move(state), Leading::AlreadyEmitted);
            if (!continuation->valid())
                return fail("terminal resident continuation is incomplete");
        }
        if (!publishDecodeTransactionPlanningPositionAfterMTPCommit(
                base, row.published_state_commit_count, "ordinary_device_generation_terminal") ||
            !device_generation_admission_.retireController(row.model_stopped))
            return fail("could not retire the terminal scheduler boundary");

        ready_mtp_condition_ = std::move(continuation);
        prefill_logits_ready_ = ready_mtp_condition_.has_value();
        result.tokens.assign(row.tokens.begin(), row.tokens.end());
        result.is_complete = row.model_stopped;
        last_token_ = result.tokens.back();
        // Host history is terminal response bookkeeping only. The captured
        // sampler updated its own persistent history for every sampled token.
        for (const auto token : result.tokens) sampler_.record_token(token);
        PerfStatsCollector::addCounter("generation", "ordinary_resident_requests", 1.0, "decode");
        PerfStatsCollector::addCounter("generation", "ordinary_resident_output_tokens",
            static_cast<double>(count), "decode");
        return result;
    }

    GenerationResult OrchestrationRunner::decodeStep()
    {
        GenerationResult result;

        if (debugEnv().runtime_debug.mtp_depth_transaction_trace &&
            (decode_step_token_budget_ <= 0 ||
             decode_step_token_budget_ > sampling_math::kMTPDepthTransactionTraceCapacity))
        {
            result.error = "MTP depth-transaction trace requires an explicit remaining token budget1..256";
            return result;
        }

        if (!initialization_lifecycle_.readyForInference())
        {
            result.error = "Runner not initialized";
            return result;
        }
        if (batched_decode_active_)
        {
            result.error =
                "decodeStep() cannot consume request-batched prefill state; "
                "use decodeStepBatch()";
            return result;
        }
        const bool mpi_coordinated_world =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            mpi_coordinated_world &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPIRootCommandFailureAbort decode_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "DECODE_STEP",
            &result.error);
        ensureFiniteDevicePlacementArmed();
        ScopedMoEOverlayRootCommand overlay_decode_command(
            moe_overlay_inference_transaction_coordinator_);

        // Keep the diagnostic catch inside both command guards. Catching after
        // their destruction is too late: a live MPI world has already aborted.
        try
        {
        const std::uint64_t retired_prefill_tokens =
            moe_overlay_pending_prefill_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        const std::uint64_t retired_decode_tokens =
            moe_overlay_pending_decode_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        if (retired_prefill_tokens >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            retired_decode_tokens >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()))
        {
            result.error =
                "ExpertOverlay retired inference progress exceeds the typed MPI command payload";
            return result;
        }

        // Broadcast to worker ranks so they run decode in lockstep. The one
        // existing payload carries both root-owned decode policy and the exact
        // continuation-authoritative prefill progress retired since the prior
        // command. No follower-local capture schedule becomes admission state.
        if (mpi_root_command)
        {
            broadcastCommand(MPICommand::DECODE_STEP);
            std::array<std::int32_t, 3> command_payload{
                static_cast<std::int32_t>(decode_step_token_budget_),
                static_cast<std::int32_t>(retired_prefill_tokens),
                static_cast<std::int32_t>(retired_decode_tokens),
            };
            mpi_ctx_->broadcast_int32(
                command_payload.data(),
                command_payload.size(),
                mpi_coordinated_root_rank_);
            // From this point a worker may enter model collectives before root
            // reaches its sampler.  Every later error must abort the command
            // world instead of returning to a caller that cannot release it.
            decode_command_failure_abort.markPublished();
        }
        if (retired_prefill_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::
                          InferencePhase::Prefill,
                      retired_prefill_tokens)))
        {
            result.error =
                "Continuation authority could not publish retired-prefill progress to the device ExpertOverlay controller";
            return result;
        }
        if (retired_decode_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::
                          InferencePhase::Decode,
                      retired_decode_tokens)))
        {
            result.error =
                "Continuation authority could not publish retired-decode progress to the device ExpertOverlay controller";
            return result;
        }

        if (overlay_decode_command.active())
        {
            const auto command_context = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            if (!command_context ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                result.error =
                    "ExpertOverlay decode authority has incomplete root, request-generation, or residency ownership";
                return result;
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                result.error =
                    "ExpertOverlay decode authority has no valid placement epoch or command identity";
                return result;
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                // Initial admitted lower bound only: mapped device packet kernels
                // authenticate their actual current placement ticket and bank.
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_decode_command.begin(
                    command_context->rank(), command_identity, &command_error))
            {
                result.error =
                    "ExpertOverlay remote decode command admission failed: " +
                    command_error;
                return result;
            }
        }

        auto *const original_global = dynamic_cast<GlobalOrchestrator *>(runner_.get());
        if (original_global && original_global->originalRankGenerationFollower())
        {
            // This original stage process has no sampler/outcome authority.
            // The actual tail publishes immutable scheduling identities; local
            // model/KV/publication work stays in its original DGO graphs.
            GlobalOrchestrator::OriginalRankGenerationTerminal terminal;
            if (!original_global->serviceOriginalRankGenerationCommand(&terminal, &result.error))
            {
                original_global->abortOriginalRankGenerationCommand();
                if (result.error.empty()) result.error = "Original rank generation follower failed";
                return result;
            }
            if (!decode_transaction_planning_position_ ||
                *decode_transaction_planning_position_ != terminal.admission_base ||
                !sampling_math::valid_device_generation_leading_row_disposition(terminal.next_leading_row_disposition) ||
                !publishDecodeTransactionPlanningPositionAfterMTPCommit(terminal.admission_base,
                    terminal.published_state_commit_count, "original_rank_generation_terminal"))
            {
                result.error = "Original rank generation terminal differs from its admitted scheduler boundary";
                return result;
            }
            // Keep only completed scheduler/accounting observations. No source
            // outcome or remote logical-state handle becomes execution input.
            ready_mtp_condition_.reset();
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
            prefill_logits_ready_ = !terminal.model_stopped;
            result.is_complete = terminal.model_stopped;
            decode_command_failure_abort.markCompleted();
            return result;
        }
        /** @brief Cancel original directed transport before the outer MPI failure guard can abort the world. */
        struct OriginalRankCommandGuard
        {
            GlobalOrchestrator *owner = nullptr; ///< Real source only; no remote runner alias.
            bool completed = false; ///< True only after actual terminal publication.
            /** @brief Keep failed original admissions absorbing while their native owners remain alive. */
            ~OriginalRankCommandGuard() { if (owner && !completed) owner->abortOriginalRankGenerationCommand(); }
            /** @brief Complete using actual terminal ledger and its original leading-row disposition. */
            bool finish(bool stopped, sampling_math::DeviceGenerationLeadingRowDisposition leading)
            {
                if (!owner) return true;
                completed = owner->completeOriginalRankGenerationCommand(stopped, leading);
                return completed;
            }
        } original_command{original_global && original_global->originalRankGenerationSource() ? original_global : nullptr};
        if (original_command.owner)
        {
            if (!decode_transaction_planning_position_ ||
                moe_overlay_inference_command_sequence_ == std::numeric_limits<uint64_t>::max() ||
                !original_command.owner->beginOriginalRankGenerationCommand(
                    original_command.owner->originalRankCommandIdentity(++moe_overlay_inference_command_sequence_),
                    *decode_transaction_planning_position_, active_sampling_params_.is_greedy() ?
                        DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic))
            {
                result.error = "Original rank generation source could not admit its exact request/command/placement identity";
                return result;
            }
        }
        const auto terminal_leading = [&] {
            return ready_mtp_condition_ && ready_mtp_condition_->wasAlreadyEmitted() ?
                sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted :
                sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;
        };

        const bool mpi_worker_rank =
            mpi_coordinated_world &&
            mpi_ctx_->rank() != mpi_coordinated_root_rank_;
        auto trace_position = [this](const char *context) -> int
        {
            return currentDecodeTransactionPositionForPlanning(context, nullptr)
                .value_or(-1);
        };
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/decodeStep] begin rank="
                     << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                     << " coordinated=" << (mpi_coordinated_world ? "true" : "false")
                     << " worker=" << (mpi_worker_rank ? "true" : "false")
                     << " prefill_logits_ready=" << (prefill_logits_ready_ ? "true" : "false")
                     << " position=" << trace_position("trace_decode_step_begin"));
        }

        std::optional<HostSparseQueryResponseScope> query_response_scope;
        try
        {
            query_response_scope.emplace(*runner_, host_sparse_query_admission_,
                hostSparseQueryAdmission(decode_step_token_budget_));
        }
        catch (const std::exception &error)
        {
            last_error_ = result.error = error.what();
            return result;
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        // A heterogeneous hosted pipeline has one tail-owned sampler across
        // the original MPI ranks. Its GPU follower's local primary device is
        // not generation authority: all ranks must enter the declared hosted
        // forward/sampling protocol together. Complete GPU pipelines instead
        // retain their original captured generation service on every rank.
        const bool hosted_pipeline_generation = original_global &&
            !original_global->requiresOriginalRankGPUTransport();
        if (!mtp.enabled && runner_->primaryDeviceId().is_gpu() && !hosted_pipeline_generation)
        {
            result = decodeOrdinaryDeviceResidentGeneration();
            if (result.success())
            {
                std::string error;
                if (!overlay_decode_command.complete(&error))
                    result.error = "Ordinary generation command completion failed: " + error;
                else if (!original_command.finish(result.is_complete, terminal_leading()))
                    result.error = "Original rank ordinary generation command could not retire its actual terminal";
                else
                    decode_command_failure_abort.markCompleted();
            }
            return result;
        }
        bool adaptive_depth_zero_step = false;
        if (mtp.enabled)
        {
            const std::string mtp_hard_failure = mtpDecodeHardFailureReason();
            if (!mtp_hard_failure.empty())
            {
                result.error = mtp_hard_failure;
                return result;
            }

            const std::string mtp_bypass_reason = mtpDecodeBypassReason();
            if (mtp_bypass_reason.empty())
            {
                if (!ensureMTPDepthController(mtp))
                {
                    result.error = last_error_.empty()
                                       ? "Invalid MTP depth policy"
                                       : last_error_;
                    return result;
                }
                const bool retained_continuation = isHostedDynamicMTPContinuation(mtp);
                const int selected_draft_depth = retained_continuation
                    ? 0 : currentMTPDraftDepth(mtp);
                if (retained_continuation || selected_draft_depth > 0)
                {
                    const std::uint64_t verifier_runs_before =
                        mtp_stats_.verifier_runs;
                    std::optional<MoEOverlayInferenceInterferenceScope> interference_scope;
                    // A continuation may select several widths on device, so
                    // this fixed-shape calibration scope cannot describe it.
                    // Native branch PerfStats retain the actual geometry; the
                    // dormant host selector must not label the whole response.
                    if (!retained_continuation)
                        interference_scope.emplace(
                            moe_expert_overlay_interference_probe_.get(),
                            makeMoEOverlayInferenceWorkloadIdentity(
                                ExpertHistogramSource::GroupedVerifier,
                                selected_draft_depth + 1,
                                effectiveMTPMaxDraftDepth(mtp) + 1,
                                1,
                                selected_draft_depth));
                    GenerationResult mtp_result = decodeStepMTP();
                    if (interference_scope && mtp_stats_.verifier_runs == verifier_runs_before)
                        interference_scope->discard();
                    if (!mtp_result.success())
                    {
                        /*
                         * The outer command guards abort the live distributed
                         * transaction while this stack unwinds. Emit the root
                         * cause before that terminal publication so MPI_Abort
                         * cannot erase the only actionable diagnostic.
                         */
                        LOG_ERROR(
                            "[OrchestrationRunner] MTP decode failed after "
                            "ExpertOverlay command admission: "
                            << (mtp_result.error.empty()
                                    ? "no failure detail was recorded"
                                    : mtp_result.error));
                    }
                    if (mtp_result.success())
                    {
                        std::string command_error;
                        if (!overlay_decode_command.complete(&command_error))
                        {
                            mtp_result.error =
                                "ExpertOverlay MTP command completion failed: " +
                                command_error;
                            LOG_ERROR("[OrchestrationRunner] " << mtp_result.error);
                            result = std::move(mtp_result);
                            return result;
                        }
                        if (!original_command.finish(mtp_result.is_complete, terminal_leading()))
                            mtp_result.error = "Original rank MTP generation command could not retire its actual terminal";
                        else
                            decode_command_failure_abort.markCompleted();
                    }
                    result = std::move(mtp_result);
                    return result;
                }
                if (ready_mtp_condition_.has_value() &&
                    ready_mtp_condition_->isDeviceResidentOnly())
                {
                    result.error =
                        "Device-resident MTP continuation cannot cross into a depth-zero serial decode transaction";
                    return result;
                }
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
                recordMTPDepthZeroBypass();
                adaptive_depth_zero_step = true;
            }
            else
            {
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
                recordMTPBypass(mtp_bypass_reason);
            }
        }

        std::optional<int32_t> ready_token_for_decode;
        const bool can_defer_decode_sampling_sync =
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            (active_sampling_params_.is_greedy() ||
             (active_sampling_params_.top_k > 0 &&
              active_sampling_params_.top_k <= 256));
        bool decode_sampling_sync_deferred = false;
        if (prefill_logits_ready_)
        {
            // First decode step after prefill: sample from the already-computed
            // prefill logits instead of re-feeding the last prompt token.
            // This avoids processing the last token twice (which corrupts GDN
            // recurrence state and creates duplicate KV cache entries).
            if (ready_mtp_condition_.has_value())
            {
                if (!ready_mtp_condition_->valid())
                {
                    result.error =
                        "Ready MTP condition has an invalid authority payload";
                    return result;
                }
                if (!samplingParamsEqual(
                        ready_mtp_condition_->sampling_params,
                        active_sampling_params_))
                {
                    result.error =
                        "Ready MTP condition was sampled with different sampling parameters";
                    return result;
                }
                if (ready_mtp_condition_->isDeviceResidentOnly())
                {
                    result.error =
                        "Device-resident MTP continuation reached the serial decode sampler";
                    return result;
                }
                ready_token_for_decode =
                    ready_mtp_condition_->host_token;
                ready_mtp_condition_.reset();
            }
            prefill_logits_ready_ = false;
            LOG_TRACE("[decodeStep] Using prefill logits (skipping forward)");
        }
        else
        {
            if (ready_mtp_condition_.has_value())
            {
                result.error =
                    "Ready MTP condition exists without a terminal-logits boundary";
                return result;
            }
            LOG_TRACE("[decodeStep] Running forward with last_token_=" << last_token_);
            /*
             * Single-token decode produces logits that are immediately
             * consumed by GPU sampling below.  When the backend can keep that
             * consumer on device, arm the same one-shot stream handoff used by
             * MTP verification so graph replay does not synchronize merely to
             * hand the logits to the next GPU kernel.
             */
            runner_->setMTPMainDecodeSyncDeferralEnabled(
                can_defer_decode_sampling_sync);
            decode_sampling_sync_deferred = can_defer_decode_sampling_sync;
            std::optional<MoEOverlayInferenceInterferenceScope>
                interference_scope;
            interference_scope.emplace(
                moe_expert_overlay_interference_probe_.get(),
                makeMoEOverlayInferenceWorkloadIdentity(
                    ExpertHistogramSource::DecodeToken,
                    1,
                    1,
                    1,
                    0));
            // Run single-token forward with last token.
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                         << " forward begin token=" << last_token_);
            }
            std::string sequence_error;
            if (!overlay_decode_command.beginGraphSequence(
                    /*draft_depth=*/0, &sequence_error))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error =
                    "ExpertOverlay serial graph-sequence admission failed: " +
                    sequence_error;
                return result;
            }
            if (!runner_->forward(&last_token_, 1))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error = "Forward pass failed during decode";
                return result;
            }
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "decode_condition_forward"))
            {
                result.error = last_error_;
                return result;
            }
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << (mpi_ctx_ ? mpi_ctx_->rank() : 0)
                         << " forward complete position="
                         << trace_position("trace_decode_step_forward_complete"));
            }
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
        }

        // Tail stage: keep GPU logits on the device sampling path.  CPU-only
        // runners may sample from host logits below, but a GPU sampler failure
        // is a hard error rather than a hidden D2H host sampling path.
        int token = -1;
        bool device_sampling_attempted = false;
        bool device_penalty_application_failed = false;
        auto current_stochastic_sample_logical_position = [&]() -> std::optional<int>
        {
            std::string position_error;
            const std::optional<int> mtp_position =
                currentDecodeTransactionPositionForPlanning(
                    "stochastic main-logits sample",
                    &position_error);
            if (mtp_position.has_value())
                return mtp_position;

            LOG_DEBUG("[OrchestrationRunner/decodeStep] could not determine "
                      "logical stochastic sample position: "
                      << position_error);
            return std::nullopt;
        };

        auto sample_current_logits_on_device = [&]() -> int
        {
            device_sampling_attempted = true;
            if (active_sampling_params_.is_greedy())
                return runner_->sampleGreedyOnDevice();

            const std::optional<int> logical_position =
                current_stochastic_sample_logical_position();
            if (!logical_position.has_value())
                return -1;

            const float threshold = mtpSpecStochasticThresholdForPosition(
                active_sampling_params_,
                sampler_,
                *logical_position,
                MTPSpecStochasticDrawPurpose::Sample);
            const int sampled_token =
                runner_->sampleOnDeviceAtLogicalPosition(
                    active_sampling_params_,
                    *logical_position);
            if (sampled_token >= 0)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "sampling",
                    "main_stochastic_logical_position_samples",
                    1.0,
                    {static_cast<uint64_t>(*logical_position), std::bit_cast<uint64_t>(static_cast<double>(threshold))},
                    "decode",
                    {},
                    {{"logical_position_sampler", "true"}});
            }
            return sampled_token;
        };

        const bool mpi_sampling_collective_required =
            mpi_coordinated_world &&
            runner_->requiresMPICoordinatedDecodeSampling(active_sampling_params_);
        const bool worker_sampling_required =
            mpi_worker_rank && mpi_sampling_collective_required;
        /*
         * The coordinated root is always the token authority. `requiresMPICoordinatedDecodeSampling()`
         * only tells an auxiliary rank whether that rank must participate in a
         * sharded sampler collective before root publishes the committed token;
         * it never transfers sampling authority away from root. Treating a
         * false value as "nobody samples" left a GPU continuation rank with
         * token == -1 after a successful prefill and stranded its workers at
         * the post-sampling barrier.
         */
        const bool this_rank_must_sample =
            !mpi_worker_rank || !mpi_coordinated_world ||
            mpi_sampling_collective_required;
        if (mpi_worker_rank && !ready_token_for_decode.has_value())
        {
            /*
             * Worker ranks must participate in any device/distributed sampling
             * collectives, but the root owns the committed token. Do not run
             * root-only CPU sampling here; after the post-sampling fence below,
             * workers receive the authoritative token and record that in their
             * local sampler history.
             */
            if (!worker_sampling_required)
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling skipped; awaiting root token");
                }
                token = 0;
            }
            else if (active_sampling_params_.has_penalties())
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling with penalties begin");
                }
                int vocab = vocabSize();
                auto penalty_map =
                    sampler_.compute_penalty_map(active_sampling_params_, vocab);
                bool gpu_penalties_applied = penalty_map.empty() ||
                    runner_->applyPenaltiesOnDevice(penalty_map, vocab);
                if (gpu_penalties_applied)
                {
                    token = sample_current_logits_on_device();
                }
                else if (runner_->primaryDeviceId().is_gpu())
                {
                    device_penalty_application_failed = true;
                }
            }
            else
            {
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                             << mpi_ctx_->rank()
                             << " worker sampling begin");
                }
                token = sample_current_logits_on_device();
            }
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " worker sampling complete token=" << token);
            }
            if (token < 0)
                token = 0;
        }
        else if (ready_token_for_decode.has_value())
        {
            token = *ready_token_for_decode;
            PerfStatsCollector::addCounter("mtp", "ready_token_direct_emits", 1.0, "decode");
        }
        else if (this_rank_must_sample &&
                 (runner_->primaryDeviceId().is_gpu() ||
                  active_sampling_params_.is_greedy()))
        {
            /*
             * Backend sampling consumes already-penalized logits. CPU
             * stochastic sampling instead builds its distribution below and
             * applies history there exactly once. The backend penalty API also
             * supports CPU rows, so trying it before an unsupported stochastic
             * device sample would mutate the row and double every penalty in
             * the host distribution builder. Select the owner before mutation.
             * CPU greedy candidate reduction remains a backend operation.
             */
            bool penalties_applied = true;
            if (active_sampling_params_.has_penalties())
            {
                const int vocab = vocabSize();
                const auto penalty_map =
                    sampler_.compute_penalty_map(active_sampling_params_, vocab);
                penalties_applied = penalty_map.empty() ||
                    runner_->applyPenaltiesOnDevice(penalty_map, vocab);
            }
            if (penalties_applied)
            {
                token = sample_current_logits_on_device();
                if (token >= 0 && !active_sampling_params_.is_greedy())
                {
                    LOG_TRACE("[decodeStep] GPU top-k/top-p sampled token=" << token);
                }
            }
            else if (runner_->primaryDeviceId().is_gpu())
            {
                device_penalty_application_failed = true;
            }
        }

        if (token < 0)
        {
            if (runner_->primaryDeviceId().is_gpu())
            {
                if (device_penalty_application_failed)
                {
                    result.error =
                        "GPU decode penalty application failed; host logits sampling is CPU-only";
                }
                else if (device_sampling_attempted)
                {
                    result.error =
                        "GPU decode sampling failed; host logits sampling is CPU-only";
                }
                else
                {
                    result.error =
                        "GPU decode sampling was required but not attempted; host logits sampling is CPU-only";
                }
                if (decode_sampling_sync_deferred)
                    result.error += " after deferred logits sync";
                return result;
            }

            // CPU-only host sampling.  GPU runners must not reach this path.
            const float *logits = runner_->logits();
            if (!logits)
            {
                result.error = "No logits available";
                return result;
            }
            int vocab = vocabSize();
            const auto logical_position = current_stochastic_sample_logical_position();
            if (!logical_position)
            {
                result.error = "CPU sampling requires the live logical output position";
                return result;
            }
            token = sampleHostLogitsAtLogicalPosition(
                sampler_, logits, static_cast<size_t>(vocab),
                active_sampling_params_, *logical_position);
        }

        if (mpi_coordinated_world &&
            !moe_overlay_inference_transaction_coordinator_ &&
            !moe_overlay_inference_transaction_follower_)
        {
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " entering post-sampling coordinated fence");
            }
            mpi_ctx_->barrier();
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " leaving post-sampling coordinated fence");
            }

            int32_t coordinated_token = static_cast<int32_t>(token);
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " publishing/receiving coordinated token candidate="
                         << coordinated_token);
            }
            mpi_ctx_->broadcast_int32(
                &coordinated_token, 1, mpi_coordinated_root_rank_);
            token = coordinated_token;
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[OrchestrationRunner/decodeStep] rank="
                         << mpi_ctx_->rank()
                         << " coordinated token=" << token);
            }
        }

        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        if (adaptive_depth_zero_step && !token_is_stop)
        {
            std::string position_error;
            const std::optional<int> base_sidecar_position =
                currentDecodeTransactionPositionForPlanning(
                    "dynamic depth-zero bypass",
                    &position_error);
            if (!base_sidecar_position)
            {
                result.error = position_error;
                return result;
            }
            bool shifted_commit_ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "depth_zero_bypass_shifted_commit",
                    "decode");
                shifted_commit_ok =
                    runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true,
                        *base_sidecar_position);
            }
            if (!shifted_commit_ok)
            {
                result.error =
                    "MTP dynamic depth-zero shifted-cache maintenance failed";
                return result;
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "depth_zero_bypass_shifted_commits",
                1.0,
                "decode");
        }

        // Record token for presence/frequency penalty tracking
        sampler_.record_token(token);

        LOG_TRACE("[decodeStep] sampled token=" << token << " stop_tokens_size=" << stop_tokens_.size());

        result.tokens.push_back(token);
        last_token_ = token; // Store for next decode step

        // Check stop tokens
        result.is_complete = token_is_stop;

        std::string overlay_completion_error;
        if (!overlay_decode_command.complete(&overlay_completion_error))
        {
            result.error =
                "ExpertOverlay serial command completion failed: " +
                overlay_completion_error;
            return result;
        }
        if (!original_command.finish(result.is_complete, terminal_leading()))
        {
            result.error = "Original rank serial generation command could not retire its actual terminal";
            return result;
        }
        decode_command_failure_abort.markCompleted();
        return result;
        }
        catch (const std::exception &error)
        {
            if (result.error.empty()) result.error = error.what();
            LOG_ERROR("[OrchestrationRunner] DECODE_STEP threw before its command terminal: "
                      << error.what());
            throw; // Preserve the original exception in scripted/local callers.
        }
        catch (...)
        {
            if (result.error.empty()) result.error = "DECODE_STEP threw a non-standard exception";
            LOG_ERROR("[OrchestrationRunner] DECODE_STEP threw before its command terminal: non-standard exception");
            throw;
        }
    }

    GenerationResult OrchestrationRunner::forceDecodeToken(int32_t token)
    {
        GenerationResult result;

        if (host_sparse_query_admission_)
        {
            result.error = "Forced decode cannot nest inside an admitted host response";
            return result;
        }
        if (!initialization_lifecycle_.readyForInference())
        {
            result.error = "Runner not initialized";
            return result;
        }
        if (batched_decode_active_)
        {
            result.error =
                "forceDecodeToken() cannot consume request-batched prefill state";
            return result;
        }
        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
        const bool coordinated_force_command =
            mpi_coordinated_mode_ && mpi_ctx_ && mpi_ctx_->world_size() > 1;
        const bool mpi_root_command =
            coordinated_force_command &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_;
        ScopedMPICoordinatedCommandFence force_command_fence(
            mpi_ctx_.get(),
            coordinated_force_command,
            "forceDecodeToken");
        ScopedMPIRootCommandFailureAbort force_command_failure_abort(
            mpi_ctx_.get(),
            mpi_root_command,
            "FORCE_DECODE_TOKEN",
            &result.error);
        ensureFiniteDevicePlacementArmed();
        ScopedMoEOverlayRootCommand overlay_force_command(
            moe_overlay_inference_transaction_coordinator_);

        auto trace_position = [this](const char *context) -> int
        {
            return currentDecodeTransactionPositionForPlanning(context, nullptr)
                .value_or(-1);
        };
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/forceDecodeToken] begin rank="
                     << rank
                     << " token=" << token
                     << " last_token=" << last_token_
                     << " prefill_logits_ready=" << (prefill_logits_ready_ ? "true" : "false")
                     << " position=" << trace_position("trace_force_decode_begin"));
        }

        const std::uint64_t retired_prefill_tokens =
            moe_overlay_pending_prefill_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        const std::uint64_t retired_decode_tokens =
            moe_overlay_pending_decode_progress_tokens_.exchange(
                0u, std::memory_order_acq_rel);
        if (retired_prefill_tokens >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max()) ||
            retired_decode_tokens >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::int32_t>::max()))
        {
            result.error =
                "ExpertOverlay retired inference progress exceeds the typed forced-token payload";
            return result;
        }
        if (mpi_root_command)
        {
            if (traceChatGeneratedTokensEnabled())
                LOG_INFO("[OrchestrationRunner/forceDecodeToken] broadcasting forced token");
            broadcastCommand(MPICommand::FORCE_DECODE_TOKEN);
            std::array<std::int32_t, 3> command_payload{
                token,
                static_cast<std::int32_t>(retired_prefill_tokens),
                static_cast<std::int32_t>(retired_decode_tokens),
            };
            mpi_ctx_->broadcast_int32(
                command_payload.data(),
                command_payload.size(),
                mpi_coordinated_root_rank_);
            force_command_failure_abort.markPublished();
        }
        if (retired_prefill_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::
                          InferencePhase::Prefill,
                      retired_prefill_tokens)))
        {
            result.error =
                "Continuation authority could not publish retired-prefill progress before forced-token execution";
            return result;
        }
        if (retired_decode_tokens != 0u &&
            (!moe_overlay_device_controller_graph_service_ ||
             !moe_overlay_device_controller_graph_service_
                  ->notifyInferenceProgress(
                      MoEOverlayDeviceControllerGraphService::
                          InferencePhase::Decode,
                      retired_decode_tokens)))
        {
            result.error =
                "Continuation authority could not publish retired-decode progress before forced-token execution";
            return result;
        }

        if (overlay_force_command.active())
        {
            const auto command_context = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            if (!command_context ||
                moe_overlay_collective_generation_id_ == 0 ||
                !moe_expert_overlay_residency_authority_)
            {
                result.error =
                    "ExpertOverlay forced-token authority has incomplete root, request-generation, or residency ownership";
                return result;
            }
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                moe_overlay_inference_command_sequence_ ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                result.error =
                    "ExpertOverlay forced-token authority has no valid placement epoch or command identity";
                return result;
            }
            const MoEOverlayInferenceCommandIdentity command_identity{
                .request_generation =
                    moe_overlay_collective_generation_id_,
                .command_id =
                    ++moe_overlay_inference_command_sequence_,
                // Initial admitted lower bound only: mapped device packet kernels
                // authenticate their actual current placement ticket and bank.
                .initial_placement_epoch = snapshot->epoch,
            };
            std::string command_error;
            if (!overlay_force_command.begin(
                    command_context->rank(), command_identity, &command_error))
            {
                result.error =
                    "ExpertOverlay forced-token command admission failed: " +
                    command_error;
                return result;
            }
        }

        auto *const original_global = dynamic_cast<GlobalOrchestrator *>(runner_.get());
        if (original_global && original_global->originalRankGenerationFollower())
        {
            GlobalOrchestrator::OriginalRankGenerationTerminal terminal;
            if (!original_global->serviceOriginalRankGenerationCommand(&terminal, &result.error))
            {
                original_global->abortOriginalRankGenerationCommand();
                if (result.error.empty()) result.error = "Original forced-command follower failed";
                return result;
            }
            if (!decode_transaction_planning_position_ ||
                *decode_transaction_planning_position_ != terminal.admission_base ||
                !publishDecodeTransactionPlanningPositionAfterMTPCommit(terminal.admission_base,
                    terminal.published_state_commit_count, "original_rank_forced_terminal"))
            {
                result.error = "Original forced terminal differs from its admitted scheduler boundary";
                return result;
            }
            ready_mtp_condition_.reset();
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
            prefill_logits_ready_ = !terminal.model_stopped;
            result.is_complete = terminal.model_stopped;
            result.returned_token_commit = terminal.model_stopped ? ReturnedTokenCommitState::NoModelRow : ReturnedTokenCommitState::Pending;
            force_command_failure_abort.markCompleted();
            return result;
        }
        /** @brief Keep original FORCE transport and transaction ownership absorbing on failure. */
        struct OriginalRankForceCommandGuard
        {
            GlobalOrchestrator *owner = nullptr; ///< Actual original vocabulary-tail source only.
            bool completed = false; ///< Set only after native terminal and channel retirement.
            /** @brief Cancel original directions before the outer MPI abort fence destroys owners. */
            ~OriginalRankForceCommandGuard() { if (owner && !completed) owner->abortOriginalRankGenerationCommand(); }
        } original_force{original_global && original_global->originalRankGenerationSource() ? original_global : nullptr};
        if (original_force.owner && (!decode_transaction_planning_position_ ||
            moe_overlay_inference_command_sequence_ == std::numeric_limits<uint64_t>::max() ||
            !original_force.owner->beginOriginalRankGenerationCommand(
                original_force.owner->originalRankCommandIdentity(++moe_overlay_inference_command_sequence_),
                *decode_transaction_planning_position_, active_sampling_params_.is_greedy() ?
                    DeviceGenerationSamplingMode::Greedy : DeviceGenerationSamplingMode::Stochastic)))
        {
            result.error = "Original forced-command source lost its exact request/command admission";
            return result;
        }

        const auto begin_serial_overlay_graph = [&]() -> bool
        {
            std::string sequence_error;
            if (!overlay_force_command.beginGraphSequence(
                    /*draft_depth=*/0, &sequence_error))
            {
                result.error =
                    "ExpertOverlay forced-token graph-sequence admission failed: " +
                    sequence_error;
                return false;
            }
            return true;
        };

        const bool token_is_stop =
            std::find(stop_tokens_.begin(), stop_tokens_.end(), token) != stop_tokens_.end();
        std::optional<HostSparseQueryResponseScope> forced_query_scope;
        try
        {
            forced_query_scope.emplace(*runner_, host_sparse_query_admission_,
                token_is_stop ? std::nullopt : hostSparseQueryAdmission(0, true));
        }
        catch (const std::exception &error)
        {
            last_error_ = result.error = error.what();
            return result;
        }
        const bool gpu_mtp_force =
            shouldUseMTPDecode() &&
            runner_->primaryDeviceId().is_gpu();
        const bool gpu_ordinary_force =
            !activeMTPRequestConfig().enabled &&
            runner_->primaryDeviceId().is_gpu();

        if ((gpu_mtp_force || gpu_ordinary_force) && token_is_stop)
        {
            /*
             * A terminal control token ends generation and therefore never
             * becomes a model-state row. Keep this as an explicit GPU-MTP
             * lifecycle state instead of allowing it to enter the CPU/non-MTP
             * branch below. The ready logits and their sampled shadows belong
             * to the completed request and are retired without publishing,
             * committing, or forwarding the terminal token on any device.
             */
            prefill_logits_ready_ = false;
            ready_mtp_condition_.reset();
            result.returned_token_commit =
                ReturnedTokenCommitState::NoModelRow;
            PerfStatsCollector::addCounter(
                "mtp",
                "forced_stop_token_state_mutations_skipped",
                1.0,
                "decode",
                {},
                {{"state_transition", "terminal_control_only"},
                 {"device_execution", "none"}});
        }
        else if (gpu_mtp_force)
        {
            /*
             * A forced policy token replaces the sample selected from an already
             * materialized terminal-logits boundary. An unreturned bonus can
             * be replaced; an AlreadyEmitted condition is part of the public
             * response and must first enter shifted/main state exactly once.
             * Neither case samples or replays a host token. The forced token
             * then becomes the next row and recreates terminal logits for a
             * following policy token in a multi-token stop-thinking sequence.
             */
            if (!prefill_logits_ready_)
            {
                result.error =
                    "GPU MTP forced decode requires a ready terminal-logits boundary";
                return result;
            }

            if (ready_mtp_condition_)
            {
                if (!ready_mtp_condition_->valid() ||
                    !ready_mtp_condition_->isDeviceResidentOnly() ||
                    !samplingParamsEqual(
                        ready_mtp_condition_->sampling_params,
                        active_sampling_params_))
                {
                    result.error =
                        "GPU MTP forced decode received a stale or foreign resident condition";
                    return result;
                }
                if (ready_mtp_condition_->wasAlreadyEmitted())
                {
                    const auto resident = *ready_mtp_condition_->resident_state;
                    // Preserve the original token before shifted commit consumes
                    // its one-shot mailbox/lease. AlreadyPublished closes this
                    // target event's history right without incrementing counts or
                    // changing the grouped carry's penalty-frontier policy bit.
                    if (!runner_->publishDeviceResidentConditionTokenToTargetSampleSlot(
                            resident, /*request_index=*/0, /*target_sample_slot=*/0) ||
                        !runner_->publishMTPDirectTokenHistory(
                            0, MTPDirectTokenHistoryDisposition::AlreadyPublished) ||
                        !runner_->commitMTPShiftedRowFromDeviceResidentLogicalState(
                            resident, /*request_index=*/0,
                            /*already_appended_tokens=*/0,
                            /*allow_speculative_discard=*/true))
                    {
                        result.error =
                            "GPU MTP forced decode could not preserve its previously emitted resident condition";
                        return result;
                    }
                    runner_->setMTPMainDecodeSyncDeferralEnabled(true);
                    if (!begin_serial_overlay_graph())
                    {
                        runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                        return result;
                    }
                    if (!runner_->advanceMTPMainConditionFromDeviceTargetSample(
                            last_token_, /*target_sample_slot=*/0,
                            MTPConditionForwardPurpose::CommittedSerialToken))
                    {
                        runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                        result.error =
                            "GPU MTP forced decode could not commit its previously emitted resident condition";
                        return result;
                    }
                    if (!advanceDecodeTransactionPlanningPositionAfterForward(
                            "forced_mtp_resident_condition_forward"))
                    {
                        result.error = last_error_;
                        return result;
                    }
                    std::string sequence_error;
                    if (!overlay_force_command.retireGraphSequence(&sequence_error))
                    {
                        result.error =
                            "ExpertOverlay forced-token prior-condition graph retirement failed: " +
                            sequence_error;
                        return result;
                    }
                    // Sparse-query admission is immutable and proved for one
                    // forward at its initial target position. Close that row's
                    // policy before admitting the actual forced row at P+1.
                    forced_query_scope.reset();
                    try
                    {
                        forced_query_scope.emplace(*runner_, host_sparse_query_admission_,
                            hostSparseQueryAdmission(0, true));
                    }
                    catch (const std::exception &error)
                    {
                        last_error_ = result.error = error.what();
                        return result;
                    }
                }
            }
            prefill_logits_ready_ = false;
            ready_mtp_condition_.reset();

            /*
             * Request policy selected `token` on the host, but execution state
             * becomes device-owned here. The backend publishes the scalar with a
             * one-thread kernel into the persistent target slot and records the
             * exact producer stream. No stack-backed H2D copy, allocation,
             * default stream, or synchronization is permitted.
             */
            if (!runner_->stageStochasticTargetTokenForDeviceSampling(
                    token,
                    /*target_sample_slot=*/0))
            {
                result.error =
                    "GPU MTP forced decode could not publish its control token to device state";
                return result;
            }

            if (!runner_->publishMTPDirectTokenHistory(0,
                    MTPDirectTokenHistoryDisposition::NewlyEmitted))
            {
                result.error = "GPU MTP forced output could not publish its original device-token history";
                return result;
            }

            /*
             * Shifted MTP KV consumes the device token first, while terminal
             * hidden still names the state immediately before that token. The
             * commit's transaction lease and target-slot readiness event order
             * this mutation without observing a host position mirror.
             */
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "forced_token_shifted_commit_device",
                    "decode");
                if (!runner_->commitMTPShiftedRowFromDeviceTargetSample(
                        /*target_sample_slot=*/0,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true))
                {
                    result.error =
                        "GPU MTP forced-token shifted-cache maintenance failed";
                    return result;
                }
            }

            runner_->setMTPMainDecodeSyncDeferralEnabled(true);
            if (!begin_serial_overlay_graph())
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                return result;
            }
            if (!runner_->advanceMTPMainConditionFromDeviceTargetSample(
                    token,
                    /*target_sample_slot=*/0,
                    MTPConditionForwardPurpose::CommittedSerialToken))
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                result.error =
                    "GPU MTP forced-token main-state advance failed";
                return result;
            }
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "forced_decode_device_condition_forward"))
            {
                result.error = last_error_;
                return result;
            }

            prefill_logits_ready_ = true;
            result.returned_token_commit =
                ReturnedTokenCommitState::Committed;
            PerfStatsCollector::addCounter(
                "mtp",
                "forced_token_device_owned_transactions",
                1.0,
                "decode",
                {},
                {{"publication", "control_scalar_kernel"},
                 {"shifted_state", "device_target_slot"},
                 {"main_forward", "device_token_ids"}});
        }
        else if (gpu_ordinary_force)
        {
            using Leading =
                sampling_math::DeviceGenerationLeadingRowDisposition;
            if (!prefill_logits_ready_ ||
                !device_generation_admission_.awaitsAdmission())
            {
                result.error =
                    "GPU ordinary forced decode requires an admission-ready resident boundary";
                return result;
            }

            /*
             * A bounded publication window returns its final token before that
             * token enters KV. Commit that exact mailbox row first. The
             * retained ordinary graph consumes device token and position rows;
             * the host scalar is diagnostic identity only.
             */
            if (ready_mtp_condition_)
            {
                if (!ready_mtp_condition_->valid() ||
                    !ready_mtp_condition_->isDeviceResidentOnly() ||
                    !ready_mtp_condition_->wasAlreadyEmitted() ||
                    !samplingParamsEqual(
                        ready_mtp_condition_->sampling_params,
                        active_sampling_params_))
                {
                    result.error =
                        "GPU ordinary forced decode received a stale or foreign resident condition";
                    return result;
                }
                const auto resident =
                    *ready_mtp_condition_->resident_state;
                if (!resident.coversRequest(0))
                {
                    result.error =
                        "GPU ordinary forced decode has no request-zero resident condition";
                    return result;
                }
                if (!begin_serial_overlay_graph())
                    return result;
                runner_->setMTPMainDecodeSyncDeferralEnabled(true);
                if (!runner_
                         ->advanceOrdinaryMainConditionFromDeviceResidentLogicalState(
                             last_token_, resident, /*request_index=*/0))
                {
                    runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                    result.error =
                        "GPU ordinary forced decode could not commit its prior resident condition";
                    return result;
                }
                if (!advanceDecodeTransactionPlanningPositionAfterForward(
                        "forced_ordinary_resident_condition_forward"))
                {
                    result.error = last_error_;
                    return result;
                }
                // The sampler-free forward has retired its actual device
                // transaction. A future ordinary response cannot continue a
                // ForwardOnly policy row. Preserve live model/history state
                // while opening the typed fresh-response admission edge.
                if (!device_generation_admission_.reopenAfterOrdinaryForwardOnly())
                {
                    result.error =
                        "GPU ordinary forced decode could not retire its completed ForwardOnly response boundary";
                    return result;
                }
                ready_mtp_condition_.reset();
            }

            /*
             * The request-selected scalar becomes the next already-emitted
             * condition without entering a host token replay. Publication also
             * advances device penalty history, so the next captured ordinary
             * window sees exactly the output sequence visible to the client.
             */
            if (!runner_->publishForcedDeviceResidentConditionToken(
                    token, /*target_sample_slot=*/0))
            {
                result.error =
                    "GPU ordinary forced decode could not publish its device-resident policy token";
                return result;
            }
            auto state = runner_->deviceResidentLogicalSequenceState();
            if (!state.coversRequest(0))
            {
                result.error =
                    "GPU ordinary forced decode did not publish a live condition mailbox";
                return result;
            }
            ready_mtp_condition_ = ReadyMTPCondition::deviceResident(
                active_sampling_params_, std::move(state),
                Leading::AlreadyEmitted);
            if (!ready_mtp_condition_->valid())
            {
                result.error =
                    "GPU ordinary forced decode published an incomplete resident condition";
                return result;
            }
            prefill_logits_ready_ = true;
            result.returned_token_commit = ReturnedTokenCommitState::Pending;
            PerfStatsCollector::addCounter(
                "generation",
                "forced_ordinary_device_resident_conditions",
                1.0,
                "decode",
                {},
                {{"publication", "control_scalar_kernel"},
                 {"condition", "already_emitted"},
                 {"host_replay", "false"}});
        }
        else
        {
            result.returned_token_commit =
                token_is_stop
                    ? ReturnedTokenCommitState::NoModelRow
                    : ReturnedTokenCommitState::Pending;
            /*
             * CPU and non-MTP execution retain their serial token contract. A
             * forced token chosen at request level replaces ready logits when
             * present; otherwise the previous token must first advance ordinary
             * serial decode state.
             */
            if (prefill_logits_ready_)
            {
                prefill_logits_ready_ = false;
                ready_mtp_condition_.reset();
            }
            else if (!token_is_stop)
            {
                runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[OrchestrationRunner/forceDecodeToken] rank="
                             << rank << " forwarding previous token " << last_token_);
                }
                if (!begin_serial_overlay_graph())
                    return result;
                if (!runner_->forward(&last_token_, 1))
                {
                    result.error =
                        "Forward pass failed while committing forced token";
                    /*
                     * The forward failure is already terminal, so observing
                     * the system-coherent barrier record here cannot affect
                     * admission or ordering.  Preserve that exact mapped
                     * lifecycle evidence in the propagated error: an MPI
                     * follower may otherwise abort before buffered device
                     * diagnostics become visible, leaving an indistinguishable
                     * generic ticket-service timeout.
                     */
                    if (moe_overlay_device_controller_fabric_)
                    {
                        result.error += "; ";
                        result.error +=
                            moe_overlay_device_controller_fabric_
                                ->describeInferenceEpochBarrier();
                    }
                    return result;
                }
                if (!advanceDecodeTransactionPlanningPositionAfterForward(
                        "forced_decode_condition_forward"))
                {
                    result.error = last_error_;
                    return result;
                }
            }

            /*
             * CPU MTP maintains the same shifted sidecar stream using its
             * serial-row implementation. GPU MTP never enters this host-token
             * API: its complete device-owned transaction is handled above.
             */
            if (shouldUseMTPDecode() && !token_is_stop)
            {
                std::string position_error;
                const std::optional<int> base_sidecar_position =
                    currentDecodeTransactionPositionForPlanning(
                        "forced decode token",
                        &position_error);
                if (!base_sidecar_position)
                {
                    result.error = position_error;
                    return result;
                }

                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "forced_token_shifted_commit",
                    "decode");
                if (!runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                        token,
                        /*already_appended_tokens=*/0,
                        /*allow_speculative_discard=*/true,
                        *base_sidecar_position))
                {
                    result.error =
                        "MTP forced-token shifted-cache maintenance failed";
                    return result;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "forced_token_shifted_commits",
                    1.0,
                    "decode");
            }
        }

        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();

        sampler_.record_token(token);
        last_token_ = token;
        result.tokens.push_back(token);
        result.is_complete = token_is_stop;
        if (traceChatGeneratedTokensEnabled())
        {
            LOG_INFO("[OrchestrationRunner/forceDecodeToken] done rank="
                     << rank
                     << " token=" << token
                     << " position=" << trace_position("trace_force_decode_done"));
        }

        std::string overlay_completion_error;
        if (!overlay_force_command.complete(&overlay_completion_error))
        {
            result.error =
                "ExpertOverlay forced-token command completion failed: " +
                overlay_completion_error;
            return result;
        }
        if (original_force.owner)
        {
            const auto leading = ready_mtp_condition_ && ready_mtp_condition_->wasAlreadyEmitted() ?
                sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted :
                sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;
            if (!original_force.owner->completeOriginalRankGenerationCommand(token_is_stop, leading))
            {
                result.error = "Original forced-command terminal or transport retirement failed";
                return result;
            }
            original_force.completed = true;
        }
        force_command_failure_abort.markCompleted();
        return result;
    }

    std::optional<DeviceGenerationAdmissionRequest>
    OrchestrationRunner::hostSparseQueryAdmission(int budget, bool forward_only) const
    {
        if (!runner_ || (!forward_only && runner_->primaryDeviceId().is_gpu()) || host_sparse_query_admission_)
            return std::nullopt;
        std::string error;
        const auto initial = currentDecodeTransactionPositionForPlanning("host_response_admission", &error);
        if (!initial) throw std::invalid_argument(error);
        if (*initial < 0)
            throw std::invalid_argument("Host response initial target count is negative");
        if (forward_only)
        {
            DeviceGenerationAdmissionRequest request{
                .request_count = 1,
                .max_new_tokens = 0,
                .depth_policy = sampling_math::DeviceGenerationPolicy::forwardOnly(),
                .initial_leading_row_disposition =
                    sampling_math::DeviceGenerationLeadingRowDisposition::AlreadyEmitted,
            };
            request.initial_target_counts.emplace(std::vector<int>{*initial});
            if (!request.valid()) throw std::invalid_argument("Invalid forced sparse-query admission");
            return request;
        }
        if (budget == 0)
        {
            const int capacity = plan_.runtime.max_seq_len > 0 ? plan_.runtime.max_seq_len : config_.max_seq_len;
            const int64_t remaining = static_cast<int64_t>(capacity) - *initial;
            if (remaining <= 0 || remaining > std::numeric_limits<int>::max())
                throw std::invalid_argument("Host response has no valid context-derived budget");
            budget = static_cast<int>(remaining);
        }
        if (budget <= 0)
            throw std::invalid_argument("Host response budget is not positive");
        // Positive caller budgets retain their existing semantics; the shared
        // visibility resolver checks overflow without adding a context ceiling.
        const auto mtp = activeMTPRequestConfig();
        DeviceGenerationAdmissionRequest request{
            .request_count = 1,
            .max_new_tokens = budget,
            .depth_policy = resolveSparseQueryHostDepthEnvelope(mtp),
        };
        request.initial_target_counts.emplace(std::vector<int>{*initial});
        if (!request.valid()) throw std::invalid_argument("Invalid immutable host response admission");
        return request;
    }

    void OrchestrationRunner::setDecodeStepTokenBudget(int max_tokens)
    {
        decode_step_token_budget_ = std::max(0, max_tokens);
    }

    GenerationResult OrchestrationRunner::generate(
        const std::vector<int32_t> &prompt_tokens,
        int max_new_tokens,
        const SamplingParams &sampling)
    {
        GenerationResult result;

        if (!initialization_lifecycle_.readyForInference())
        {
            result.error = "Runner not initialized";
            return result;
        }

        /*
         * Sampling policy is request admission state. Install it before prefill
         * so GPU request-input or prefix-restore publication can initialize the
         * graph-stable penalty policy on its exact reset-ordered stream.
         */
        active_sampling_params_ = sampling;
        sampler_ = Sampler(sampling.seed);

        // Prefill
        // After prefill, the first decodeStep() samples from the prefill logits
        // directly (via prefill_logits_ready_ flag) instead of re-feeding the
        // last prompt token. GPU-side sampling (sampleGreedyOnDevice) works on
        // device logits without D2H, so no gather is needed for GPU models.
        // For CPU models, logits are already on host.
        if (!prefill(prompt_tokens))
        {
            result.error = last_error_;
            return result;
        }

        std::optional<HostSparseQueryResponseScope> query_response_scope;
        try
        {
            query_response_scope.emplace(*runner_, host_sparse_query_admission_,
                max_new_tokens > 0 ? hostSparseQueryAdmission(max_new_tokens) : std::nullopt);
        }
        catch (const std::exception &error)
        {
            last_error_ = result.error = error.what();
            return result;
        }

        while (static_cast<int>(result.tokens.size()) < max_new_tokens)
        {
            // Use decodeStep() which uses last_token_ internally
            decode_step_token_budget_ = max_new_tokens - static_cast<int>(result.tokens.size());
            GenerationResult step = decodeStep();
            decode_step_token_budget_ = 0;

            if (!step.error.empty())
            {
                result.error = step.error;
                break;
            }

            // Collect tokens from step (on tail stage)
            for (int32_t token : step.tokens)
            {
                result.tokens.push_back(token);
            }

            if (step.is_complete)
            {
                result.is_complete = true;
                break;
            }

            if (!maybeApplyMoERebalance(step.tokens.size()))
            {
                result.error = last_error_.empty() ? "MoE rebalance failed" : last_error_;
                break;
            }
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        if (mtp.enabled)
        {
            LOG_INFO("[OrchestrationRunner] MTP summary: draft_steps="
                     << mtp_stats_.draft_steps
                     << " accepted_tokens=" << mtp_stats_.accepted_tokens
                     << " rejected_tokens=" << mtp_stats_.rejected_tokens
                     << " rollbacks=" << mtp_stats_.rollbacks
                     << " bypasses=" << mtp_stats_.bypasses
                     << " last_bypass_reason="
                     << (mtp_bypass_reason_.empty() ? "none" : mtp_bypass_reason_)
                     << " verifier_runs=" << mtp_stats_.verifier_runs
                     << " verifier_tokens=" << mtp_stats_.verifier_token_count
                     << " verify_mode=" << mtpVerifyModeToString(mtp.verify_mode)
                     << " stochastic_accept_tests=" << mtp_stats_.stochastic_accept_tests
                     << " stochastic_residual_samples=" << mtp_stats_.stochastic_residual_samples
                     << " stochastic_terminal_samples=" << mtp_stats_.stochastic_terminal_samples
                     << " depth_policy=" << mtpDepthPolicyModeToString(mtp.depth_policy.mode)
                     << " current_depth="
                     << (mtp_depth_controller_ ? mtp_depth_controller_->currentDepth() : mtp.draft_tokens)
                     << " depth_updates=" << mtp_stats_.depth_policy_updates);
        }

        return result;
    }

    uint64_t OrchestrationRunner::moeRuntimeMovementEpoch() const
    {
        std::uint64_t epoch =
            runner_ ? runner_->moeRuntimeMovementEpoch() : 0u;
        if (moe_expert_overlay_residency_authority_)
        {
            /*
             * Heterogeneous placement is RCU-owned by this sole authority.
             * Query its immutable publication directly instead of retaining a
             * second host epoch mirror beside the child runner.
             */
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid())
            {
                throw std::logic_error(
                    "ExpertOverlay runtime movement epoch has no valid authoritative snapshot");
            }
            epoch = std::max(epoch, snapshot->epoch);
        }
        if (moe_overlay_device_controller_fabric_)
        {
            epoch = std::max(
                epoch,
                moe_overlay_device_controller_fabric_
                    ->completedDurableMovementEpochs());
        }
        return epoch;
    }

    MoEOptimizationMovementTopology OrchestrationRunner::moeOptimizationMovementTopology() const
    {
        if (!moe_pipeline_prepared_plan_)
            return describeMoEOptimizationMovementTopology(config_.moe_routed_expert_plan.get());
        const auto &prepared = moe_pipeline_prepared_plan_->stages();
        const auto &admitted = moe_pipeline_prepared_plan_->admission()->stages();
        std::vector<MoEOptimizationStages<MoEOptimizationMovementTopology>::Entry> stages;
        stages.reserve(prepared.size());
        for (std::size_t i = 0; i < prepared.size(); ++i)
        {
            const auto &scope = admitted.at(i).topology().scope();
            const auto &metadata = prepared[i].metadata;
            // The admitted continuation domain owns the participant identity
            // for both singleton and TP stages. A singleton's rank projection
            // deliberately has no local_tp_devices and cannot supply this view.
            const auto &placement = *prepared[i].placement;
            const auto continuation = std::find_if(placement.domains.begin(), placement.domains.end(),
                [&](const auto &domain) { return domain.name == placement.effectiveBaseModelDomain(); });
            if (continuation == placement.domains.end())
                throw std::logic_error("Pipeline movement topology lost its admitted continuation domain");
            stages.push_back({MoEOptimizationStageIdentity{i, scope.first_layer, scope.last_layer,
                metadata.first_model_layer + metadata.num_layers, continuation->participants, scope.has_lm_head},
                describeMoEOptimizationMovementTopology(prepared[i].placement.get())});
        }
        return composeMoEOptimizationMovementTopology(MoEOptimizationStages<MoEOptimizationMovementTopology>::seal(std::move(stages)));
    }

    std::optional<MoEOptimizationStatus>
    OrchestrationRunner::localPipelineStageOptimizationStatus(int stage_id) const
    {
        const auto *pipeline = dynamic_cast<const GlobalOrchestrator *>(runner_.get());
        const auto *stage = pipeline ? pipeline->stageRunnerForStage(stage_id) : nullptr;
        if (!stage) return std::nullopt;
        return stage->moeOptimizationStatus();
    }

    MoEOptimizationStatus OrchestrationRunner::moeOptimizationStatus() const
    {
        const auto native_status = runner_ ? runner_->moeOptimizationStatus() : MoEOptimizationStatus{};
        if (native_status.authority != MoEOptimizationAuthority::None)
        {
            if (moe_expert_overlay_maintenance_service_ || moe_overlay_device_controller_graph_service_)
                throw std::logic_error("Native and external MoE optimization authorities are both live");
            return native_status;
        }
        if (moe_expert_overlay_maintenance_service_ &&
            moe_overlay_device_controller_graph_service_)
        {
            return {
                .authority = MoEOptimizationAuthority::None,
                .state = MoEOptimizationLifecycleState::Failed,
                .activity = MoEOptimizationActivityState::Failed,
                .diagnostic =
                    "Host and device ExpertOverlay optimization authorities are both live",
            };
        }
        if (moe_expert_overlay_maintenance_service_)
            return moe_expert_overlay_maintenance_service_->optimizationStatus();
        if (moe_overlay_device_controller_graph_service_)
        {
            return moe_overlay_device_controller_graph_service_
                ->optimizationStatus({
                    moe_overlay_pending_prefill_progress_tokens_.load(
                        std::memory_order_acquire),
                    moe_overlay_pending_decode_progress_tokens_.load(
                        std::memory_order_acquire),
                });
        }
        if (terminal_moe_optimization_status_)
            return *terminal_moe_optimization_status_;
        if (!moe_expert_overlay_residency_authority_)
            return {};

        const auto authority_stats =
            moe_expert_overlay_residency_authority_->stats();
        if (moe_expert_overlay_residency_authority_->maintenanceMode() !=
            MoERebalanceRuntimeMode::Dynamic)
        {
            return {
                .authority = MoEOptimizationAuthority::Host,
                .state =
                    MoEOptimizationLifecycleState::MovementDisabled,
                .activity = MoEOptimizationActivityState::Dormant,
                .published_movement_waves =
                    authority_stats.committed_waves,
                .completed_movement = {
                    .transactions = authority_stats.committed_waves,
                    .commands = authority_stats.committed_migrations,
                    .physical_bytes =
                        moe_expert_overlay_migration_transport_
                            ? moe_expert_overlay_migration_transport_
                                  ->completedPlacementPayloadBytes()
                            : 0u,
                    .promotions = authority_stats.promotions,
                    .demotions = authority_stats.demotions,
                    .same_priority_moves =
                        authority_stats.same_priority_moves,
                },
            };
        }
        return {
            .authority = MoEOptimizationAuthority::Host,
            .state = MoEOptimizationLifecycleState::Failed,
            .activity = MoEOptimizationActivityState::Failed,
            .published_movement_waves = authority_stats.committed_waves,
            .completed_movement = {
                .transactions = authority_stats.committed_waves,
                .commands = authority_stats.committed_migrations,
                .physical_bytes =
                    moe_expert_overlay_migration_transport_
                        ? moe_expert_overlay_migration_transport_
                              ->completedPlacementPayloadBytes()
                        : 0u,
                .promotions = authority_stats.promotions,
                .demotions = authority_stats.demotions,
                .same_priority_moves =
                    authority_stats.same_priority_moves,
            },
            .diagnostic =
                "Dynamic ExpertOverlay authority has no live optimization service",
        };
    }

    MoEOptimizationMovementLedger
    OrchestrationRunner::moeOptimizationMovementLedger() const
    {
        if (runner_ && runner_->moeOptimizationStatus().authority != MoEOptimizationAuthority::None)
        {
            if (moe_expert_overlay_maintenance_service_ || moe_overlay_device_controller_graph_service_)
                throw std::logic_error("Native and external MoE movement publishers are both live");
            return runner_->moeOptimizationMovementLedger();
        }
        if (moe_expert_overlay_maintenance_service_ &&
            moe_overlay_device_controller_graph_service_)
        {
            return {
                .edges = {},
                .discarded_edges = 1u,
            };
        }
        if (moe_expert_overlay_maintenance_service_ &&
            moe_expert_overlay_residency_authority_)
        {
            return moe_expert_overlay_residency_authority_->movementLedger();
        }
        if (moe_overlay_device_controller_graph_service_)
        {
            return moe_overlay_device_controller_graph_service_
                ->movementLedger();
        }
        if (terminal_moe_optimization_movement_ledger_)
            return *terminal_moe_optimization_movement_ledger_;
        if (moe_expert_overlay_residency_authority_)
            return moe_expert_overlay_residency_authority_->movementLedger();
        return {};
    }

    bool OrchestrationRunner::maybeApplyMoERebalance(
        uint64_t committed_tokens)
    {
        if (committed_tokens == 0u)
        {
            return setError(
                "ExpertOverlay maintenance cannot retire an empty decode transaction");
        }
        const std::string device =
            runner_ && runner_->primaryDeviceId().is_valid()
                ? runner_->primaryDeviceId().toString()
                : std::string{};

        for (const auto &binding : moe_pipeline_stage_bindings_)
            if (const auto &owner = binding->owners().host_maintenance; owner)
            {
                if (!owner->ready()) return setError("CPU pipeline host maintenance is not healthy/prepared");
                owner->notify({.phase = MoEOverlayInferenceProgressPhase::Decode, .completed_logical_tokens = committed_tokens});
            }
        if (device_generation_embedded_moe_maintenance_pending_ack_)
        {
            if (!runner_ ||
                runner_->moeOverlayAuthorityExecution() !=
                    MoEOverlayAuthorityExecutionKind::
                        DeviceResident ||
                !runner_
                     ->deviceResidentMoEOverlayMaintenanceReady())
            {
                return setError(
                    "Embedded ExpertOverlay maintenance acknowledgement lost its device-resident authority binding");
            }
            device_generation_embedded_moe_maintenance_pending_ack_ = false;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "device_generation_embedded_maintenance_acknowledgements",
                static_cast<double>(committed_tokens),
                "maintenance",
                device,
                {{"authority_execution", "device_resident"},
                 {"outer_host_launch", "false"},
                 {"replayed", "false"}});
            return true;
        }

        if (moe_overlay_device_controller_graph_service_)
        {
            if (!moe_overlay_device_controller_graph_service_->healthy())
            {
                return setError(
                    "Topology-wide device ExpertOverlay maintenance failed: " +
                    moe_overlay_device_controller_graph_service_
                        ->failureMessage());
            }
            if (config_.moe_rebalance.mode !=
                MoERebalanceRuntimeMode::Dynamic)
            {
                return true;
            }

            const std::uint64_t hosted_progress =
                hosted_device_generation_decode_progress_retired_.exchange(
                    0u, std::memory_order_acq_rel);
            if (hosted_progress != 0u)
            {
                if (hosted_progress != committed_tokens)
                {
                    return setError(
                        "Hosted device-controller progress disagrees with the terminal committed-token boundary: ticketed=" +
                        std::to_string(hosted_progress) +
                        ", terminal=" + std::to_string(committed_tokens));
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "hosted_device_generation_progress_acknowledgements",
                    static_cast<double>(committed_tokens),
                    "maintenance",
                    device,
                    {{"blocking", "false"},
                     {"authority_execution", "device_resident"},
                     {"scheduler", "authenticated_transaction_ticket"},
                     {"replayed", "false"}});
                return true;
            }

            /*
             * A non-hosted generation path has no per-sequence transaction
             * ticket. Queue its committed boundary without issuing another
             * collective; the next ordinary decode/forced-token payload
             * sidebands this continuation-authored count to every rank. Hosted
             * paths returned above because their authenticated tickets already
             * woke every process-local controller at the true retirement edge.
             */
            auto &pending = moe_overlay_pending_decode_progress_tokens_;
            std::uint64_t observed = pending.load(std::memory_order_relaxed);
            const std::uint64_t payload_max =
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::int32_t>::max());
            if (committed_tokens > payload_max)
            {
                return setError(
                    "One topology-wide device ExpertOverlay decode transaction exceeds the typed command sideband");
            }
            while (observed <= payload_max - committed_tokens &&
                   !pending.compare_exchange_weak(
                       observed,
                       observed + committed_tokens,
                       std::memory_order_release,
                       std::memory_order_relaxed))
            {
                // compare_exchange refreshes observed before retrying.
            }
            if (observed > payload_max - committed_tokens)
            {
                return setError(
                    "Topology-wide device ExpertOverlay decode progress exceeds the typed command sideband");
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "decode_boundary_progress_queued",
                static_cast<double>(committed_tokens),
                "maintenance",
                device,
                {{"blocking", "false"},
                 {"authority_execution", "device_resident"},
                 {"scheduler", "topology_wide_device_controller"},
                 {"sideband", "next_inference_command"},
                 {"extra_mpi_command", "false"},
                 {"physical_follower", "host_byte_transport_only"}});
            return true;
        }

        if (moe_pipeline_prepared_plan_)
        {
            if (config_.moe_rebalance.mode != MoERebalanceRuntimeMode::Dynamic)
                return true;
            if (!runner_ || moe_pipeline_stage_bindings_.empty() ||
                moe_pipeline_stage_bindings_.size() != moe_pipeline_prepared_plan_->stages().size())
                return setError("MoE pipeline progress lost its complete stage ownership");
            const bool all_no_peer = std::all_of(moe_pipeline_stage_bindings_.begin(),
                moe_pipeline_stage_bindings_.end(), [](const auto &binding) {
                    return binding && binding->owners().authority->noPeerResidency().has_value();
                });
            // All-zero topology work needs no participant fan-out, scheduler
            // ticket, host mirror, or per-token evidence publication.
            if (all_no_peer) return true;
            return runner_ && runner_->maybeApplyDecodeBoundaryMaintenance(committed_tokens);
        }

        if (usesNativeGpuStageMoEOverlayAuthority())
        {
            if (config_.moe_rebalance.mode !=
                MoERebalanceRuntimeMode::Dynamic)
            {
                return true;
            }
            if (!runner_ ||
                runner_->moeOverlayAuthorityExecution() !=
                    MoEOverlayAuthorityExecutionKind::
                        DeviceResident)
            {
                return setError(
                    "Native GPU-stage device-resident ExpertOverlay lost its graph-family authority binding");
            }
            try
            {
                if (!runner_->maybeApplyDecodeBoundaryMaintenance(
                        committed_tokens))
                {
                    return setError(
                        "Native GPU-stage device-resident ExpertOverlay maintenance enqueue failed");
                }
            }
            catch (const std::exception &error)
            {
                return setError(
                    std::string(
                        "Native GPU-stage device-resident ExpertOverlay maintenance failed: ") +
                    error.what());
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "decode_boundary_background_notifications",
                static_cast<double>(committed_tokens),
                "maintenance",
                device,
                {{"blocking", "false"},
                 {"authority_execution",
                  "device_resident"},
                 {"scheduler", "captured_device_stream"}});
            return true;
        }

        if (moe_expert_overlay_maintenance_service_)
        {
            if (!moe_expert_overlay_maintenance_service_->healthy())
            {
                return setError(
                    "ExpertOverlay background residency maintenance failed: " +
                    moe_expert_overlay_maintenance_service_
                        ->failureMessage());
            }

            const std::uint64_t hosted_progress =
                hosted_device_generation_decode_progress_retired_.exchange(
                    0u, std::memory_order_acq_rel);
            if (hosted_progress != 0u)
            {
                if (hosted_progress != committed_tokens)
                {
                    return setError(
                        "Hosted ExpertOverlay transaction progress disagrees "
                        "with the terminal committed-token boundary: ticketed=" +
                        std::to_string(hosted_progress) +
                        ", terminal=" + std::to_string(committed_tokens));
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "hosted_device_generation_progress_acknowledgements",
                    static_cast<double>(committed_tokens),
                    "maintenance",
                    device,
                    {{"blocking", "false"},
                     {"authority_execution", "host_coordinated"},
                     {"scheduler", "authenticated_hip_ticket"},
                     {"replayed", "false"}});
                return true;
            }

            /* Logical progress only rate-limits a later exact histogram drain;
             * inference never polls, joins, or waits for movement. */
            notifyHostMoEOverlayInferenceProgress(
                MoEOverlayInferenceProgressPhase::Decode,
                committed_tokens);
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "decode_boundary_background_notifications",
                static_cast<double>(committed_tokens),
                "maintenance",
                device,
                {{"blocking", "false"},
                 {"authority_execution", "host_coordinated"},
                 {"scheduler", "host_background_service"}});
            return true;
        }

        /* Static and observe-only authorities intentionally own no maintenance
         * worker. Their setup-time immobility evidence is sufficient and a
         * committed inference boundary has no placement work to perform. */
        return true;
    }

} // namespace llaminar2
