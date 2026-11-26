/**
 * @file OrchestrationRunnerRuntime.cpp
 * @brief OrchestrationRunner state accessors, diagnostics/snapshots, sampling helpers, and MPI worker loop.
 */

#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{
    // =========================================================================
    // Configuration
    // =========================================================================

    const RankExecutionPlan &OrchestrationRunner::executionPlan() const
    {
        return plan_;
    }

    const OrchestrationConfig &OrchestrationRunner::config() const
    {
        return config_;
    }

    const MTPRuntimeConfig &OrchestrationRunner::retainedMTPConfig() const noexcept
    {
        /* A disabled initial request may still reserve the complete MTP graph
         * envelope. Prefer that plan-owned physical contract whenever it
         * carries capacity; only pre-plan/test runners fall back to config_. */
        if (retainsMTPGraphCapacity(plan_.runtime.mtp))
            return plan_.runtime.mtp;
        return config_.mtp;
    }

    MTPRuntimeConfig OrchestrationRunner::activeMTPRequestConfig() const
    {
        const MTPRuntimeConfig &retained = retainedMTPConfig();
        auto active = resolveMTPSamplingRequestConfig(
            retained,
            mtp_request_policy_.value_or(
                makeMTPRequestPolicy(retained)),
            active_sampling_params_);
        // Both host and device-controller admission consume this same context.
        // Previously only the host controller received backend/model identity,
        // leaving the GPU ABI on a different, untrained initial-depth policy.
        if (runner_)
        {
            active = resolveMTPRequestDepthPolicyContext(
                std::move(active), runner_->primaryDeviceId(), model_ctx_.get());
            active.depth_policy.scope = mtpDepthPolicyScope();
        }
        return active;
    }

    MTPDepthPolicyScope OrchestrationRunner::mtpDepthPolicyScope() const
    {
        auto scope = mtp_depth_policy_scope_geometry_;
        if (!scope.specified() || !runner_)
            return {};
        const auto request = mtp_request_policy_.value_or(makeMTPRequestPolicy(retainedMTPConfig()));
        const auto topology = request.depth_policy.mode == MTPDepthPolicyMode::Dynamic
            ? DeviceGenerationLoopTopology::DynamicDepth : DeviceGenerationLoopTopology::FixedDepth;
        const auto policy = runner_->deviceGenerationExecutionPolicy(topology);
        if (policy == DeviceGenerationExecutionPolicy::Unsupported && runner_->primaryDeviceId().is_gpu())
            return {};
        const auto role = runner_->primaryDeviceId().is_cpu() ? MTPVerifierExecutionRole::Standalone
            : mtpVerifierExecutionRole(policy);
        scope.verifier_width_policy = resolveMTPVerifierPhysicalWidthPolicy(role, request.depth_policy.mode);
        return scope;
    }

    const char *OrchestrationRunner::mtpDepthPolicyScopeGenericReason() const
    {
        if (mtp_depth_policy_scope_geometry_.specified() && !mtpDepthPolicyScope().specified())
            return "unsupported_generation_policy";
        return mtp_depth_policy_scope_generic_reason_;
    }

    bool OrchestrationRunner::mtpRequestPolicyTransitionIsIdle() const noexcept
    {
        /* Every listed value is request-owned. The retained graph family,
         * prefix archive, prepared weights, and ExpertOverlay placement epoch
         * intentionally do not participate in this admission decision. */
        return initialization_lifecycle_.readyForInference() && runner_ && runner_->get_position() == 0 &&
               !prefill_logits_ready_ && !ready_mtp_condition_ &&
               !pending_mtp_condition_token_ &&
               !pending_mtp_condition_params_ &&
               !pending_mtp_condition_resident_state_ &&
               !prelaunched_mtp_first_sidecar_resident_state_ &&
               !prelaunched_mtp_first_sidecar_params_ &&
               batched_request_states_.empty();
    }

    bool OrchestrationRunner::configureMTPRequestPolicy(
        const MTPRequestPolicy &policy)
    {
        if (!mtpRequestPolicyTransitionIsIdle())
        {
            return setError(
                "MTP request policy may change only on an initialized, reset runner with no live request state");
        }
        if (const auto error =
                validateMTPRequestPolicy(policy, retainedMTPConfig()))
        {
            return setError(*error);
        }

        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->world_size() > 1 &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_)
        {
            /* Policy selection is a request-admission command, not mutable
             * model state. Publish the complete value before changing the
             * root so a failed wire transfer cannot leave ranks selecting
             * different retained transactions. */
            broadcastCommand(MPICommand::SET_MTP_REQUEST_POLICY);
            MTPRequestPolicyWire wire = encodeMTPRequestPolicy(policy);
            mpi_ctx_->broadcast_int32(
                wire.data(), wire.size(), mpi_coordinated_root_rank_);
        }

        mtp_request_policy_ = policy;
        /* The depth controller and counters are request-owned derivatives of
         * this policy. Rebuild the controller lazily on the next decode; graph
         * and weight owners remain untouched. */
        mtp_depth_controller_.reset();
        mtp_stats_ = {};
        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        PerfStatsCollector::addCounter(
            "mtp",
            "request_policy_publications",
            1.0,
            "request_admission",
            runner_->primaryDeviceId().toString(),
            {{"enabled", policy.enabled ? "true" : "false"},
             {"depth_policy",
              mtpDepthPolicyModeToString(policy.depth_policy.mode)},
             {"maximum_execution_depth",
              std::to_string(
                  policy.enabled
                      ? (policy.depth_policy.mode == MTPDepthPolicyMode::Fixed
                             ? policy.draft_tokens
                             : (policy.depth_policy.max_depth > 0
                                    ? policy.depth_policy.max_depth
                                    : defaultMTPAdaptiveMaximumDraftDepth()))
                      : 0)}});
        return true;
    }

    MTPRequestPolicy OrchestrationRunner::mtpRequestPolicy() const
    {
        return mtp_request_policy_.value_or(
            makeMTPRequestPolicy(retainedMTPConfig()));
    }

    // =========================================================================
    // Status
    // =========================================================================

    bool OrchestrationRunner::isInitialized() const
    {
        return initialization_lifecycle_.readyForInference();
    }

    const std::string &OrchestrationRunner::lastError() const
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        return last_error_;
    }

    int OrchestrationRunner::vocabSize() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->vocab_size();
    }

    int OrchestrationRunner::currentPosition() const
    {
        if (!runner_)
        {
            return 0;
        }
        return runner_->get_position();
    }

    void OrchestrationRunner::ensureFiniteDevicePlacementArmed()
    {
        if(finite_device_placement_submission_!=FiniteDevicePlacementSubmission::PendingArm)return;
        if(!moe_overlay_device_controller_graph_service_)
            throw std::logic_error("Device finite placement lost its original controller owner");
        // Public clearCache has already retired the previous source producer.
        // Every original owner Arm terminal precedes any new source or follower.
        try {
            moe_overlay_device_controller_graph_service_->runFinitePlacementBoundary(
                DeviceMoEOverlayEpochParentAction::Arm);
        } catch(const std::exception &error) {
            // Some original owners may already be Armed. Without the failed
            // owner's native proof no host flag may certify rollback/reuse.
            LOG_ERROR("Device finite placement Arm lost original native proof: " << error.what());
            std::terminate();
        } catch(...) {
            LOG_ERROR("Device finite placement Arm lost original native proof through a nonstandard exception");
            std::terminate();
        }
        finite_device_placement_submission_=FiniteDevicePlacementSubmission::Armed;
    }

    void OrchestrationRunner::withMoEOverlayDevicePlacementScope(const std::function<void()> &body)
    {
        if(!body || !initialization_lifecycle_.readyForInference() || !runner_ || !model_ctx_ ||
           !physical_memory_authority_ || !moe_overlay_device_controller_graph_service_ ||
           runner_->moeOverlayAuthorityExecution()!=MoEOverlayAuthorityExecutionKind::DeviceResident ||
           !config_.moe_routed_expert_plan ||
           MoEExpertOwnerMap::usesCPUJointFragments(*config_.moe_routed_expert_plan) ||
           config_.moe_rebalance.mode!=MoERebalanceRuntimeMode::Dynamic ||
           (mpi_ctx_ && mpi_ctx_->world_size()!=1) ||
           finite_device_placement_submission_!=FiniteDevicePlacementSubmission::Inactive)
            throw std::logic_error("Finite device placement requires its ready original rank-local device authority");
        finite_device_placement_submission_=FiniteDevicePlacementSubmission::PendingArm;
        std::exception_ptr failure;
        try { body(); } catch(...) { failure=std::current_exception(); }
        if(finite_device_placement_submission_==FiniteDevicePlacementSubmission::Armed) {
            try {
                // The device rejects End if any actual child still owns a bank.
                // Missing native proof is fatal, never a successful scope close.
                moe_overlay_device_controller_graph_service_->runFinitePlacementBoundary(
                    DeviceMoEOverlayEpochParentAction::Close);
            } catch(const std::exception &error) {
                if(failure) {
                    try {std::rethrow_exception(failure);}
                    catch(const std::exception &original) {
                        LOG_ERROR("Original finite device request failed: " << original.what());
                    } catch(...) {
                        LOG_ERROR("Original finite device request failed with a nonstandard exception");
                    }
                }
                LOG_ERROR("Device finite placement cannot retire its original parent: " << error.what());
                std::terminate();
            } catch(...) {
                if(failure) {
                    try {std::rethrow_exception(failure);}
                    catch(const std::exception &original) {
                        LOG_ERROR("Original finite device request failed: " << original.what());
                    } catch(...) {
                        LOG_ERROR("Original finite device request failed with a nonstandard exception");
                    }
                }
                LOG_ERROR("Device finite placement closure lost original native proof through a nonstandard exception");
                std::terminate();
            }
        } else if(!failure) {
            failure=std::make_exception_ptr(std::logic_error(
                "Finite device placement callback executed no original request"));
        }
        finite_device_placement_submission_=FiniteDevicePlacementSubmission::Inactive;
        if(failure)std::rethrow_exception(failure);
    }

    MoEOverlayInferencePlacementScopeObservation
    OrchestrationRunner::withMoEOverlayInferencePlacementScope(const std::function<void()> &body)
    {
        if (!initialization_lifecycle_.readyForInference() || !runner_ || !model_ctx_ ||
            !physical_memory_authority_ || !moe_expert_overlay_residency_authority_ ||
            !moe_overlay_inference_transaction_coordinator_ ||
            !config_.moe_routed_expert_plan ||
            !MoEExpertOwnerMap::usesCPUJointFragments(*config_.moe_routed_expert_plan) ||
            (mpi_ctx_ && mpi_ctx_->rank() != mpi_coordinated_root_rank_))
            throw std::logic_error("Finite placement scope requires its ready original host-coordinated model/source/PMA root");
        // Retain the original coordinator (and its original shared authority)
        // across synchronous public reset/request calls. No replacement runner,
        // source, physical byte authority, or follower admission is created.
        const auto coordinator = moe_overlay_inference_transaction_coordinator_;
        return coordinator->withMoEOverlayInferencePlacementScope(body);
    }

    void OrchestrationRunner::clearCache()
    {
        /*
         * SHUTDOWN is a terminal command-channel transition on every rank.
         * A request reset after that transition cannot be coordinated because
         * followers have deliberately left their receive loops. Reject it
         * before either the root publishes another command or a follower
         * advances process-local state against an absent authority.
         */
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->world_size() > 1 &&
            mpi_worker_command_lifecycle_ ==
                MPIWorkerCommandLifecycle::ShutdownPublished)
        {
            LLAMINAR_UNREACHABLE(
                "Cannot clear request state after the MPI worker command channel "
                "has closed; construct a new OrchestrationRunner for the next "
                "serving session");
        }

        // Request-boundary reset: broadcast to worker ranks so they clear
        // KV/recurrent state in lockstep while preserving reusable graph caches.
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
            broadcastCommand(MPICommand::CLEAR_CACHE);

        resetUnderlyingRunnerRequestState("request-clear-cache");
        if (!advanceMoEOverlayCollectiveRequestGeneration("request-clear-cache"))
        {
            throw std::runtime_error(
                "clearCache() could not publish a fresh graph-native MoE "
                "overlay collective generation");
        }
        retireMTPRequestContinuationState();
        device_generation_admission_.reset();
        generation_request_seeds_.reset();
        device_generation_terminal_ledger_authoritative_ = false;
        device_generation_embedded_moe_maintenance_pending_ack_ = false;
        hosted_device_generation_decode_progress_retired_.store(
            0u, std::memory_order_release);
        clearBatchedDecodeState();
        sampler_ = Sampler(active_sampling_params_.seed);
        mtp_bypassed_ = false;
        mtp_bypass_recorded_for_request_ = false;
        mtp_bypass_reason_.clear();
        /*
         * clearCache() is the request boundary used by benchmark iterations
         * and server sessions. Keep adaptive-depth state request-scoped so the
         * reported counters and current depth describe the same request.
         */
        if (mtp_depth_controller_)
        {
            mtp_depth_controller_->reset();
        }
        mtp_stats_ = {};
    }

    bool OrchestrationRunner::purgePrefixCache()
    {
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->world_size() > 1 &&
            mpi_worker_command_lifecycle_ ==
                MPIWorkerCommandLifecycle::ShutdownPublished)
        {
            LLAMINAR_UNREACHABLE(
                "Cannot purge reusable prefix state after the MPI worker command "
                "channel has closed");
        }

        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::PURGE_PREFIX_CACHE);
        }

        if (!runner_)
            return true;
        if (!runner_->purgePrefixCache())
        {
            last_error_ =
                "Reusable prefix-cache purge was rejected by an active cache lease";
            return false;
        }
        return true;
    }

    void OrchestrationRunner::drainCompletedDecodeBoundaryMaintenanceDiagnostics()
    {
        if (runner_)
            runner_->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
    }

    RequestRuntimeSummary OrchestrationRunner::requestRuntimeSummary() const
    {
        // Every input is already owned by this request authority or surfaced
        // in its validated terminal result. Never walk child/device state here.
        RequestRuntimeSummary summary;
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        summary.mtp_request.current_depth =
            device_generation_terminal_ledger_authoritative_
                ? mtp_stats_.current_depth
                : (mtp_depth_controller_
                       ? mtp_depth_controller_->currentDepth()
                       : std::max(0, mtp.draft_tokens));
        summary.mtp_request.min_depth =
            mtp_depth_controller_ ? mtp_depth_controller_->minDepth()
                                  : std::max(0, mtp.draft_tokens);
        summary.mtp_request.max_depth =
            mtp_depth_controller_ ? mtp_depth_controller_->maxDepth()
                                  : std::max(0, mtp.draft_tokens);
        summary.prefix_request = prefix_request_summary_;
        summary.mtp_request.enabled = mtp.enabled;
        summary.mtp_request.bypassed = mtp_bypassed_;
        summary.mtp_request.bypass_reason = mtp_bypass_reason_;
        summary.mtp_request.verify_mode = mtpVerifyModeToString(mtp.verify_mode);
        summary.mtp_request.stochastic_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling;
        summary.mtp_request.adaptive_depth_enabled =
            mtp.depth_policy.mode != MTPDepthPolicyMode::Fixed;
        summary.mtp_request.depth_policy_mode =
            mtpDepthPolicyModeToString(mtp.depth_policy.mode);
        summary.mtp_request.depth_policy_updates = mtp_stats_.depth_policy_updates;
        if (device_generation_terminal_ledger_authoritative_)
        {
            /*
             * Native generation owns every adaptive transition after admission.
             * The terminal ledger currently exports aggregate transition counts,
             * not a host-readable per-window decision enum.  Reporting the
             * dormant host controller's reason beside the device-selected depth
             * would manufacture a mixed-authority diagnostic.
             */
            summary.mtp_request.last_depth_policy_reason =
                "device_terminal_ledger";
        }
        else if (mtp_depth_controller_)
        {
            summary.mtp_request.last_depth_policy_reason =
                toString(mtp_depth_controller_->lastDecision().reason);
        }
        summary.mtp_request.draft_steps = mtp_stats_.draft_steps;
        summary.mtp_request.accepted_tokens = mtp_stats_.accepted_tokens;
        summary.mtp_request.rejected_tokens = mtp_stats_.rejected_tokens;
        summary.mtp_request.rollbacks = mtp_stats_.rollbacks;
        // Rejections count transactions, not all unaccepted draft suffixes.
        // The attempted-draft ledger is the token denominator at every depth.
        summary.mtp_request.acceptance_rate =
            mtp_stats_.draft_steps > 0
                ? static_cast<double>(mtp_stats_.accepted_tokens) / static_cast<double>(mtp_stats_.draft_steps)
                : 0.0;
        summary.mtp_request.stochastic_accept_tests = mtp_stats_.stochastic_accept_tests;
        summary.mtp_request.stochastic_accepts = mtp_stats_.stochastic_accepts;
        summary.mtp_request.stochastic_residual_samples =
            mtp_stats_.stochastic_residual_samples;
        summary.mtp_request.stochastic_terminal_samples =
            mtp_stats_.stochastic_terminal_samples;
        summary.mtp_request.stochastic_acceptance_rate =
            mtp_stats_.stochastic_accept_tests > 0
                ? static_cast<double>(mtp_stats_.stochastic_accepts) /
                      static_cast<double>(mtp_stats_.stochastic_accept_tests)
                : 0.0;
        summary.mtp_verifier_runs = mtp_stats_.verifier_runs;
        summary.mtp_verifier_token_count = mtp_stats_.verifier_token_count;
        return summary;
    }

    PrefixRuntimeStateSnapshot OrchestrationRunner::prefixStateProbe(
        const PrefixProbeCapturePolicy &capture_policy) const
    {
        PrefixRuntimeStateSnapshot snapshot = runner_ ? runner_->prefixStateProbe(capture_policy)
                                                      : PrefixRuntimeStateSnapshot{};
        snapshot.initialized = initialization_lifecycle_.readyForInference();
        snapshot.prefill_logits_ready = prefill_logits_ready_;
        if (capture_policy.capture_device_logical_state && snapshot.resident_condition &&
            ready_mtp_condition_ && ready_mtp_condition_->isDeviceResidentOnly())
        {
            const auto current = runner_->deviceResidentLogicalSequenceState();
            const auto &observed = *snapshot.resident_condition;
            if (!ready_mtp_condition_->valid() ||
                !samplingParamsEqual(ready_mtp_condition_->sampling_params, active_sampling_params_) ||
                !ready_mtp_condition_->resident_state->sameMailboxAs(current) ||
                observed.device != current.device ||
                observed.publication_generation != current.publication_generation ||
                observed.live_state_epoch != current.live_state_epoch)
                throw std::logic_error("Continuation diagnostics rejected a stale or foreign original ready condition");
            snapshot.resident_condition->response_disposition = ready_mtp_condition_->response_disposition;
        }
        /*
         * The heterogeneous controller's mapped durable word is the only
         * movement authority for this regime. The narrow epoch accessor also
         * preserves the child runner's device-local movement evidence.
         */
        if (snapshot.moe_runtime_movement_epoch.isLeaf())
            snapshot.moe_runtime_movement_epoch = PrefixMovementEpochSnapshot::leaf(std::max(
                snapshot.moe_runtime_movement_epoch.epoch(), moeRuntimeMovementEpoch()));
        /*
         * The child probe owns the meaning of live position. GPU runners read
         * it from the device-resident logical-state mailbox, while
         * get_position() is only a host planning mirror and is intentionally
         * allowed to lag accepted grouped publication. Overwriting the child
         * result here made a preserved sidecar appear to move main state
         * backwards and, more seriously, taught diagnostics to trust stale
         * host state over the device owner.
         */
        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        snapshot.mtp_config_enabled = mtp.enabled;
        snapshot.mtp_bypassed = mtp_bypassed_;
        snapshot.mtp_bypass_reason = mtp_bypass_reason_;
        snapshot.mtp_draft_steps = mtp_stats_.draft_steps;
        snapshot.mtp_accepted_tokens = mtp_stats_.accepted_tokens;
        snapshot.mtp_rejected_tokens = mtp_stats_.rejected_tokens;
        snapshot.mtp_rollbacks = mtp_stats_.rollbacks;
        snapshot.mtp_bypasses = mtp_stats_.bypasses;
        snapshot.mtp_verifier_runs = mtp_stats_.verifier_runs;
        snapshot.mtp_verifier_token_count = mtp_stats_.verifier_token_count;
        snapshot.mtp_last_transaction_draft_depth =
            mtp_stats_.last_transaction_draft_depth;
        snapshot.mtp_last_transaction_emitted_token_count =
            mtp_stats_.last_transaction_emitted_token_count;
        if (runner_)
        {
            /*
             * Keep the same ownership split used by transaction planning.
             * The GPU scalar is scheduler metadata advanced from validated
             * compact publication; it is not a shadow of device KV, GDN, or
             * token state. CPU execution keeps its live position on host.
             */
            snapshot.mtp_next_condition_position =
                runner_->primaryDeviceId().is_gpu()
                    ? decode_transaction_planning_position_.value_or(-1)
                    : runner_->get_position();
        }
        snapshot.mtp_stochastic_accept_tests = mtp_stats_.stochastic_accept_tests;
        snapshot.mtp_stochastic_accepts = mtp_stats_.stochastic_accepts;
        snapshot.mtp_stochastic_residual_samples = mtp_stats_.stochastic_residual_samples;
        snapshot.mtp_stochastic_terminal_samples = mtp_stats_.stochastic_terminal_samples;
        snapshot.mtp_transaction_commits = mtp_stats_.transaction_commits;
        snapshot.mtp_transaction_rollbacks = mtp_stats_.transaction_rollbacks;
        snapshot.mtp_transaction_validation_failures =
            mtp_stats_.transaction_validation_failures;
        snapshot.mtp_unsafe_verifier_state_rejections =
            mtp_stats_.unsafe_verifier_state_rejections;
        snapshot.mtp_depth_policy_windows = mtp_stats_.depth_policy_windows;
        snapshot.mtp_depth_policy_updates = mtp_stats_.depth_policy_updates;
        snapshot.mtp_depth_policy_promotions = mtp_stats_.depth_policy_promotions;
        snapshot.mtp_depth_policy_demotions = mtp_stats_.depth_policy_demotions;
        snapshot.mtp_depth_policy_utility_returns = mtp_stats_.depth_policy_utility_returns;
        snapshot.mtp_depth_policy_utility_keeps = mtp_stats_.depth_policy_utility_keeps;
        snapshot.mtp_depth_policy_utility_probes = mtp_stats_.depth_policy_utility_probes;
        snapshot.mtp_depth_policy_observe_recommendations =
            mtp_stats_.depth_policy_observe_recommendations;
        const auto request_summary = requestRuntimeSummary();
        snapshot.mtp_current_depth = request_summary.mtp_request.current_depth;
        snapshot.mtp_min_depth = request_summary.mtp_request.min_depth;
        snapshot.mtp_max_depth = request_summary.mtp_request.max_depth;
        snapshot.prefill_chunk_schedules = prefill_chunk_stats_.schedules;
        snapshot.prefill_chunk_successful_schedules = prefill_chunk_stats_.successful_schedules;
        snapshot.prefill_chunks = prefill_chunk_stats_.chunks;
        snapshot.prefill_chunk_real_tokens = prefill_chunk_stats_.real_tokens;
        snapshot.prefill_chunk_padded_tokens = prefill_chunk_stats_.padded_tokens;
        snapshot.prefill_chunk_failures = prefill_chunk_stats_.failures;
        snapshot.prefix_request = request_summary.prefix_request;
        snapshot.mtp_request = request_summary.mtp_request;
        if (snapshot.architecture.empty() && model_ctx_)
        {
            snapshot.architecture = model_ctx_->architecture();
        }
        return snapshot;
    }

    DeviceId OrchestrationRunner::primaryDeviceId() const
    {
        return runner_ ? runner_->primaryDeviceId() : DeviceId::cpu();
    }

    // =========================================================================
    // Advanced
    // =========================================================================

    const float *OrchestrationRunner::lastLogits() const
    {
        if (!runner_)
        {
            return nullptr;
        }
        PerfStatsCollector::addCounter(
            "sampling",
            "host_logits_access",
            1.0,
            {},
            runner_->primaryDeviceId().toString(),
            {{"source", "orchestration_runner_last_logits"}});
        return runner_->logits();
    }

    void OrchestrationRunner::setStopTokens(const std::vector<int32_t> &stop_tokens)
    {
        if (stop_tokens.size() >
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        {
            throw std::invalid_argument(
                "Request stop-token policy exceeds the MPI protocol's int32 count range");
        }

        /*
         * Stop recognition changes whether an MTP transaction launches another
         * sidecar. Install the policy locally before publishing it: a rejected
         * root policy must never release workers into a request with a different
         * termination contract. Once accepted, the command and its two payloads
         * form one ordered worker-loop transaction ahead of PREFILL/DECODE_STEP.
         */
        if (runner_ &&
            !runner_->configureMTPRequestStopTokens(stop_tokens))
        {
            throw std::runtime_error(
                "Inference runner rejected request stop-token policy");
        }
        stop_tokens_ = stop_tokens;

        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_STOP_TOKENS);
            int32_t stop_token_count = static_cast<int32_t>(stop_tokens_.size());
            mpi_ctx_->broadcast_int32(
                &stop_token_count, 1, mpi_coordinated_root_rank_);
            if (stop_token_count > 0)
            {
                mpi_ctx_->broadcast_int32(
                    stop_tokens_.data(),
                    static_cast<size_t>(stop_token_count),
                    mpi_coordinated_root_rank_);
            }
        }
    }

    bool OrchestrationRunner::hasLocalTP() const
    {
        return plan_.local_tp_devices.size() > 1;
    }

    std::shared_ptr<ITokenizer> OrchestrationRunner::tokenizer() const
    {
        return tokenizer_;
    }

    const std::string &OrchestrationRunner::architecture() const
    {
        static const std::string kEmpty;
        return model_ctx_ ? model_ctx_->architecture() : kEmpty;
    }

    const IModelContext *OrchestrationRunner::modelContextForDiagnostics() const
    {
        return model_ctx_.get();
    }

    std::shared_ptr<const MoEOverlayResidencySnapshot>
    OrchestrationRunner::expertOverlayResidencySnapshotForDiagnostics() const
    {
        return moe_expert_overlay_residency_authority_
                   ? moe_expert_overlay_residency_authority_->snapshot()
                   : nullptr;
    }

    std::optional<ModelContextReuseContract>
    OrchestrationRunner::modelContextReuseContract() const
    {
        /*
         * A reuse contract is proof emitted by a successfully initialized
         * production runner, not a generic ModelContext accessor. One ordinary
         * OrchestrationRunner owns one ModelContext per MPI rank; that model-owned
         * PreparedWeightStore may contain an exact LocalTP/GlobalTP/PP or
         * ExpertOverlay set of device handles. Cross-rank collectives do not
         * change that cardinality. The runner factory's typed authority scope is
         * the sole source of truth for configurations that instead compose a
         * multi-rank context set.
         */
        const RunnerModelAuthorityScope authority_scope =
            resolveRunnerModelAuthorityScope(config_);
        if (!initialization_lifecycle_.readyForInference() || !model_ctx_ ||
            authority_scope != RunnerModelAuthorityScope::RankLocal)
        {
            LOG_DEBUG(
                "[ModelContextReuseContract] unavailable: initialized="
                << initialization_lifecycle_.readyForInference()
                << " model=" << static_cast<bool>(model_ctx_)
                << " authority_scope="
                << (authority_scope == RunnerModelAuthorityScope::RankLocal
                        ? "rank_local"
                        : "multi_rank_set"));
            return std::nullopt;
        }
        if ((config_.moe_routed_expert_plan || moe_pipeline_topology_) &&
            requested_routed_weight_authority_identity_.empty())
        {
            LOG_DEBUG(
                "[ModelContextReuseContract] unavailable: routed plan lacks "
                "a requested weight-authority identity");
            return std::nullopt;
        }
        const auto weight_manager = model_ctx_->concreteWeightManager();
        if (!weight_manager)
            return std::nullopt;
        const auto &gates = weight_manager->lifecycleGates();
        const auto store = weight_manager->preparedWeightStoreIfInitialized();
        const auto prepared_devices = rankLocalPreparedWeightAuthorityDevices(
            plan_,
            config_.moe_routed_expert_plan,
            moe_expert_overlay_mpi_ctx_
                ? moe_expert_overlay_mpi_ctx_
                : mpi_ctx_);
        if (!gates.device_preparation_complete ||
            !gates.graph_materialization_complete || !store ||
            prepared_devices.empty())
        {
            LOG_DEBUG(
                "[ModelContextReuseContract] unavailable: device_preparation="
                << gates.device_preparation_complete
                << " graph_materialization="
                << gates.graph_materialization_complete
                << " store=" << static_cast<bool>(store)
                << " devices=" << prepared_devices.size());
            return std::nullopt;
        }
        for (const DeviceId device : prepared_devices)
        {
            if (weight_manager->preparedRecordCountForDevice(device) == 0u)
            {
                LOG_DEBUG(
                    "[ModelContextReuseContract] unavailable: no prepared "
                    "records for "
                    << device.toString());
                return std::nullopt;
            }
        }

        /*
         * Sharing extends only model-owned lifetime. All mutable execution state
         * remains below runner_; shutdown retires it before the consumer starts.
         * The copied plan is the certificate the consumer must match.
         */
        std::shared_ptr<ModelContextReuseAuthority> reuse_authority;
        {
            std::lock_guard<std::mutex> lock(model_context_reuse_mutex_);
            if (!model_context_reuse_authority_)
            {
                model_context_reuse_authority_ =
                    std::make_shared<ModelContextReuseAuthority>();
            }
            reuse_authority = model_context_reuse_authority_;
        }
        if (reuse_authority->state() !=
            ModelContextReuseAuthority::State::RunnerExclusive)
        {
            LOG_ERROR(
                "[ModelContextReuseContract] exporter does not exclusively own the prepared model context");
            return std::nullopt;
        }

        std::shared_ptr<const MoEOverlaySealedMigrationMeasurements>
            migration_profile =
                retained_expert_overlay_migration_profile_;
        std::string migration_profile_identity =
            retained_expert_overlay_migration_profile_identity_;
        if (moe_expert_overlay_economy_calibration_)
        {
            migration_profile.reset();
            migration_profile_identity.clear();
            if (moe_expert_overlay_economy_calibration_->state() ==
                MoEOverlayEconomyCalibrationState::Complete)
            {
                const auto *sealed =
                    moe_expert_overlay_economy_calibration_
                        ->sealedMeasurements();
                if (sealed && sealed->valid() &&
                    !moe_expert_overlay_migration_profile_identity_.empty())
                {
                    migration_profile = std::make_shared<
                        const MoEOverlaySealedMigrationMeasurements>(*sealed);
                    migration_profile_identity =
                        moe_expert_overlay_migration_profile_identity_;
                }
            }
        }
        if (static_cast<bool>(migration_profile) !=
            !migration_profile_identity.empty())
        {
            LOG_ERROR(
                "[ModelContextReuseContract] migration profile and exact identity presence disagree");
            return std::nullopt;
        }
        const auto installed_memory_authority =
            weight_manager->physicalMemoryAuthority();
        if (!physical_memory_authority_ || !physical_memory_admission_ ||
            physical_memory_authority_->admission().get() !=
                physical_memory_admission_.get() ||
            !installed_memory_authority ||
            installed_memory_authority.get() !=
                physical_memory_authority_.get())
        {
            LOG_ERROR(
                "[ModelContextReuseContract] model and runner do not share one physical-memory authority");
            return std::nullopt;
        }

        return ModelContextReuseContract{
            .context = model_ctx_,
            .reuse_authority = std::move(reuse_authority),
            .reusable_execution_workspaces =
                reusable_execution_workspaces_,
            .physical_memory_authority =
                physical_memory_authority_,
            .prepared_weight_plan = plan_,
            .prepared_routed_weight_plan =
                config_.moe_routed_expert_plan
                    ? std::make_shared<const MoERoutedExpertPlacementPlan>(
                          *config_.moe_routed_expert_plan)
                    : nullptr,
            .expert_overlay_memory_admission =
                moe_expert_overlay_memory_admission_,
            .pipeline_prepared_plan = moe_pipeline_prepared_plan_,
            .routed_weight_authority_identity =
                requested_routed_weight_authority_identity_,
            .expert_overlay_migration_profile =
                std::move(migration_profile),
            .expert_overlay_migration_profile_identity =
                std::move(migration_profile_identity),
        };
    }

    ModelContextReuseStatus
    OrchestrationRunner::modelContextReuseStatus() const
    {
        /*
         * Every field is owned by the runner's typed admission lifecycle. No
         * telemetry snapshot, timing heuristic, or caller cache-hit claim is
         * consulted. The shared authority generation advances only after a
         * completed Sealing -> Reusable -> RunnerExclusive handoff.
         */
        return {
            .imported_contract =
                imported_model_context_reuse_authority_,
            .prepared_weight_plan_validated =
                retained_prepared_weight_plan_validated_,
            .prepared_entry_count = retained_prepared_entry_count_,
            .authority_generation =
                imported_model_context_reuse_authority_ &&
                        model_context_reuse_authority_
                    ? model_context_reuse_authority_->generation()
                    : 0u,
        };
    }

    // =========================================================================
    // Error Handling
    // =========================================================================

    bool OrchestrationRunner::setError(const std::string &error)
    {
        std::lock_guard<std::mutex> lock(error_mutex_);
        last_error_ = error;
        LOG_ERROR(error);
        return false;
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void OrchestrationRunner::enableSnapshotCapture(const std::string &output_dir)
    {
        snapshot_combined_cache_.clear();
        snapshot_capture_setup_.mode = SnapshotCaptureSetup::Mode::Enabled;
        snapshot_capture_setup_.output_directory = output_dir;
        if (runner_)
        {
            /* Keep the delegated runner aligned with the one retained policy. */
            runner_->setSnapshotCaptureFilter(snapshot_capture_setup_.filter);
            runner_->enableSnapshotCapture(output_dir);
        }
    }

    void OrchestrationRunner::setSnapshotCaptureFilter(const std::vector<std::string> &keys)
    {
        snapshot_capture_setup_.filter = keys;
        if (runner_)
        {
            runner_->setSnapshotCaptureFilter(keys);
        }
    }

    bool OrchestrationRunner::setOrdinaryTerminalProjectionCapacity(
        OrdinaryTerminalProjectionCapacity capacity)
    {
        // A precompiled plan is already immutable; its caller supplies demand
        // through RuntimeConfig before construction, never by editing children.
        if (plan_built_ || initialization_lifecycle_.admissionComplete() || runner_)
            return setError("Ordinary terminal projection capacity must be declared before plan compilation and model admission");
        ordinary_terminal_projection_setup_ = capacity;
        return true;
    }

    bool OrchestrationRunner::setSnapshotMemoryCapacity(
        GraphSnapshotMemoryCapacity capacity)
    {
        if (initialization_lifecycle_.admissionComplete() || runner_)
        {
            return setError(
                "Graph snapshot memory capacity must be declared before runner initialization");
        }
        if (!capacity.valid())
        {
            return setError(
                "Graph snapshot memory capacity must contain positive per-accelerator bytes");
        }
        // Freeze optional diagnostic topology at the same setup boundary as
        // its reference inventory. MemoryPlanner prices cache banks from model
        // geometry; the parity caller must not guess those physical bytes.
        if (debugEnv().attention.debug_effective_kv_snapshot)
        {
            capacity.effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{
                .layer = debugEnv().attention.debug_effective_kv_snapshot_layer};
        }
        snapshot_capture_setup_.memory_capacity = capacity;
        return true;
    }

    bool OrchestrationRunner::prepareInactiveSnapshotCapture(
        const std::string &output_dir)
    {
        if (initialization_lifecycle_.admissionComplete() || runner_)
        {
            return setError(
                "Prepared-inactive snapshot topology must be declared before runner initialization");
        }
        snapshot_combined_cache_.clear();
        snapshot_capture_setup_.mode =
            SnapshotCaptureSetup::Mode::PreparedInactive;
        snapshot_capture_setup_.output_directory = output_dir;
        prepared_inactive_snapshot_capture_ready_ = false;
        return true;
    }

    bool OrchestrationRunner::activatePreparedSnapshotCapture()
    {
        if (!initialization_lifecycle_.readyForInference() || !runner_ ||
            snapshot_capture_setup_.mode !=
                SnapshotCaptureSetup::Mode::PreparedInactive ||
            !prepared_inactive_snapshot_capture_ready_)
        {
            return setError(
                "Snapshot activation requires a setup-certified prepared-inactive diagnostic family");
        }

        snapshot_combined_cache_.clear();
        runner_->setSnapshotCaptureFilter(snapshot_capture_setup_.filter);
        runner_->enableSnapshotCapture(
            snapshot_capture_setup_.output_directory);

        /*
         * This call is a readiness assertion, not a materialization request. Each
         * leaf must recognize the stable snapshot identity and return before any
         * coordinator, capture stream, or executable is touched.
         */
        if (!materializeCurrentServingGraphFamilyVariantWithoutLaunch())
            return false;

        snapshot_capture_setup_.mode = SnapshotCaptureSetup::Mode::Enabled;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "prepared_snapshot_family_activations",
            1.0,
            "diagnostic_transition",
            runner_->primaryDeviceId().toString(),
            {{"capture", "false"},
             {"instantiation", "false"},
             {"active_variant", "diagnostic"}});
        return true;
    }

    void OrchestrationRunner::disableSnapshotCapture()
    {
        snapshot_combined_cache_.clear();
        snapshot_capture_setup_.mode = SnapshotCaptureSetup::Mode::Disabled;
        snapshot_capture_setup_.output_directory.clear();
        prepared_inactive_snapshot_capture_ready_ = false;
        if (runner_)
        {
            runner_->disableSnapshotCapture();
        }
    }

    void OrchestrationRunner::clearSnapshots()
    {
        snapshot_combined_cache_.clear();
        if (runner_)
        {
            runner_->clearSnapshots();
        }
    }

    const float *OrchestrationRunner::getSnapshot(const std::string &key, size_t &out_size) const
    {
        if (runner_)
        {
            if (const auto *rank = dynamic_cast<const RankOrchestrator *>(runner_.get()))
            {
                TPSnapshot tp_snapshot = rank->getTPSnapshot(key);
                if (tp_snapshot.mode == SnapshotShardingMode::UNKNOWN &&
                    tp_snapshot.tp_degree > 1 &&
                    !tp_snapshot.device_data.empty())
                {
                    LOG_ERROR("[OrchestrationRunner] Refusing to return TP snapshot '" << key
                                                                                      << "' because its sharding mode is UNKNOWN. "
                                                                                         "Add the snapshot key to the model schema or a runtime "
                                                                                         "sharding override before comparing multi-device output.");
                    out_size = 0;
                    return nullptr;
                }

                size_t combined_size = 0;
                const float *combined = tp_snapshot.getCombinedData(combined_size);
                if (combined && combined_size > 0)
                {
                    auto &cache = snapshot_combined_cache_[key];
                    cache.assign(combined, combined + combined_size);
                    out_size = cache.size();
                    return cache.data();
                }
                if (tp_snapshot.tp_degree > 1 && !tp_snapshot.device_data.empty())
                {
                    std::ostringstream geometry;
                    for (const auto &participant : tp_snapshot.device_data)
                    {
                        if (geometry.tellp() > 0)
                            geometry << ',';
                        geometry << "{tp_index=" << participant.device_index
                                 << ",rows=" << participant.rows
                                 << ",cols=" << participant.cols
                                 << ",elements=" << participant.data.size()
                                 << '}';
                    }
                    LOG_ERROR("[OrchestrationRunner] Failed to combine TP snapshot '" << key
                                                                                    << "' with mode "
                                                                                    << shardingModeToString(tp_snapshot.mode)
                                                                                    << " (tp_degree=" << tp_snapshot.tp_degree
                                                                                    << ",published=" << tp_snapshot.device_data.size()
                                                                                    << ",geometry=[" << geometry.str()
                                                                                    << "]). Refusing to fall back to a single participant view."
                                                                                    << rank->describeFailedMirroredMTPState());
                    out_size = 0;
                    return nullptr;
                }
            }
            return runner_->getSnapshot(key, out_size);
        }
        out_size = 0;
        return nullptr;
    }

    SnapshotInfo OrchestrationRunner::getSnapshotWithShape(const std::string &key) const
    {
        return runner_ ? runner_->getSnapshotWithShape(key) : SnapshotInfo{};
    }

    std::vector<std::string> OrchestrationRunner::getSnapshotKeys() const
    {
        if (runner_)
        {
            return runner_->getSnapshotKeys();
        }
        return {};
    }

    // =========================================================================
    // Profiling
    // =========================================================================

    const GraphExecutorStats *OrchestrationRunner::executorStats() const
    {
        if (runner_)
        {
            return runner_->executorStats();
        }
        return nullptr;
    }

    void OrchestrationRunner::resetExecutorStats()
    {
        if (runner_)
        {
            runner_->resetExecutorStats();
        }
    }

    void OrchestrationRunner::resetUnderlyingRunnerRequestState(const char *reason)
    {
        if (!runner_)
            return;

        LOG_DEBUG("[OrchestrationRunner] Resetting underlying runner request-owned inference state"
                  << (reason && reason[0] ? std::string(" for ") + reason : std::string{}));
        runner_->resetInferenceState(
            InferenceStateResetRequest::requestBoundary(reason));
    }

    void OrchestrationRunner::retireMTPRequestContinuationState()
    {
        /*
         * Clear lease-bearing values before their host companions. Although no
         * caller can observe this synchronous transition halfway through, this
         * order makes the performance-critical backend event lifetime explicit
         * to readers and to custom shared-pointer deleters.
         */
        ready_mtp_condition_.reset();
        pending_mtp_condition_resident_state_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();

        pending_mtp_condition_token_.reset();
        pending_mtp_condition_params_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        prefill_logits_ready_ = false;
        decode_transaction_planning_position_.reset();
    }

    /**
     * @brief Publish the continuation-authoritative sparse generation for this request.
     *
     * Every ExpertOverlay graph, including a rank-local GPU/CPU topology, uses
     * one typed request identity. Cross-rank peers receive that identity at the
     * synchronized lifecycle boundary; rank-local participants receive it
     * directly from this runner. Independent graph capture/cache lifetimes may
     * never invent stage-local identities.
     */
    bool OrchestrationRunner::publishMoEOverlayCollectiveRequestGeneration(
        const char *reason)
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto request_authority = resolveMoEOverlayRequestAuthorityRank(
            config_.moe_routed_expert_plan,
            moe_pipeline_prepared_plan_,
            overlay_context);
        if (!request_authority)
        {
            return true;
        }
        if (!runner_)
        {
            return setError(
                "MoE ExpertOverlay has no runner for collective "
                "generation publication");
        }
        const int authority_rank = *request_authority;
        if (authority_rank < 0 || authority_rank >= overlay_context->world_size())
        {
            return setError(
                "MoE ExpertOverlay resolved an invalid continuation "
                "rank for collective-generation publication");
        }

        /*
         * The continuation rank is the sole authority for this value.  The
         * other ranks retain only the exact generation it publishes; they do
         * not advance a local counter and hope that independent request/cache
         * lifecycles happened to remain aligned.
         *
         * IMPIContext exposes a typed int32 broadcast.  Two words preserve the
         * complete uint64_t identity instead of truncating a long-running
         * serving process's request epoch.  Both ranks execute this only at a
         * synchronized initialization or request-reset boundary, never from
         * the captured inference hot path.
         */
        std::array<int32_t, 2> generation_wire{};
        if (overlay_context->rank() == authority_rank)
        {
            if (request_epoch_ == std::numeric_limits<uint64_t>::max())
            {
                return setError(
                    "Distributed MoE ExpertOverlay request-generation counter "
                    "overflowed");
            }

            const uint64_t authority_generation = request_epoch_ + 1;
            generation_wire[0] = static_cast<int32_t>(
                static_cast<uint32_t>(authority_generation));
            generation_wire[1] = static_cast<int32_t>(
                static_cast<uint32_t>(authority_generation >> 32));
        }
        if (overlay_context->world_size() > 1)
        {
            overlay_context->broadcast_int32(
                generation_wire.data(),
                generation_wire.size(),
                authority_rank);
        }
        const uint64_t generation_id =
            static_cast<uint64_t>(static_cast<uint32_t>(generation_wire[0])) |
            (static_cast<uint64_t>(static_cast<uint32_t>(generation_wire[1])) << 32);
        if (generation_id == 0)
        {
            return setError(
                "MoE ExpertOverlay continuation authority published "
                "an invalid zero collective generation");
        }
        // Keep a received mirror for lifecycle diagnostics only.  The source
        // of truth remains the continuation rank and is re-published at every
        // request boundary.
        request_epoch_ = generation_id - 1;
        moe_overlay_collective_generation_id_ = generation_id;
        if (!runner_->setMoEOverlayCollectiveRequestGeneration(generation_id))
        {
            return setError(
                "MoE ExpertOverlay runner rejected continuation-authoritative "
                "collective request generation");
        }

        PerfStatsCollector::recordOrderedSequenceStep(
            "forward_graph",
            "moe_overlay_collective_request_generation_sequence",
            {generation_id},
            "request_lifecycle",
            {},
            {{"authority",
              overlay_context->world_size() > 1
                  ? "continuation_root_rank"
                  : "rank_local_continuation_runner"},
             {"authority_rank", std::to_string(authority_rank)},
             {"immutable_until_request_reset", "true"}});
        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_collective_request_generation",
            1.0,
            "request_lifecycle",
            {},
            {{"authority",
              overlay_context->world_size() > 1
                  ? "continuation_root_rank"
                  : "rank_local_continuation_runner"},
             {"authority_rank", std::to_string(authority_rank)},
             {"reason", reason && reason[0] ? reason : "unspecified"},
             {"immutable_until_request_reset", "true"}});
        return true;
    }

    /**
     * @brief Advance the distributed request generation after a complete reset boundary.
     */
    bool OrchestrationRunner::advanceMoEOverlayCollectiveRequestGeneration(
        const char *reason)
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto request_authority = resolveMoEOverlayRequestAuthorityRank(
            config_.moe_routed_expert_plan,
            moe_pipeline_prepared_plan_,
            overlay_context);
        const bool distributed_overlay =
            request_authority && overlay_context &&
            overlay_context->world_size() > 1;
        const bool is_authority_rank =
            distributed_overlay &&
            overlay_context->rank() == *request_authority;

        if ((!distributed_overlay || is_authority_rank) &&
            request_epoch_ >= std::numeric_limits<uint64_t>::max() - 1)
        {
            return setError(
                "Distributed MoE ExpertOverlay request-generation counter "
                "overflowed before request reset");
        }

        if (!distributed_overlay || is_authority_rank)
        {
            // Only the continuation rank advances the source counter. Remote
            // ranks receive the resulting generation in publish... below.
            ++request_epoch_;
        }
        return publishMoEOverlayCollectiveRequestGeneration(reason);
    }

    int OrchestrationRunner::sampleGreedyOnDevice()
    {
        if (runner_)
        {
            return runner_->sampleGreedyOnDevice();
        }
        return -1;
    }

    int OrchestrationRunner::sampleOnDevice(const SamplingParams &params)
    {
        if (runner_)
        {
            return runner_->sampleOnDevice(params);
        }
        return -1;
    }

    bool OrchestrationRunner::waitForLastInferenceCompletionForBenchmark()
    {
        if (!runner_)
        {
            LOG_ERROR("[OrchestrationRunner] Benchmark transaction-completion boundary has no inference runner");
            return false;
        }
        return runner_->waitForLastInferenceCompletionForBenchmark();
    }

    InferenceMeasurementReadiness
    OrchestrationRunner::internalInferencePreparationReadiness() const
    {
        InferenceMeasurementReadiness overlay_readiness;
        InferenceMeasurementReadiness stage_readiness;
        for (const auto &binding : moe_pipeline_stage_bindings_)
            if (const auto &owner = binding->owners().host_maintenance; owner)
            {
                if (!owner->service()) return {.state = InferenceMeasurementReadinessState::Failed,
                    .owner = "cpu_pipeline_host_maintenance", .phase = "unprepared",
                    .diagnostic = "CPU pipeline maintenance has no prepared service"};
                const auto observed = owner->service()->measurementReadiness();
                if (observed.failed()) return observed;
                if (!observed.ready()) stage_readiness = observed;
            }
        if (!stage_readiness.ready()) return stage_readiness;
        if (moe_expert_overlay_maintenance_service_ &&
            moe_overlay_device_controller_graph_service_)
        {
            return {
                .state = InferenceMeasurementReadinessState::Failed,
                .owner = "expert_overlay",
                .phase = "duplicate_authority",
                .diagnostic = "Host and device ExpertOverlay maintenance authorities are both live",
            };
        }
        if (moe_expert_overlay_maintenance_service_)
        {
            overlay_readiness =
                moe_expert_overlay_maintenance_service_
                    ->measurementReadiness();
        }
        else if (moe_overlay_device_controller_graph_service_)
        {
            overlay_readiness =
                moe_overlay_device_controller_graph_service_
                    ->measurementReadiness();
        }
        if (overlay_readiness.failed() || !overlay_readiness.ready())
            return overlay_readiness;
        return overlay_readiness;
    }

    InferenceReadiness OrchestrationRunner::inferenceReadiness() const
    {
        if (!runner_)
        {
            return {
                .state = InferenceReadinessState::Failed,
                .diagnostic =
                    "Inference readiness requested before runner initialization",
            };
        }

        const InferenceReadiness runner_readiness =
            runner_->inferenceReadiness();
        if (!runner_readiness.ready())
            return runner_readiness;

        const InferenceMeasurementReadiness internal =
            internalInferencePreparationReadiness();
        if (internal.failed())
        {
            std::string diagnostic = internal.diagnostic;
            if (diagnostic.empty())
                diagnostic = "Internal inference preparation failed";
            if (!internal.phase.empty())
                diagnostic += " (phase=" + internal.phase + ')';
            return {
                .state = InferenceReadinessState::Failed,
                .diagnostic = std::move(diagnostic),
            };
        }
        if (!internal.ready())
        {
            return {
                .state = InferenceReadinessState::Preparing,
                .diagnostic = {},
            };
        }
        return {};
    }

    bool OrchestrationRunner::prepareForInference()
    {
        std::lock_guard<std::mutex> preparation_lock(
            inference_preparation_mutex_);

        const InferenceReadiness initial = inferenceReadiness();
        if (initial.ready())
            return true;
        if (initial.failed())
            return setError(initial.diagnostic);
        if (!runner_)
        {
            return setError(
                "Inference preparation requires an initialized runner");
        }
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->world_size() > 1 &&
            mpi_ctx_->rank() != mpi_coordinated_root_rank_)
        {
            return setError(
                "Only the coordinated request authority may drive inference preparation");
        }

        std::optional<InferenceMeasurementReadiness> last_report;

        while (true)
        {
            const InferenceMeasurementReadiness readiness =
                internalInferencePreparationReadiness();
            if (readiness.ready())
            {
                LOG_INFO(
                    "Inference topology preparation complete: owner="
                    << (readiness.owner.empty() ? "runtime" : readiness.owner)
                    << " work=" << readiness.completed_work_units << '/'
                    << readiness.required_work_units
                    << " synthetic_inference=false");
                return inferenceReadiness().ready();
            }
            if (readiness.failed())
            {
                std::string message =
                    "Internal inference preparation failed";
                if (!readiness.phase.empty())
                    message += " during " + readiness.phase;
                if (!readiness.diagnostic.empty())
                    message += ": " + readiness.diagnostic;
                return setError(std::move(message));
            }
            if (readiness.required_work_units == 0u)
            {
                return setError(
                    "Internal inference preparation reported an unbounded calibration corpus");
            }

            if (!last_report || readiness.crossedProgressMilestone(*last_report))
            {
                LOG_INFO(
                    "Preparing inference runtime: owner=" << readiness.owner
                    << " phase=" << readiness.phase << " work="
                    << readiness.completed_work_units << '/'
                    << readiness.required_work_units);
                last_report = readiness;
            }

            if (readiness.inference_requested ||
                readiness.requested_workload !=
                    InferenceMeasurementWorkloadKind::None ||
                readiness.inference_request_generation != 0u)
            {
                return setError(
                    "Inference topology preparation attempted to request synthetic inference");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    void OrchestrationRunner::setSkipLogitsGatherDecode(bool skip)
    {
        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SKIP_LOGITS_DECODE);
            int32_t val = skip ? 1 : 0;
            mpi_ctx_->broadcast_int32(
                &val, 1, mpi_coordinated_root_rank_);
        }

        if (runner_)
        {
            runner_->setSkipLogitsGatherDecode(skip);
        }
    }

    void OrchestrationRunner::setSkipLogitsGatherPrefill(bool skip)
    {
        if (runner_)
        {
            runner_->setSkipLogitsGatherPrefill(skip);
        }
    }

    void OrchestrationRunner::setSuppressTimeline(bool suppress)
    {
        if (runner_)
        {
            runner_->setSuppressTimeline(suppress);
        }
    }

    void OrchestrationRunner::setAccumulatePrefill(bool accumulate)
    {
        if (runner_)
        {
            runner_->setAccumulatePrefill(accumulate);
        }
    }

    void OrchestrationRunner::flushStageTimeline()
    {
        if (runner_)
        {
            runner_->flushStageTimeline();
        }
        MoEExpertOverlayProfiler::flush();
    }

    void OrchestrationRunner::setSamplingParams(const SamplingParams &params)
    {
        if (!std::isfinite(params.repetition_penalty) || params.repetition_penalty <= 0.0f)
            throw std::invalid_argument("repetition_penalty must be positive and finite");
        // Reject unsupported laws before an MPI command or any retained request
        // mutation. A failed admission must leave the previous law intact.
        if (runner_)
            requireSupportedMTPSamplingRequest(runner_->primaryDeviceId(), activeMTPRequestConfig(), params);

        // Broadcast to worker ranks
        if (mpi_coordinated_mode_ && mpi_ctx_ &&
            mpi_ctx_->rank() == mpi_coordinated_root_rank_ &&
            mpi_ctx_->world_size() > 1)
        {
            broadcastCommand(MPICommand::SET_SAMPLING);
            float params_buf[7] = {
                params.temperature,
                params.top_p,
                static_cast<float>(params.top_k),
                static_cast<float>(params.seed),
                params.presence_penalty,
                params.frequency_penalty,
                params.repetition_penalty};
            mpi_ctx_->broadcast(
                params_buf, 7, mpi_coordinated_root_rank_);
        }

        // Depth observations belong to one sampling law. A long-lived service
        // may alternate stochastic and argmax requests over the same retained
        // graph family; the new specialization must not reuse the old controller.
        if (active_sampling_params_.is_greedy() != params.is_greedy())
            mtp_depth_controller_.reset();
        active_sampling_params_ = params;
        generation_request_seeds_.reset();
        // Reset token history and deterministic RNG for a new conversation/request.
        sampler_ = Sampler(params.seed);
        if (runner_ &&
            !runner_->configureMTPRequestPenaltyPolicy(
                MTPRequestPenaltyPolicy{
                    .presence_penalty = params.presence_penalty,
                    .frequency_penalty = params.frequency_penalty,
                    .repetition_penalty = params.repetition_penalty,
                }))
        {
            throw std::runtime_error(
                "Inference runner rejected request penalty policy");
        }
    }

    SamplingParams OrchestrationRunner::getRecommendedSamplingParams(ThinkingMode mode) const
    {
        return model_generation_policy_.forMode(mode);
    }

    std::string OrchestrationRunner::getStopThinkingPrompt() const
    {
        return stop_thinking_prompt_;
    }

    ToolCallFormat OrchestrationRunner::getToolCallFormat() const
    {
        return tool_call_format_;
    }

    // =========================================================================
    // MPI Worker Loop (non-root ranks in server mode)
    // =========================================================================

    void OrchestrationRunner::broadcastCommand(MPICommand cmd)
    {
        if (!mpi_coordinated_mode_ || !mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return;

        if (mpi_ctx_->rank() != mpi_coordinated_root_rank_)
        {
            LLAMINAR_UNREACHABLE(
                "Only the coordinated root may publish an MPI worker command");
        }
        if (mpi_worker_command_lifecycle_ !=
            MPIWorkerCommandLifecycle::AcceptingCommands)
        {
            LLAMINAR_UNREACHABLE(
                "Cannot publish an MPI worker command after terminal SHUTDOWN");
        }

        int32_t tag = static_cast<int32_t>(cmd);
        mpi_ctx_->broadcast_int32(
            &tag, 1, mpi_coordinated_root_rank_);
    }

    bool OrchestrationRunner::yieldMPIWorkersForRetainedRunner()
    {
        if (!mtpRequestPolicyTransitionIsIdle())
        {
            return setError(
                "Cannot yield an MPI worker session until all request-owned state is reset");
        }
        if (mpi_worker_command_lifecycle_ !=
            MPIWorkerCommandLifecycle::AcceptingCommands)
        {
            return setError(
                "MPI worker session yield requires an accepting command channel");
        }

        if (mpi_ctx_ && mpi_ctx_->world_size() > 1)
        {
            if (!mpi_coordinated_mode_)
            {
                return setError(
                    "Multi-rank retained-runner yield requires coordinated mode");
            }
            if (mpi_ctx_->rank() != mpi_coordinated_root_rank_)
            {
                return setError(
                    "Only the coordinated root may publish a retained-runner yield");
            }
            broadcastCommand(MPICommand::YIELD_RETAINED_RUNNER);
        }

        /* The command is ordered after the preceding CLEAR_CACHE broadcast.
         * Followers cannot observe this edge until their own request reset has
         * completed, so no extra barrier or host synchronization is needed. */
        mpi_worker_command_lifecycle_ =
            MPIWorkerCommandLifecycle::YieldedRetainedRunner;
        PerfStatsCollector::addCounter(
            "orchestration",
            "retained_runner_session_yields",
            1.0,
            "request_teardown",
            runner_ ? runner_->primaryDeviceId().toString() : std::string{},
            {{"request_state", "reset"},
             {"model_state", "retained"}});
        return true;
    }

    void OrchestrationRunner::shutdownMPIWorkers()
    {
        /*
         * This is the terminal admission boundary even when the topology has
         * no remote command follower. Single-rank ExpertOverlay still owns a
         * background maintenance service whose final published bank must pass
         * its reader grace period and recycle its bootstrap slots while graph
         * diagnostics remain alive. Treating this branch as a no-op strands
         * that retirement until object destruction and makes terminal evidence
         * depend on MPI cardinality.
         */
        if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
        {
            if (mpi_worker_command_lifecycle_ ==
                MPIWorkerCommandLifecycle::ShutdownPublished)
            {
                return;
            }
            mpi_worker_command_lifecycle_ =
                MPIWorkerCommandLifecycle::ShutdownPublished;
            shutdownMoEExpertOverlayResidencyMaintenance(
                MoEOverlayMaintenanceDrainIntent::TerminalContextSeal);
            return;
        }

        if (mpi_ctx_->rank() != mpi_coordinated_root_rank_)
        {
            LLAMINAR_UNREACHABLE(
                "Only the coordinated root may close the MPI worker loop");
        }

        // The first shutdown releases worker ranks from their blocking command
        // receive.  Re-broadcasting after that point has no receiver and would
        // deadlock, so shutdown is deliberately idempotent at the protocol
        // boundary rather than relying on each caller to remember cleanup.
        if (mpi_worker_command_lifecycle_ ==
            MPIWorkerCommandLifecycle::ShutdownPublished)
        {
            return;
        }

        /*
         * SHUTDOWN closes new inference admission; it does not destroy
         * maintenance. Every worker handles the command by entering the same
         * topology-wide drain that the root enters immediately below. No rank
         * may enter a controller barrier while a peer is still blocked in this
         * command broadcast, which would mismatch Barrier against Bcast and
         * deadlock teardown after an otherwise successful request.
         */
        LOG_DEBUG("[MPI] Coordinated root rank "
                  << mpi_coordinated_root_rank_
                  << " publishing SHUTDOWN before the collective maintenance drain");
        broadcastCommand(MPICommand::SHUTDOWN);
        mpi_worker_command_lifecycle_ =
            MPIWorkerCommandLifecycle::ShutdownPublished;
        shutdownMoEExpertOverlayResidencyMaintenance(
            MoEOverlayMaintenanceDrainIntent::TerminalContextSeal);
    }

    void OrchestrationRunner::abortMPIWorkers(const std::string& reason)
    {
        if (mpi_ctx_ && mpi_ctx_->world_size() > 1 &&
            mpi_ctx_->rank() != mpi_coordinated_root_rank_)
        {
            terminateFailedMPIWorkerCommand("coordinated cancellation", reason);
        }
        shutdownMPIWorkers();
    }

    [[noreturn]] void OrchestrationRunner::terminateFailedMPIWorkerCommand(
        const char *command,
        const std::string &reason)
    {
        const int rank = mpi_ctx_ ? mpi_ctx_->rank() : -1;
        const std::string diagnostic =
            "[MPIWorkerLoop] rank " + std::to_string(rank) +
            " failed coordinated " + command + " command: " +
            (reason.empty() ? "no failure detail was recorded" : reason) +
            "; aborting because further collective order is indeterminate";
        LOG_ERROR(diagnostic);

        /*
         * A coordinated fatal error is precisely where participant-local
         * timings are most useful.  MPI_Abort bypasses ordinary runner
         * teardown, so publish the already-collected, rank-qualified evidence
         * before terminating the communicator. This performs no recovery and
         * cannot alter collective order; it only preserves diagnostics when
         * the caller explicitly configured a PerfStats artifact.
         */
        (void)PerfStatsCollector::flushFromEnv();

        // A real worker cannot safely receive another root command after a
        // graph-stage failure: a peer may already be waiting in a sparse MoE
        // all-gather or a TP collective that this rank will never enter.
        // MPI_Abort terminates every participant in this communicator promptly.
        if (mpi_ctx_)
        {
            const MPI_Comm communicator = mpi_ctx_->communicator();
            if (communicator != MPI_COMM_NULL)
            {
                const int abort_status = MPI_Abort(communicator, EXIT_FAILURE);
                LOG_ERROR("[MPIWorkerLoop] MPI_Abort unexpectedly returned with status "
                          << abort_status << "; terminating this worker directly");
                std::abort();
            }
        }

        // Scripted contexts intentionally have no live communicator. Throwing
        // preserves the same non-returning contract while making the invariant
        // observable in a device-free unit test.
        throw std::runtime_error(diagnostic);
    }

    void OrchestrationRunner::runMPIWorkerLoop()
    {
        if (!mpi_ctx_ || mpi_ctx_->rank() == mpi_coordinated_root_rank_)
        {
            LOG_WARN("[MPIWorkerLoop] Should only be called on worker ranks");
            return;
        }

        LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                          << " entering worker loop");

        const auto notifyHostOverlayMaintenance =
            [this](const char *command, uint64_t committed_tokens)
        {
            /*
             * The coordinated root calls maybeApplyMoERebalance() at the same
             * committed request boundaries. Host-authoritative ExpertOverlay
             * maintenance is process-local, so a worker wakes that service at
             * its existing command boundary. The topology-wide device
             * controller is deliberately excluded: a hosted follower wakes it
             * from each authenticated transaction ticket, while a non-hosted
             * path receives continuation progress in the next inference payload.
             */
            if (!moe_expert_overlay_maintenance_service_ ||
                committed_tokens == 0u)
                return;
            if (!maybeApplyMoERebalance(committed_tokens))
            {
                terminateFailedMPIWorkerCommand(
                    command,
                    lastError().empty()
                        ? "ExpertOverlay maintenance notification failed"
                        : lastError());
            }
        };

        while (true)
        {
            // Wait for a command from the coordinated root.
            int32_t tag = 0;
            mpi_ctx_->broadcast_int32(
                &tag, 1, mpi_coordinated_root_rank_);
            auto cmd = static_cast<MPICommand>(tag);
            if (traceChatGeneratedTokensEnabled())
            {
                LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                 << " received command tag " << tag);
            }

            switch (cmd)
            {
            case MPICommand::CLEAR_CACHE:
            {
                clearCache();
                break;
            }

            case MPICommand::PURGE_PREFIX_CACHE:
            {
                if (!purgePrefixCache())
                {
                    terminateFailedMPIWorkerCommand(
                        "PURGE_PREFIX_CACHE",
                        lastError().empty()
                            ? "Reusable prefix-cache purge failed"
                            : lastError());
                }
                break;
            }

            case MPICommand::SET_SAMPLING:
            {
                // Receive sampling params
                float params_buf[7]; // temperature, top_p, top_k, seed, penalties
                mpi_ctx_->broadcast(
                    params_buf, 7, mpi_coordinated_root_rank_);
                SamplingParams sp;
                sp.temperature = params_buf[0];
                sp.top_p = params_buf[1];
                sp.top_k = static_cast<int>(params_buf[2]);
                sp.seed = static_cast<uint64_t>(params_buf[3]);
                sp.presence_penalty = params_buf[4];
                sp.frequency_penalty = params_buf[5];
                sp.repetition_penalty = params_buf[6];
                setSamplingParams(sp);
                break;
            }

            case MPICommand::SET_STOP_TOKENS:
            {
                int32_t stop_token_count = 0;
                mpi_ctx_->broadcast_int32(
                    &stop_token_count, 1, mpi_coordinated_root_rank_);
                if (stop_token_count < 0)
                {
                    terminateFailedMPIWorkerCommand(
                        "SET_STOP_TOKENS",
                        "received a negative request stop-token count");
                }

                std::vector<int32_t> stop_tokens(
                    static_cast<size_t>(stop_token_count));
                if (stop_token_count > 0)
                {
                    mpi_ctx_->broadcast_int32(
                        stop_tokens.data(),
                        static_cast<size_t>(stop_token_count),
                        mpi_coordinated_root_rank_);
                }
                setStopTokens(stop_tokens);
                break;
            }

            case MPICommand::SET_MTP_REQUEST_POLICY:
            {
                MTPRequestPolicyWire wire{};
                mpi_ctx_->broadcast_int32(
                    wire.data(), wire.size(), mpi_coordinated_root_rank_);
                std::string decode_error;
                const auto policy =
                    decodeMTPRequestPolicy(wire, &decode_error);
                if (!policy)
                {
                    terminateFailedMPIWorkerCommand(
                        "SET_MTP_REQUEST_POLICY",
                        decode_error.empty()
                            ? "received a malformed MTP request policy"
                            : decode_error);
                }
                if (!configureMTPRequestPolicy(*policy))
                {
                    terminateFailedMPIWorkerCommand(
                        "SET_MTP_REQUEST_POLICY",
                        lastError().empty()
                            ? "worker rejected the root-authored MTP request policy"
                            : lastError());
                }
                break;
            }

            case MPICommand::PREFILL:
            {
                /* Receive the typed command header before entering the retained
                 * transaction follower. This is the same continuation-owned
                 * cadence sideband used by decode and forced-token commands. */
                std::array<std::int32_t, 3> command_header{};
                mpi_ctx_->broadcast_int32(
                    command_header.data(),
                    command_header.size(),
                    mpi_coordinated_root_rank_);
                const std::int32_t n_tokens = command_header[0];
                const std::int32_t retired_prefill_tokens = command_header[1];
                const std::int32_t retired_decode_tokens = command_header[2];

                if (n_tokens <= 0 || retired_prefill_tokens < 0 ||
                    retired_decode_tokens < 0)
                {
                    terminateFailedMPIWorkerCommand(
                        "PREFILL",
                        "received an invalid prompt or retired-progress count");
                }

                if (!continuation_request_group_)
                    terminateFailedMPIWorkerCommand(
                        "PREFILL", "worker has no frozen continuation recipient group");
                const auto tokens = continuation_request_group_->receivePrompt(
                    static_cast<std::size_t>(n_tokens), mpi_coordinated_root_rank_);

                /*
                 * This command is the remote half of the same public request
                 * boundary entered by the continuation rank. In particular an
                 * expert-only follower never calls prefill() below, so activate
                 * its process-local histogram authority before it consumes the
                 * first retained transaction ticket. Otherwise certification
                 * quarantine prevents its maintenance worker from receiving
                 * and acknowledging the coordinator's first proposal.
                 */
                if (!activateMoEOverlayDemandAtRequestBoundary())
                {
                    terminateFailedMPIWorkerCommand(
                        "PREFILL",
                        lastError().empty()
                            ? "ExpertOverlay follower rejected request-boundary demand activation"
                            : lastError());
                }

                if ((retired_prefill_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Prefill,
                               static_cast<std::uint64_t>(
                                   retired_prefill_tokens)))) ||
                    (retired_decode_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Decode,
                               static_cast<std::uint64_t>(
                                   retired_decode_tokens)))))
                {
                    terminateFailedMPIWorkerCommand(
                        "PREFILL",
                        "worker rejected continuation-authoritative retired inference progress");
                }

                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * A remote ExpertOverlay rank has no token embedding, KV,
                     * logits, or sampler authority during prefill either. The
                     * root graph publishes one authenticated ticket per
                     * retained prefill segment and then a terminal ticket for
                     * this outer command. The world header admits that lifecycle;
                     * the continuation group excludes this follower from prompt
                     * allocation and transport entirely.
                     */
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "PREFILL",
                            follower_result.error.empty()
                                ? "ExpertOverlay prefill transaction follower failed"
                                : follower_result.error);
                    }
                    if (traceChatGeneratedTokensEnabled())
                    {
                        LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                         << " completed PREFILL follower command transactions="
                                                         << follower_result.executed_transactions);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_prefill_commands",
                        1.0,
                        "prefill",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyHostOverlayMaintenance(
                        "PREFILL transaction-follower maintenance boundary",
                        0u);
                    break;
                }

                if (!prefill(tokens))
                    terminateFailedMPIWorkerCommand("PREFILL", lastError());
                notifyHostOverlayMaintenance(
                    "PREFILL maintenance boundary", 0u);
                break;
            }

            case MPICommand::DECODE_STEP:
            {
                std::array<std::int32_t, 3> command_payload{};
                mpi_ctx_->broadcast_int32(
                    command_payload.data(),
                    command_payload.size(),
                    mpi_coordinated_root_rank_);
                const std::int32_t token_budget = command_payload[0];
                const std::int32_t retired_prefill_tokens =
                    command_payload[1];
                const std::int32_t retired_decode_tokens =
                    command_payload[2];
                if (retired_prefill_tokens < 0 ||
                    retired_decode_tokens < 0 ||
                    (retired_prefill_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Prefill,
                               static_cast<std::uint64_t>(
                                   retired_prefill_tokens)))) ||
                    (retired_decode_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Decode,
                               static_cast<std::uint64_t>(
                                   retired_decode_tokens)))))
                {
                    terminateFailedMPIWorkerCommand(
                        "DECODE_STEP",
                        "worker rejected continuation-authoritative retired inference progress");
                }
                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * A heterogeneous expert rank owns neither continuation KV
                     * nor sampling state. It follows the root's authenticated
                     * graph tickets until Complete instead of masquerading as a
                     * second decoder and entering unrelated token collectives.
                     */
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "DECODE_STEP",
                            follower_result.error.empty()
                                ? "ExpertOverlay transaction follower failed"
                                : follower_result.error);
                    }
                    if (traceChatGeneratedTokensEnabled())
                    {
                        LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                         << " completed DECODE_STEP follower command transactions="
                                                         << follower_result.executed_transactions);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_decode_commands",
                        1.0,
                        "decode",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyHostOverlayMaintenance(
                        "DECODE_STEP transaction-follower maintenance boundary",
                        follower_result.retired_decode_progress_tokens);
                    break;
                }
                setDecodeStepTokenBudget(token_budget);
                GenerationResult decode_result = decodeStep();
                // Rank 0 scopes thinking-budget decode through ChatCompletionHandler;
                // reset workers too so forced-token or later control commands never
                // observe a stale request-local budget.
                setDecodeStepTokenBudget(0);
                if (!decode_result.success())
                    terminateFailedMPIWorkerCommand("DECODE_STEP", decode_result.error);
                notifyHostOverlayMaintenance(
                    "DECODE_STEP maintenance boundary",
                    decode_result.tokens.size());
                break;
            }

            case MPICommand::FORCE_DECODE_TOKEN:
            {
                std::array<std::int32_t, 3> command_payload{};
                mpi_ctx_->broadcast_int32(
                    command_payload.data(),
                    command_payload.size(),
                    mpi_coordinated_root_rank_);
                const std::int32_t forced = command_payload[0];
                const std::int32_t retired_prefill_tokens =
                    command_payload[1];
                const std::int32_t retired_decode_tokens =
                    command_payload[2];
                if (retired_prefill_tokens < 0 ||
                    retired_decode_tokens < 0 ||
                    (retired_prefill_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Prefill,
                               static_cast<std::uint64_t>(
                                   retired_prefill_tokens)))) ||
                    (retired_decode_tokens != 0 &&
                     (!moe_overlay_device_controller_graph_service_ ||
                      !moe_overlay_device_controller_graph_service_
                           ->notifyInferenceProgress(
                               MoEOverlayDeviceControllerGraphService::
                                   InferencePhase::Decode,
                               static_cast<std::uint64_t>(
                                   retired_decode_tokens)))))
                {
                    terminateFailedMPIWorkerCommand(
                        "FORCE_DECODE_TOKEN",
                        "worker rejected continuation-authoritative retired inference progress");
                }
                if (traceChatGeneratedTokensEnabled())
                {
                    LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                     << " received forced token "
                                                     << forced);
                }
                if (moe_overlay_inference_transaction_follower_)
                {
                    /*
                     * The forced token is continuation policy state. A remote
                     * expert rank consumes only the authenticated retained
                     * graph (if this command advances model state) followed by
                     * its terminal ticket. It must still join the ordinary MPI
                     * command fence after the follower returns so the next root
                     * command cannot overtake this control boundary.
                     */
                    ScopedMPICoordinatedCommandFence command_fence(
                        mpi_ctx_.get(),
                        /*active=*/true,
                        "forceDecodeTokenFollower");
                    const auto follower_result =
                        moe_overlay_inference_transaction_follower_
                            ->runOneCommand();
                    if (!follower_result.ok)
                    {
                        terminateFailedMPIWorkerCommand(
                            "FORCE_DECODE_TOKEN",
                            follower_result.error.empty()
                                ? "ExpertOverlay forced-token transaction follower failed"
                                : follower_result.error);
                    }
                    if (traceChatGeneratedTokensEnabled())
                    {
                        LOG_INFO("[MPIWorkerLoop] rank " << mpi_ctx_->rank()
                                                         << " completed FORCE_DECODE_TOKEN follower command transactions="
                                                         << follower_result.executed_transactions);
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_transaction",
                        "follower_forced_token_commands",
                        1.0,
                        "decode",
                        {},
                        {{"transactions",
                          std::to_string(
                              follower_result.executed_transactions)},
                         {"token_state_mutated", "false"}});
                    notifyHostOverlayMaintenance(
                        "FORCE_DECODE_TOKEN transaction-follower maintenance boundary",
                        follower_result.retired_decode_progress_tokens);
                    break;
                }
                GenerationResult forced_result = forceDecodeToken(forced);
                if (!forced_result.success())
                    terminateFailedMPIWorkerCommand(
                        "FORCE_DECODE_TOKEN",
                        forced_result.error);
                notifyHostOverlayMaintenance(
                    "FORCE_DECODE_TOKEN maintenance boundary",
                    forced_result.tokens.size());
                break;
            }

            case MPICommand::SKIP_LOGITS_DECODE:
            {
                int32_t skip = 0;
                mpi_ctx_->broadcast_int32(
                    &skip, 1, mpi_coordinated_root_rank_);
                runner_->setSkipLogitsGatherDecode(skip != 0);
                break;
            }

            case MPICommand::YIELD_RETAINED_RUNNER:
            {
                if (!mtpRequestPolicyTransitionIsIdle())
                {
                    terminateFailedMPIWorkerCommand(
                        "YIELD_RETAINED_RUNNER",
                        "worker retained live request state after the ordered reset command");
                }
                LOG_DEBUG(
                    "[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                    << " yielded an idle retained runner");
                mpi_worker_command_lifecycle_ =
                    MPIWorkerCommandLifecycle::YieldedRetainedRunner;
                PerfStatsCollector::addCounter(
                    "orchestration",
                    "retained_runner_session_yields",
                    1.0,
                    "request_teardown",
                    runner_ ? runner_->primaryDeviceId().toString()
                            : std::string{},
                    {{"request_state", "reset"},
                     {"model_state", "retained"}});
                return;
            }

            case MPICommand::SHUTDOWN:
            {
                LOG_DEBUG("[MPIWorkerLoop] Rank " << mpi_ctx_->rank()
                                                  << " received SHUTDOWN");
                /*
                 * The collective receive is the follower's terminal protocol
                 * edge. Publish it locally before draining maintenance so no
                 * later fixture or server cleanup can mutate request state
                 * after this rank has left the command channel.
                 */
                mpi_worker_command_lifecycle_ =
                    MPIWorkerCommandLifecycle::ShutdownPublished;
                shutdownMoEExpertOverlayResidencyMaintenance(
                    MoEOverlayMaintenanceDrainIntent::TerminalContextSeal);
                return;
            }

            default:
                terminateFailedMPIWorkerCommand(
                    "unknown",
                    "received unsupported command tag " + std::to_string(tag));
            }
        }
    }

    PrefixCacheTelemetrySources OrchestrationRunner::prefixCacheTelemetrySources() const
    {
        return runner_ ? runner_->prefixCacheTelemetrySources() : PrefixCacheTelemetrySources{};
    }

} // namespace llaminar2
