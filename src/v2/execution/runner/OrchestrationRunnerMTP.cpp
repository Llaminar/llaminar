/**
 * @file OrchestrationRunnerMTP.cpp
 * @brief OrchestrationRunner speculative MTP transaction execution (decodeStepMTP).
 */

#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{
    GenerationResult OrchestrationRunner::decodeStepMTP()
    {
        PerfStatsCollector::ScopedTimer step_timer("mtp", "decode_step_total", "decode");
        PerfStatsCollector::addCounter("mtp", "decode_step_calls", 1.0, "decode");

        GenerationResult result;
        const int vocab = vocabSize();
        if (vocab <= 0)
        {
            result.error = "Invalid vocabulary size for MTP decode";
            return result;
        }

        const bool use_ready_logits = prefill_logits_ready_;
        const std::optional<ReadyMTPCondition> ready_condition =
            ready_mtp_condition_;
        const std::optional<int32_t> pending_condition_token =
            pending_mtp_condition_token_;
        const std::optional<SamplingParams> pending_condition_params =
            pending_mtp_condition_params_;
        const std::optional<DeviceResidentLogicalSequenceStateHandle>
            pending_condition_resident_state =
                pending_mtp_condition_resident_state_;
        const std::optional<DeviceResidentLogicalSequenceStateHandle>
            prelaunched_first_sidecar_resident_state =
                prelaunched_mtp_first_sidecar_resident_state_;
        const std::optional<SamplingParams> prelaunched_first_sidecar_params =
            prelaunched_mtp_first_sidecar_params_;
        prefill_logits_ready_ = false;
        ready_mtp_condition_.reset();
        prelaunched_mtp_first_sidecar_resident_state_.reset();
        prelaunched_mtp_first_sidecar_params_.reset();
        PrefixStateSnapshot rollback_checkpoint;
        bool rollback_checkpoint_captured = false;
        PrefixStateSnapshot verifier_base_checkpoint;
        int transaction_base_cached_tokens = -1;

        auto current_checkpoint_capture_request =
            [&](int draft_tokens,
                int preceding_condition_rows,
                const char *context,
                std::string *error)
            -> std::optional<PrefixCheckpointCaptureRequest>
        {
            const std::optional<int> position =
                currentDecodeTransactionPositionForPlanning(context, error);
            if (!position)
                return std::nullopt;
            return PrefixCheckpointCaptureRequest{
                .sequence_index = 0,
                .logical_cached_tokens = *position,
                .maximum_main_append_tokens =
                    draft_tokens + 1 + preceding_condition_rows,
                .maximum_shifted_append_tokens =
                    std::max(1, draft_tokens) + preceding_condition_rows};
        };

        auto fail_without_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            PerfStatsCollector::addCounter("mtp", "decode_step_failures", 1.0, "decode",
                                           std::string{}, {{"reason", message}});
            prefill_logits_ready_ = use_ready_logits;
            ready_mtp_condition_ = ready_condition;
            pending_mtp_condition_token_ = pending_condition_token;
            pending_mtp_condition_params_ = pending_condition_params;
            pending_mtp_condition_resident_state_ =
                pending_condition_resident_state;
            prelaunched_mtp_first_sidecar_resident_state_.reset();
            prelaunched_mtp_first_sidecar_params_.reset();
            result.error = message;
            return result;
        };

        auto capture_rollback_checkpoint =
            [&](int draft_tokens, int preceding_condition_rows) -> bool
        {
            if (rollback_checkpoint_captured)
                return true;

            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "capture_live_prefix_state",
                    "decode");
                if (!runner_->ensureMTPCheckpointTerminalHidden())
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "capture_live_prefix_terminal_hidden_failures",
                        1.0,
                        "decode");
                    return false;
                }
                std::string position_error;
                const auto capture_request =
                    current_checkpoint_capture_request(
                        draft_tokens,
                        preceding_condition_rows,
                        "rollback checkpoint capture",
                        &position_error);
                if (!capture_request)
                {
                    last_error_ = std::move(position_error);
                    return false;
                }
                rollback_checkpoint =
                    runner_->captureLivePrefixCheckpoint(*capture_request);
            }
            if (!rollback_checkpoint.valid)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "capture_live_prefix_state_failures",
                    1.0,
                    "decode");
                return false;
            }
            rollback_checkpoint_captured = true;
            transaction_base_cached_tokens = rollback_checkpoint.cached_tokens;
            PerfStatsCollector::addCounter(
                "mtp",
                rollback_checkpoint.logical_checkpoint
                    ? "live_prefix_checkpoint_logical"
                    : "live_prefix_checkpoint_payload",
                1.0,
                "decode");
            return true;
        };

        auto fail_after_checkpoint = [&](const std::string &message) -> GenerationResult
        {
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "disable_all_position_logits_after_failure", "decode");
                runner_->setComputeAllPositionLogits(false);
                runner_->setComputeRowIndexedAllPositionLogits(false, 0);
            }
            bool restored = false;
            if (rollback_checkpoint_captured)
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "restore_live_prefix_state_after_failure", "decode");
                restored = runner_->restoreLivePrefixState(rollback_checkpoint);
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "decode_step_failures_without_rollback_checkpoint",
                    1.0,
                    "decode",
                    {},
                    {{"reason", message}});
            }
            if (restored)
            {
                ++mtp_stats_.rollbacks;
                ++mtp_stats_.transaction_rollbacks;
                PerfStatsCollector::addCounter("mtp", "rollbacks", 1.0, "decode");
                PerfStatsCollector::addCounter("mtp", "transaction_rollbacks", 1.0, "decode");
            }
            PerfStatsCollector::addCounter("mtp", "decode_step_failures", 1.0, "decode",
                                           std::string{}, {{"reason", message}});
            prefill_logits_ready_ = use_ready_logits;
            ready_mtp_condition_ = ready_condition;
            pending_mtp_condition_token_ = pending_condition_token;
            pending_mtp_condition_params_ = pending_condition_params;
            pending_mtp_condition_resident_state_ =
                pending_condition_resident_state;
            prelaunched_mtp_first_sidecar_resident_state_.reset();
            prelaunched_mtp_first_sidecar_params_.reset();
            result.error = message;
            return result;
        };

        /*
         * A heterogeneous ExpertOverlay command may contain more than one
         * complete graph sequence.  In particular, an MTP condition forward is
         * a serial main-model transaction, while the sidecar/verifier work that
         * follows it is a separate speculative transaction whose exact width is
         * known only after response-budget clipping.  Keeping those two shapes
         * separate prevents the configured draft depth from masquerading as
         * work that never ran at a one-token response boundary.
         */
        bool overlay_graph_sequence_active = false;
        auto begin_overlay_graph_sequence =
            [&](int draft_depth, const char *context) -> std::optional<std::string>
        {
            if (!moe_overlay_inference_transaction_coordinator_)
                return std::nullopt;
            if (overlay_graph_sequence_active)
            {
                return std::string("ExpertOverlay MTP attempted to overlap graph sequences at ") +
                       (context ? context : "unknown");
            }
            std::string sequence_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->beginGraphSequence(draft_depth, &sequence_error))
            {
                return std::string("ExpertOverlay MTP graph-sequence admission failed at ") +
                       (context ? context : "unknown") + ": " +
                       sequence_error;
            }
            overlay_graph_sequence_active = true;
            return std::nullopt;
        };
        auto retire_overlay_graph_sequence =
            [&](const char *context) -> std::optional<std::string>
        {
            if (!moe_overlay_inference_transaction_coordinator_ ||
                !overlay_graph_sequence_active)
            {
                return std::nullopt;
            }
            std::string sequence_error;
            if (!moe_overlay_inference_transaction_coordinator_
                     ->retireCompletedGraphSequence(&sequence_error))
            {
                return std::string("ExpertOverlay MTP graph-sequence retirement failed at ") +
                       (context ? context : "unknown") + ": " +
                       sequence_error;
            }
            overlay_graph_sequence_active = false;
            return std::nullopt;
        };

        const bool needs_mpi_mtp_boundary_fence =
            mtpRankRequiresModelCollectives(mtpRankParticipation());
        auto fence_mpi_mtp_boundary =
            [&](const char *boundary) -> std::optional<std::string>
        {
            if (!needs_mpi_mtp_boundary_fence)
                return std::nullopt;

            /*
             * Global/NodeLocal TP ranks must enter sidecar and verifier
             * collectives in the same logical order. A local runner flush only
             * drains the participant's own work; it does not stop the root from
             * starting target-verifier allreduces while rank 1 is still inside
             * the MTP sidecar. The shared-memory fast path matches by epoch and
             * payload size, so this boundary fence is part of the transaction
             * contract rather than a best-effort scheduling hint.
             */
            const std::string boundary_name = boundary ? boundary : "unknown";
            try
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "sidecar_mpi_boundary_fence",
                    "decode",
                    std::string{},
                    {{"boundary", boundary_name},
                     {"rank", std::to_string(mpi_ctx_->rank())},
                     {"world_size", std::to_string(mpi_ctx_->world_size())}});
                mpi_ctx_->barrier();
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_mpi_boundary_fences",
                    1.0,
                    "decode",
                    std::string{},
                    {{"boundary", boundary_name},
                     {"rank", std::to_string(mpi_ctx_->rank())},
                     {"world_size", std::to_string(mpi_ctx_->world_size())}});
            }
            catch (const std::exception &ex)
            {
                return std::string("MTP MPI sidecar boundary fence failed at ") +
                       boundary_name + ": " + ex.what();
            }
            catch (...)
            {
                return std::string("MTP MPI sidecar boundary fence failed at ") +
                       boundary_name;
            }
            return std::nullopt;
        };

        if (ready_condition.has_value())
        {
            /*
             * A ready verifier condition is sampled one decode boundary before
             * it is consumed. Treat its authority and sampling contract as one
             * atomic state: a detached condition or a changed policy would mix
             * two sampling regimes in one request.
             */
            if (!use_ready_logits)
            {
                return fail_after_checkpoint(
                    "Ready MTP condition exists without a terminal-logits boundary");
            }
            if (!ready_condition->valid())
            {
                return fail_after_checkpoint(
                    "Ready MTP condition has an invalid authority payload");
            }
            if (!samplingParamsEqual(
                    ready_condition->sampling_params,
                    active_sampling_params_))
            {
                return fail_after_checkpoint(
                    "Ready MTP condition was sampled with different sampling parameters");
            }
        }

        const MTPRuntimeConfig mtp = activeMTPRequestConfig();
        try
        {
            requireSupportedMTPSamplingRequest(runner_->primaryDeviceId(), mtp, active_sampling_params_);
        }
        catch (const UnsupportedMTPSamplingRequest &error)
        {
            return fail_without_checkpoint(error.what());
        }
        const bool stochastic_verify =
            mtp.verify_mode == MTPVerifyMode::SpeculativeSampling &&
            !active_sampling_params_.is_greedy();
        const bool stochastic_device_verify =
            stochastic_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceStochasticMTPVerification();
        // Unseeded means fresh request entropy, not host-generated draws on
        // each transaction. Retain that entropy across budget continuations;
        // captured rejection kernels own every subsequent logical-position draw.
        if (stochastic_device_verify && !generation_request_seeds_)
            generation_request_seeds_.emplace(std::vector<uint64_t>{
                resolveRequestBatchedStochasticSeed(active_sampling_params_, 0)});
        const bool stochastic_host_verify =
            stochastic_verify &&
            !runner_->primaryDeviceId().is_gpu();
        const bool use_serial_sample_equivalent_host_stochastic =
            stochastic_host_verify &&
            active_sampling_params_.seed != 0;
        const bool use_sampling_penalties =
            active_sampling_params_.has_penalties() && !stochastic_verify;
        if (runner_->primaryDeviceId().is_gpu() &&
            !runner_->supportsMTPSidecarLogitsStreamHandoff())
        {
            /*
             * This is a foundational GPU MTP ownership contract, so validate it
             * before selecting a verifier policy or inspecting any secondary
             * grouped-publication capability. No GPU MTP transaction may begin
             * if its first sidecar producer cannot hand ownership to the next
             * device consumer through an explicit stream event.
             */
            return fail_without_checkpoint(
                "GPU MTP requires device-resident sidecar stream handoff");
        }
        const MTPDepthPolicyModelClass verifier_model_class =
            resolveMTPDepthPolicyModelClass(model_ctx_.get());
        const int verifier_policy_probe_rows =
            std::max(1, effectiveMTPMaxDraftDepth(mtp));
        const MTPVerifierPolicyDecision verifier_policy =
            chooseMTPVerifierPolicy(
                MTPVerifierPolicyInput{
                    .greedy_sampling = active_sampling_params_.is_greedy(),
                    .stochastic_verify = stochastic_verify,
                    .uses_sampling_penalties = use_sampling_penalties,
                    .supports_row_local_penalty_application =
                        !use_sampling_penalties ||
                        runner_->supportsRowLocalAllPositionPenaltyApplication(),
                });
        if (verifier_policy.path == MTPVerifierExecutionPath::Unsupported)
        {
            return fail_after_checkpoint(
                std::string("MTP verifier policy selected unsupported path: ") +
                verifier_policy.reason);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "verifier_policy_selections",
            1.0,
            "decode",
            {},
            {{"path", std::to_string(static_cast<int>(verifier_policy.path))},
             {"reason", verifier_policy.reason},
             {"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
             {"probe_rows", std::to_string(verifier_policy_probe_rows)},
             {"grouped_outcome_required", "true"},
             {"device_publication_supported",
              perfBool(
                  runner_->supportsDeviceResidentMTPSpecStatePublication())}});
        const bool use_grouped_outcome_device_resident_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication();
        const bool use_grouped_decode_equivalent_outcome_verifier =
            verifier_policy.path ==
            MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome;
        if (use_grouped_decode_equivalent_outcome_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            stochastic_verify &&
            !stochastic_device_verify)
        {
            return fail_without_checkpoint(
                "GPU stochastic MTP requires device-resident distribution verification");
        }
        if (use_grouped_decode_equivalent_outcome_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            !use_grouped_outcome_device_resident_publication_verifier)
        {
            /*
             * CUDA/ROCm grouped verifier rows are only a production path when
             * the accepted-state publication also stays device-resident.  If we
             * already know the runner cannot publish the compact outcome on
             * device, fail before enabling verifier-row logits or forwarding
             * any grouped rows; row replay and host step plans are diagnostics,
             * not GPU hot-path substitutes.
             */
            return fail_without_checkpoint(
                "Grouped decode-equivalent MTP verifier has no grouped publication path; GPU grouped verifier requires device-resident accepted-state publication");
        }
        const DeviceGenerationLoopTopology generation_loop_topology =
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        const DeviceGenerationExecutionPolicy generation_execution_policy =
            runner_->deviceGenerationExecutionPolicy(
                generation_loop_topology);
        if (use_grouped_outcome_device_resident_publication_verifier &&
            generation_execution_policy ==
                DeviceGenerationExecutionPolicy::Unsupported)
        {
            return fail_without_checkpoint(
                "GPU grouped MTP has no complete generation-loop execution policy for the requested depth topology");
        }

        /*
         * Native dynamic generation materializes one policy-complete parent
         * from the first transaction. HIP's explicit hosted policy keeps the
         * existing transaction loop and must never attempt native composition.
         * Preserve this admission edge before beginDeviceResidentGeneration()
         * consumes the pending marker.
         */
        const bool materialize_dynamic_generation_loop_this_step =
            device_generation_admission_.awaitsAdmission() &&
            use_grouped_outcome_device_resident_publication_verifier &&
            mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic;
        const bool use_grouped_outcome_host_publication_verifier =
            verifier_policy.path ==
                MTPVerifierExecutionPath::GroupedDecodeEquivalentOutcome &&
            !use_grouped_outcome_device_resident_publication_verifier &&
            !runner_->primaryDeviceId().is_gpu() &&
            (!stochastic_verify || stochastic_host_verify);
        if (use_grouped_outcome_host_publication_verifier)
        {
            /*
             * CPU grouped verification still publishes through host-visible step
             * plans. GPU grouped verification must use the compact
             * device-resident publisher; allowing this middle lane on CUDA/ROCm
             * quietly reintroduced host-owned MTP state after verifier rows had
             * already been proven decode-equivalent.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "grouped_outcome_host_publication_uses",
                1.0,
                "decode",
                {},
                {{"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
                 {"probe_rows", std::to_string(verifier_policy_probe_rows)},
                 {"reason", verifier_policy.reason}});
        }
        if (use_ready_logits && pending_condition_token.has_value())
        {
            return fail_after_checkpoint(
                "MTP decode found both ready terminal logits and a pending condition token");
        }
        const bool pending_condition_has_resident_state =
            pending_condition_resident_state.has_value() &&
            pending_condition_resident_state->valid();
        const bool pending_condition_candidate =
            pending_condition_token.has_value() && !use_ready_logits;
        /*
         * A rejected verifier row has already been emitted to the caller, but
         * it is not live model state until the next decode step consumes it as
         * the condition token.  Grouped-host publication uses host-visible
         * transaction plans rather than device-resident mailboxes, but it still
         * shares the same pending-condition contract as the other verifier-row
         * publishers.  Treating it as a bypass would sample from stale logits
         * and skip the rejected correction token on the next step.
         */
        const bool use_pending_condition_row =
            pending_condition_candidate &&
            (use_grouped_outcome_host_publication_verifier ||
             use_grouped_outcome_device_resident_publication_verifier);

        // Seal the transaction width before checkpointing or mutating caches.
        // Graph capacity, admitted drafts, and response publication are distinct:
        // use this same admitted width below instead of resolving it twice.
        const bool hosted_dynamic_continuation = isHostedDynamicMTPContinuation(mtp);
        const int requested_speculative_draft_count = hosted_dynamic_continuation
            ? 0 : currentMTPDraftDepth(mtp);
        // Capacity here admits the response/first scalar only. A continuation's
        // actual model geometry comes exclusively from native admission ticket
        // zero, after initialization preserves the device-owned learner.
        const int transaction_draft_capacity = hosted_dynamic_continuation
            ? effectiveMTPMaxDraftDepth(mtp)
            : materialize_dynamic_generation_loop_this_step &&
                    generation_execution_policy == DeviceGenerationExecutionPolicy::NativeConditionalGraph
                ? effectiveMTPMaxDraftDepth(mtp)
                : requested_speculative_draft_count;
        if (transaction_draft_capacity < requested_speculative_draft_count)
        {
            return fail_without_checkpoint(
                "Dynamic MTP graph-family capture capacity is narrower than its admitted device selector");
        }
        const bool first_condition_was_already_emitted =
            use_pending_condition_row ||
            (use_ready_logits && ready_condition.has_value() &&
             ready_condition->wasAlreadyEmitted());
        const int first_token_output_budget_cost =
            first_condition_was_already_emitted ? 0 : 1;
        const int pre_sample_effective_draft_count = mtpTransactionDraftCount(
            transaction_draft_capacity, decode_step_token_budget_,
            first_token_output_budget_cost, MTPCommitBudgetAuthority::Host);
        const int speculative_draft_count = mtpTransactionDraftCount(
            transaction_draft_capacity, decode_step_token_budget_,
            first_token_output_budget_cost,
            use_grouped_outcome_device_resident_publication_verifier
                ? MTPCommitBudgetAuthority::Device : MTPCommitBudgetAuthority::Host);
        if (speculative_draft_count < 0)
            return fail_without_checkpoint("Invalid MTP transaction draft admission");
        const bool ready_condition_has_resident_state =
            use_ready_logits &&
            ready_condition.has_value() &&
            ready_condition->resident_state.has_value() &&
            ready_condition->resident_state->valid();
        std::optional<DeviceResidentLogicalSequenceStateHandle>
            first_token_resident_state;
        if (use_pending_condition_row &&
            pending_condition_has_resident_state)
        {
            first_token_resident_state = pending_condition_resident_state;
        }
        else if (ready_condition_has_resident_state)
        {
            first_token_resident_state = ready_condition->resident_state;
        }
        if (use_pending_condition_row)
        {
            if (!pending_condition_params.has_value())
            {
                return fail_after_checkpoint(
                    "Pending MTP condition token is missing the sampling parameters that produced it");
            }
            if (!samplingParamsEqual(*pending_condition_params, active_sampling_params_))
            {
                return fail_after_checkpoint(
                    "Pending MTP condition token was sampled with different sampling parameters");
            }
            if (pending_condition_resident_state.has_value() &&
                !pending_condition_has_resident_state)
            {
                return fail_after_checkpoint(
                    "Pending MTP condition resident logical-state handle is stale or incomplete");
            }
            if (use_grouped_outcome_device_resident_publication_verifier &&
                !pending_condition_has_resident_state)
            {
                return fail_after_checkpoint(
                    "Grouped-outcome MTP pending condition is missing device-resident logical state");
            }
        }
        if (pending_condition_token.has_value() &&
            !use_pending_condition_row)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "pending_condition_fast_path_bypasses",
                1.0,
                "decode",
                {},
                {{"verifier_path",
                  std::to_string(static_cast<int>(verifier_policy.path))}});
        }
        const bool verify_sidecar_preserves_main_state =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE");
        const bool verify_commit_replay_check =
            DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_COMMIT_REPLAY_CHECK") &&
            !stochastic_verify &&
            !use_sampling_penalties &&
            active_sampling_params_.is_greedy();
        /*
         * The vLLM-style GPU lanes publish from device-resident spec slots and
         * treat the live transaction as atomic. They only need a logical base
         * stamp on the success path; rollback checkpoints are reserved for
         * verifier lanes that still mutate/restore host-owned state.
         *
         * This is deliberately not the same contract as
         * supportsLogicalMTPVerifierBaseCheckpoint().  Hybrid MoE/GDN runners may
         * be unable to restore recurrent payload state from a token-count-only
         * snapshot in replay/debug code, while still being able to publish the
         * just-produced verifier rows directly on device without ever restoring
         * the verifier base on the success path.  Grouped-outcome MoE uses the
         * same success-path contract once it proves sidecar drafting preserves
         * the main state and publishes the accepted rows from compact device
         * metadata.
         */
        const bool grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint =
            use_grouped_outcome_device_resident_publication_verifier &&
            plan_.usesLocalTP() &&
            !runner_->supportsMTPShiftedRowReuseFromSidecar();
        /*
         * LocalTP MoE grouped-outcome publication cannot reuse the first shifted
         * MTP KV row from the sidecar: routed expert state and shifted sidecar
         * KV must be committed from the accepted verifier row.  That initial
         * shifted-row commit needs the verifier-base terminal hidden payload,
         * so a token-count-only logical checkpoint is not a valid base for this
         * lane even though the accepted-state publication itself is resident.
         */
        if (hosted_dynamic_continuation &&
            (verify_sidecar_preserves_main_state || verify_commit_replay_check))
            return fail_without_checkpoint(
                "Hosted dynamic MTP continuation does not support host checkpoint/replay diagnostics");
        // The retained family commits accepted state atomically. There is no
        // external first verifier to snapshot, and admitted capacity is never
        // used as a snapshot copy or communication extent on this path.
        const bool use_device_publication_without_rollback_checkpoint =
            hosted_dynamic_continuation ||
            (use_grouped_outcome_device_resident_publication_verifier &&
            (!stochastic_verify || stochastic_device_verify) &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsDeviceResidentMTPSpecStatePublication() &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            !grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check);
        const bool can_synthesize_verifier_base_checkpoint =
            hosted_dynamic_continuation ||
            (use_grouped_outcome_device_resident_publication_verifier &&
            runner_->supportsMTPSidecarPreservesMainState() &&
            (runner_->supportsLogicalMTPVerifierBaseCheckpoint() ||
             use_device_publication_without_rollback_checkpoint) &&
            !grouped_outcome_localtp_shifted_commit_needs_terminal_hidden_checkpoint &&
            !verify_sidecar_preserves_main_state &&
            !verify_commit_replay_check);

        if (use_device_publication_without_rollback_checkpoint)
        {
            std::string position_error;
            const std::optional<int> base_position =
                currentDecodeTransactionPositionForPlanning(
                    "device-publication transaction base",
                    &position_error);
            if (!base_position)
                return fail_without_checkpoint(position_error);

            transaction_base_cached_tokens = *base_position;
            verifier_base_checkpoint =
                makeLogicalMTPVerifierBaseSnapshot(transaction_base_cached_tokens,architecture()=="qwen4exp" ? MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero : MTPShiftedCacheOrigin::SuccessorPairsOnly);
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "live_prefix_checkpoint_skipped_direct_publication",
                1.0,
                {static_cast<uint64_t>(transaction_base_cached_tokens)},
                "decode",
                {},
                {{"verifier_path", "grouped_decode_equivalent_outcome"}});
        }
        else
        {
            if (!capture_rollback_checkpoint(
                    speculative_draft_count,
                    !use_ready_logits && !use_pending_condition_row ? 1 : 0))
                return fail_without_checkpoint("MTP decode could not capture live prefix state");
            verifier_base_checkpoint = rollback_checkpoint;
        }

        auto join_tokens = [](const std::vector<int32_t> &tokens) -> std::string
        {
            std::ostringstream oss;
            for (size_t i = 0; i < tokens.size(); ++i)
            {
                if (i)
                    oss << ",";
                oss << tokens[i];
            }
            return oss.str();
        };

        enum class StochasticDrawPurpose : int
        {
            Sample = 0,
            Accept = 1,
            Residual = 2,
        };

        auto stochastic_threshold_for_position = [&](
                                                     Sampler &fallback_sampler,
                                                     int logical_position,
                                                     StochasticDrawPurpose purpose)
            -> float
        {
            if (active_sampling_params_.seed == 0)
            {
                return fallback_sampler.random_uniform_01();
            }

            /*
             * Seeded MTP sampling must not depend on when a token is sampled.
             * A ready token may be sampled as a bonus row in step N or as the
             * first token of step N+1.  Keying the threshold by logical output
             * position and purpose makes those two paths equivalent.
             */
            const uint64_t position =
                static_cast<uint64_t>(std::max(0, logical_position));
            constexpr uint64_t kDrawPurposesPerToken = 8;
            const uint64_t offset =
                position * kDrawPurposesPerToken +
                static_cast<uint64_t>(purpose);
            return sampling_math::uniform01(
                static_cast<uint64_t>(active_sampling_params_.seed),
                offset);
        };

        auto sample_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Sample);
        };

        auto accept_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Accept);
        };

        auto residual_threshold_for_position =
            [&](Sampler &fallback_sampler, int logical_position) -> float
        {
            return stochastic_threshold_for_position(
                fallback_sampler,
                logical_position,
                StochasticDrawPurpose::Residual);
        };

        auto validate_mtp_transaction = [&](
                                            const char *path,
                                            const PrefixStateSnapshot &base,
                                            int committed_token_count,
                                            int state_advanced_tokens,
                                            PrefixStateProvenance verifier_source,
                                            bool has_terminal_logits,
                                            bool has_ready_token)
            -> std::optional<std::string>
        {
            if (!base.valid)
                return std::string("MTP transaction base snapshot is invalid");
            if (state_advanced_tokens < 0 ||
                state_advanced_tokens > committed_token_count)
            {
                return std::string("MTP transaction advanced-state token count is outside committed token count");
            }

            MTPCommitValidationOptions options;
            options.require_decode_equivalent_source = true;
            options.require_base_shifted_mtp_kv = false;
            options.require_committed_shifted_mtp_kv = true;
            options.require_terminal_hidden = true;
            options.require_terminal_logits = has_terminal_logits;
            options.require_ready_token = has_ready_token;

            MTPDecodeStateStamp base_stamp = makeMTPStateStamp(
                base,
                std::string(path) + ".base",
                /*has_terminal_hidden=*/true,
                /*has_terminal_logits=*/true,
                /*has_ready_token=*/true);

            MTPDecodeStateStamp committed_stamp;
            committed_stamp.valid = base.valid;
            committed_stamp.logical_tokens =
                base.cached_tokens + state_advanced_tokens;
            committed_stamp.main_kv_tokens = committed_stamp.logical_tokens;
            committed_stamp.shifted_mtp_kv_tokens =
                expectedShiftedMTPTokens(committed_stamp.logical_tokens,base.mtp_shifted_origin);
            committed_stamp.shifted_origin=base.mtp_shifted_origin;
            committed_stamp.position = committed_stamp.logical_tokens;
            committed_stamp.has_terminal_hidden = true;
            committed_stamp.has_terminal_logits = has_terminal_logits;
            committed_stamp.has_ready_token = has_ready_token;
            committed_stamp.provenance = verifier_source;
            committed_stamp.label = std::string(path) + ".committed";

            MTPStateValidationResult validation = validateAtomicMTPCommit(
                base_stamp,
                committed_stamp,
                state_advanced_tokens,
                verifier_source,
                options);
            if (!validation)
            {
                ++mtp_stats_.transaction_validation_failures;
                if (!base_stamp.decodeEquivalent() ||
                    !committed_stamp.decodeEquivalent() ||
                    !isDecodeEquivalent(verifier_source))
                {
                    ++mtp_stats_.unsafe_verifier_state_rejections;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "unsafe_verifier_state_rejections",
                        1.0,
                        "decode",
                        {},
                        {{"path", path},
                         {"source", toString(verifier_source)}});
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "transaction_validation_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"reason", validation.reason},
                     {"source", toString(verifier_source)}});
                return std::string("MTP transaction validation failed on ") +
                       path + ": " + validation.reason;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "transaction_validation_passes",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"committed_tokens", std::to_string(committed_token_count)},
                 {"state_advanced_tokens", std::to_string(state_advanced_tokens)},
                 {"source", toString(verifier_source)}});
            return std::nullopt;
        };

        auto commit_mtp_transaction_outputs = [&](
                                                  const char *path,
                                                  const PrefixStateSnapshot &base,
                                                  const std::vector<int32_t> &tokens,
                                                  std::optional<int32_t> ready_token,
                                                  bool terminal_logits_ready,
                                                  bool is_complete,
                                                  PrefixStateProvenance verifier_source,
                                                  bool state_advanced,
                                                  int state_advanced_token_count = -1,
                                                  int emitted_token_start_index = 0,
                                                  std::optional<int32_t> next_pending_condition_token = std::nullopt,
                                                  std::optional<DeviceResidentLogicalSequenceStateHandle>
                                                      next_pending_condition_resident_state = std::nullopt,
                                                  std::optional<DeviceResidentLogicalSequenceStateHandle>
                                                      ready_condition_resident_state = std::nullopt)
            -> std::optional<std::string>
        {
            if (tokens.empty())
                return std::string("MTP transaction produced no committed tokens");
            if (emitted_token_start_index < 0 ||
                emitted_token_start_index > static_cast<int>(tokens.size()))
            {
                return std::string("MTP transaction emitted-token start is outside committed tokens");
            }
            PerfStatsCollector::ScopedTimer commit_timer(
                "mtp",
                "transaction_output_commit",
                "decode",
                {},
                {{"path", path},
                 {"source", toString(verifier_source)}});

            if (state_advanced)
            {
                const int advanced_tokens =
                    state_advanced_token_count >= 0
                        ? state_advanced_token_count
                        : static_cast<int>(tokens.size());
                if (auto validation_error = validate_mtp_transaction(
                        path,
                        base,
                        static_cast<int>(tokens.size()),
                        advanced_tokens,
                        verifier_source,
                        terminal_logits_ready && !is_complete,
                        ready_token.has_value() && !is_complete))
                {
                    return validation_error;
                }

                if (!publishDecodeTransactionPlanningPositionAfterMTPCommit(
                        base.cached_tokens,
                        advanced_tokens,
                        path))
                {
                    return last_error_.empty()
                               ? std::optional<std::string>{
                                     "MTP transaction could not publish its "
                                     "scheduler-owned position"}
                               : std::optional<std::string>{last_error_};
                }
            }

            pending_mtp_condition_token_ = next_pending_condition_token;
            pending_mtp_condition_params_ =
                next_pending_condition_token.has_value()
                    ? std::optional<SamplingParams>{active_sampling_params_}
                    : std::optional<SamplingParams>{};
            pending_mtp_condition_resident_state_ =
                next_pending_condition_token.has_value()
                    ? next_pending_condition_resident_state
                    : std::nullopt;

            prefill_logits_ready_ = terminal_logits_ready && !is_complete;
            if (prefill_logits_ready_ && ready_token.has_value())
            {
                if (ready_condition_resident_state.has_value() &&
                    !ready_condition_resident_state->valid())
                {
                    return std::string(
                        "MTP transaction produced a stale ready-token resident logical-state handle");
                }
                ready_mtp_condition_ = ReadyMTPCondition::hostVisible(
                    *ready_token,
                    active_sampling_params_,
                    ready_condition_resident_state);
                if (!ready_mtp_condition_->valid())
                {
                    return std::string(
                        "MTP transaction produced an invalid host-visible ready condition");
                }
                pending_mtp_condition_token_.reset();
                pending_mtp_condition_params_.reset();
                pending_mtp_condition_resident_state_.reset();
            }
            else
            {
                ready_mtp_condition_.reset();
            }

            for (size_t i = static_cast<size_t>(emitted_token_start_index);
                 i < tokens.size();
                 ++i)
            {
                const int32_t token = tokens[i];
                sampler_.record_token(token);
                result.tokens.push_back(token);
            }
            last_token_ = tokens.back();
            result.is_complete = result.is_complete || is_complete;
            if (is_complete)
            {
                /*
                 * A first-sidecar prelaunch is intentionally speculative: it
                 * only prepares the next step's draft proposal.  Stop-token
                 * detection still becomes host-visible at the response
                 * boundary, so a completed request must discard any sidecar
                 * that was queued before the compatibility response bridge.
                 */
                if (prelaunched_mtp_first_sidecar_resident_state_.has_value())
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_first_sidecar_prelaunch_discarded_complete",
                        1.0,
                        "decode",
                        {},
                        {{"path", path}});
                }
                prelaunched_mtp_first_sidecar_resident_state_.reset();
                prelaunched_mtp_first_sidecar_params_.reset();
            }

            ++mtp_stats_.transaction_commits;
            PerfStatsCollector::addCounter(
                "mtp",
                "transaction_commits",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"tokens", join_tokens(tokens)},
                 {"emitted_token_start_index",
                  std::to_string(emitted_token_start_index)},
                 {"emitted_tokens",
                  std::to_string(static_cast<int>(tokens.size()) -
                                 emitted_token_start_index)},
                 {"ready_token", ready_token.has_value()
                                     ? std::to_string(*ready_token)
                                     : std::string("none")},
                 {"next_pending_condition_token",
                  next_pending_condition_token.has_value()
                      ? std::to_string(*next_pending_condition_token)
                      : std::string("none")},
                 {"state_advanced", state_advanced ? "true" : "false"},
                 {"state_advanced_tokens",
                  std::to_string(state_advanced
                                     ? (state_advanced_token_count >= 0
                                            ? state_advanced_token_count
                                            : static_cast<int>(tokens.size()))
                                     : 0)},
                 {"complete", is_complete ? "true" : "false"},
                 {"source", toString(verifier_source)}});
            return std::nullopt;
        };

        auto inspect_spec_decode_metadata = [&](
                                                    const char *path,
                                                    const std::string &implementation,
                                                    const MTPSpecDecodeMetadataBatch &metadata,
                                                    bool stopped_on_output,
                                                    const std::string &draft_token_description,
                                                    const std::vector<int32_t> &committed_output_tokens)
            -> std::optional<std::string>
        {
            if (!metadata.ok)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "spec_decode_transaction_metadata_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"implementation", implementation},
                     {"reason", metadata.error}});
                return std::string("MTP spec-decode metadata failed on ") +
                       path + ": " + metadata.error;
            }
            if (metadata.transactions.empty())
                return std::string("MTP spec-decode metadata produced no transaction");

            const MTPSpecDecodeTransaction &tx = metadata.transactions.front();
            if (!tx.ok)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "spec_decode_transaction_metadata_failures",
                    1.0,
                    "decode",
                    {},
                    {{"path", path},
                     {"implementation", implementation},
                     {"reason", tx.error}});
                return std::string("MTP spec-decode transaction metadata failed on ") +
                       path + ": " + tx.error;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "spec_decode_transaction_metadata",
                1.0,
                "decode",
                {},
                {{"path", path},
                 {"implementation", implementation},
                 {"target_query_len", std::to_string(tx.target_query_len)},
                 {"metadata_total_target_query_tokens",
                  std::to_string(metadata.total_target_query_tokens)},
                 {"valid_sampled_count", std::to_string(tx.valid_sampled_count)},
                 {"committed_output_count",
                  std::to_string(metadata.committed_output_counts.front())},
                 {"accepted_state_count",
                  std::to_string(metadata.accepted_state_counts.front())},
                 {"committed_state_row",
                  std::to_string(metadata.committed_state_rows.front())},
                 {"committed_state_index",
                  std::to_string(metadata.committed_state_indices.front())},
                 {"accepted_state_slot_index",
                  std::to_string(metadata.accepted_state_slot_indices.front())},
                 {"bonus_ready_token_row",
                  std::to_string(metadata.bonus_ready_token_rows.front())},
                 {"bonus_ready_token_index",
                  std::to_string(metadata.bonus_ready_token_indices.front())},
                 {"bonus_ready_state_slot_index",
                  std::to_string(metadata.bonus_ready_state_slot_indices.front())},
                 {"accepted_verifier_input_prefix",
                  std::to_string(tx.accepted_speculative_prefix)},
                 {"accepted_mtp_draft_prefix",
                  std::to_string(std::max(0, tx.accepted_speculative_prefix - 1))},
                 {"rejected_token_count", std::to_string(tx.rejected_token_count)},
                 {"token_index_to_sample", std::to_string(tx.token_index_to_sample)},
                 {"next_condition_token", std::to_string(tx.next_condition_token)},
                 {"all_drafts_accepted", tx.allDraftsAccepted() ? "true" : "false"},
                 {"stopped_on_output", stopped_on_output ? "true" : "false"},
                 {"draft_tokens", draft_token_description},
                 {"committed_output_tokens",
                  join_tokens(committed_output_tokens)}});
            return std::nullopt;
        };

        auto validate_spec_decode_transaction = [&](
                                                        const char *path,
                                                        const std::string &implementation,
                                                        const std::vector<int32_t> &draft_tokens_for_tx,
                                                        const std::vector<int32_t> &committed_output_tokens,
                                                        std::optional<int32_t> ready_token,
                                                        bool all_drafts_accepted,
                                                        bool stopped_on_output,
                                                        int accepted_mtp_draft_prefix,
                                                        bool commit_boundary_clipped)
            -> std::optional<std::string>
        {
            if (draft_tokens_for_tx.empty())
                return std::string("MTP spec-decode transaction has no draft tokens");
            if (committed_output_tokens.empty())
                return std::string("MTP spec-decode transaction has no committed output tokens");
            if (!stopped_on_output && all_drafts_accepted && !ready_token.has_value())
                return std::string("MTP spec-decode transaction accepted all drafts without a ready token");

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens =
                static_cast<int>(draft_tokens_for_tx.size());

            MTPSpecDecodeMetadataBatch metadata;
            if (commit_boundary_clipped)
            {
                MTPSpecDecodeAcceptedOutcome boundary_outcome;
                boundary_outcome.request_id = 0;
                boundary_outcome.vocab_size = vocab;
                boundary_outcome.draft_count =
                    static_cast<int>(draft_tokens_for_tx.size());
                boundary_outcome.committed_output_tokens =
                    committed_output_tokens;
                boundary_outcome.commit_boundary_ready_token = ready_token;
                boundary_outcome.accepted_verifier_input_prefix =
                    accepted_mtp_draft_prefix + 1;
                boundary_outcome.target_verifier_state_commit_count =
                    static_cast<int>(committed_output_tokens.size());
                boundary_outcome.all_drafts_accepted = false;
                boundary_outcome.stopped_on_output = false;
                boundary_outcome.commit_boundary_clipped = true;
                metadata = buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
                    metadata_shape,
                    boundary_outcome);
            }
            else
            {
                MTPDecodeCatchupGreedyRequest catchup_request_for_tx;
                catchup_request_for_tx.draft_tokens = draft_tokens_for_tx;
                MTPDecodeCatchupGreedyResult catchup_result_for_tx;
                catchup_result_for_tx.ok = true;
                catchup_result_for_tx.accepted_tokens =
                    committed_output_tokens;
                catchup_result_for_tx.all_speculative_accepted =
                    all_drafts_accepted;
                catchup_result_for_tx.stopped_on_output = stopped_on_output;
                catchup_result_for_tx.accepted_speculative_prefix =
                    accepted_mtp_draft_prefix;
                catchup_result_for_tx.ready_token =
                    ready_token.value_or(kMTPSpecDecodeInvalidToken);
                metadata = buildMTPSpecDecodeMetadataBatchFromGreedyCatchup(
                    metadata_shape,
                    /*request_id=*/0,
                    vocab,
                    catchup_request_for_tx,
                    catchup_result_for_tx);
            }
            return inspect_spec_decode_metadata(
                path,
                implementation,
                metadata,
                stopped_on_output,
                join_tokens(draft_tokens_for_tx),
                committed_output_tokens);
        };

        auto validate_spec_decode_accepted_outcome = [&](
                                                        const char *path,
                                                        const std::string &implementation,
                                                        const MTPSpecDecodeAcceptedOutcome &outcome)
            -> std::optional<std::string>
        {
            if (outcome.draft_count <= 0)
                return std::string("MTP spec-decode accepted outcome has no draft rows");
            if (outcome.committed_output_tokens.empty())
                return std::string("MTP spec-decode accepted outcome has no committed output tokens");
            if (!outcome.stopped_on_output &&
                outcome.all_drafts_accepted &&
                !outcome.bonus_ready_token.has_value())
            {
                return std::string("MTP spec-decode accepted outcome accepted all drafts without a ready token");
            }

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens = outcome.draft_count;

            MTPSpecDecodeMetadataBatch metadata =
                buildMTPSpecDecodeMetadataBatchFromAcceptedOutcome(
                    metadata_shape,
                    outcome);
            return inspect_spec_decode_metadata(
                path,
                implementation,
                metadata,
                outcome.stopped_on_output,
                std::string("device_deferred:") +
                    std::to_string(outcome.draft_count),
                outcome.committed_output_tokens);
        };

        const bool can_defer_main_decode_sync =
            runner_->primaryDeviceId().is_gpu() &&
            !active_sampling_params_.has_penalties() &&
            (active_sampling_params_.is_greedy() || stochastic_device_verify);

        const int32_t condition_token =
            use_pending_condition_row ? *pending_condition_token : last_token_;
        if (use_pending_condition_row)
        {
            /*
             * The pending token was already emitted and recorded when the
             * previous MTP transaction rejected a draft.  Keep the main state
             * at the accepted prefix and let the verifier consume this token as
             * row zero.  The token is already visible to the response stream, so
             * commit code below must start output emission after row zero.
             */
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "pending_condition_verifier_rows",
                1.0,
                {static_cast<uint64_t>(transaction_base_cached_tokens)},
                "decode",
                {},
                {{"token", std::to_string(condition_token)}});
            PerfStatsCollector::addCounter(
                "mtp",
                "condition_forward_skipped_pending_condition",
                1.0,
                "decode");
        }
        else if (!use_ready_logits)
        {
            /*
             * The condition forward is an ordinary serial-decode state
             * advance: it consumes condition_token in the main graph.  MTP's
             * shifted sidecar cache must therefore publish the row derived from
             * the current terminal hidden before that main forward runs.  If we
             * skip this maintenance step, prefix-cache-restored LocalTP/MoE
             * runs leave shifted KV one row behind the main position, and the
             * next direct-emit or verifier publication observes a cache-head
             * mismatch.
             */
            const bool gpu_condition_forward =
                runner_->primaryDeviceId().is_gpu();
            constexpr int kConditionTargetSampleSlot = 0;
            DeviceResidentLogicalSequenceStateHandle condition_state;
            std::optional<int> cpu_condition_sidecar_position;

            if (gpu_condition_forward)
            {
                /*
                 * The compact verifier publisher owns the condition token and
                 * its pre-forward logical position as one event-published
                 * device transaction. Preserve the token D2D before shifted-KV
                 * maintenance consumes that one-shot mailbox. The persistent
                 * target slot then becomes the sole token source for the main
                 * condition graph; no host token or position is uploaded.
                 */
                condition_state =
                    runner_->deviceResidentLogicalSequenceState();
                if (!condition_state.valid())
                {
                    return fail_after_checkpoint(
                        "GPU MTP condition forward requires a device-resident logical-state mailbox");
                }
                if (!runner_
                         ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                             condition_state,
                             /*request_index=*/0,
                             kConditionTargetSampleSlot))
                {
                    return fail_after_checkpoint(
                        "GPU MTP condition forward could not preserve its resident token in the device target slot");
                }
            }
            else
            {
                std::string condition_sidecar_position_error;
                cpu_condition_sidecar_position =
                    currentDecodeTransactionPositionForPlanning(
                        "condition forward shifted commit",
                        &condition_sidecar_position_error);
                if (!cpu_condition_sidecar_position)
                {
                    return fail_after_checkpoint(
                        condition_sidecar_position_error);
                }
            }

            bool condition_shifted_commit_ok = false;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "condition_forward_shifted_commit",
                    "decode");
                condition_shifted_commit_ok = gpu_condition_forward
                    ? runner_
                          ->commitMTPShiftedRowFromDeviceResidentLogicalState(
                              condition_state,
                              /*request_index=*/0,
                              /*already_appended_tokens=*/0,
                              /*allow_speculative_discard=*/true)
                    : runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                          condition_token,
                          /*already_appended_tokens=*/0,
                          /*allow_speculative_discard=*/true,
                          *cpu_condition_sidecar_position);
            }
            if (!condition_shifted_commit_ok)
            {
                return fail_after_checkpoint(
                    "MTP condition-forward shifted-cache maintenance failed");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "condition_forward_shifted_commits",
                1.0,
                "decode");

            bool ok = false;
            {
                if (auto sequence_error = begin_overlay_graph_sequence(
                        /*draft_depth=*/0, "condition forward"))
                {
                    return fail_after_checkpoint(*sequence_error);
                }
                PerfStatsCollector::ScopedTimer timer("mtp", "condition_forward", "decode");
                /*
                 * The condition forward's logits are consumed immediately by a
                 * GPU sampler or distribution-builder.  Arm a one-shot stream
                 * handoff so graph replay can skip the CPU sync boundary and
                * let that consumer enforce ordering on the same stream.
                */
                runner_->setMTPMainDecodeSyncDeferralEnabled(can_defer_main_decode_sync);
                if (gpu_condition_forward)
                {
                    /*
                     * Shifted-KV publication consumed the original logical-state
                     * mailbox. Recompose the condition token from the durable
                     * target slot with the now-current device KV position, then
                     * launch the captured main graph from that typed mailbox.
                     */
                    ok = runner_
                             ->advanceMTPMainConditionFromDeviceTargetSample(
                                 condition_token,
                                 kConditionTargetSampleSlot,
                                 MTPConditionForwardPurpose::SpeculativeContinuation);
                }
                else
                {
                    ok = runner_->forward(&condition_token, 1);
                }
                if (!ok)
                {
                    runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                }
            }
            if (!ok)
                return fail_after_checkpoint("Forward pass failed during MTP condition decode");
            if (!advanceDecodeTransactionPlanningPositionAfterForward(
                    "mtp_condition_forward"))
            {
                return fail_after_checkpoint(last_error_);
            }
            if (can_synthesize_verifier_base_checkpoint)
            {
                std::string position_error;
                const std::optional<int> verifier_base_position =
                    currentDecodeTransactionPositionForPlanning(
                        "verifier-base checkpoint synthesis",
                        &position_error);
                if (!verifier_base_position)
                    return fail_after_checkpoint(position_error);
                verifier_base_checkpoint =
                    makeLogicalMTPVerifierBaseSnapshot(*verifier_base_position,architecture()=="qwen4exp" ? MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero : MTPShiftedCacheOrigin::SuccessorPairsOnly);
                PerfStatsCollector::addCounterWithSequence(
                    "mtp",
                    "capture_verifier_base_prefix_state_skipped_all_position_publication",
                    1.0,
                    {static_cast<uint64_t>(verifier_base_checkpoint.cached_tokens)},
                    "decode",
                    {},
                    {{"verifier_path", "grouped_decode_equivalent_outcome"}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "capture_verifier_base_prefix_state",
                    "decode");
                if (!runner_->ensureMTPCheckpointTerminalHidden())
                {
                    return fail_after_checkpoint(
                        "MTP decode could not materialize verifier base terminal hidden");
                }
                std::string position_error;
                const auto capture_request =
                    current_checkpoint_capture_request(
                        speculative_draft_count,
                        /*preceding_condition_rows=*/0,
                        "verifier-base checkpoint capture",
                        &position_error);
                if (!capture_request)
                    return fail_after_checkpoint(position_error);
                verifier_base_checkpoint =
                    runner_->captureLivePrefixCheckpoint(*capture_request);
            }
            if (!verifier_base_checkpoint.valid)
            {
                return fail_after_checkpoint(
                    "MTP decode could not capture verifier base state after condition forward");
            }
            pending_mtp_condition_token_.reset();
            pending_mtp_condition_params_.reset();
            pending_mtp_condition_resident_state_.reset();
            if (auto sequence_error = retire_overlay_graph_sequence(
                    "condition forward"))
            {
                return fail_after_checkpoint(*sequence_error);
            }
        }
        else if (use_ready_logits)
        {
            PerfStatsCollector::addCounter("mtp", "condition_forward_skipped_ready_logits", 1.0, "decode");
        }

        /*
         * Publish the selected transaction geometry before any response-budget
         * clipping.  This is the backend-neutral execution ledger for fixed
         * depth: CPU executes one host-owned grouped transaction at a time,
         * while CUDA and ROCm hand this same selected/capture geometry to the
         * device-resident generation parent.  Completion evidence is recorded
         * separately by verifier counters, so this record cannot certify a
         * transaction that selected a depth and then failed before execution.
         */
        if (!hosted_dynamic_continuation)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "decode_transaction_depth_selections",
                1.0,
                "decode",
                runner_->primaryDeviceId().toString(),
                {{"depth_policy",
                  mtpDepthPolicyModeToString(mtp.depth_policy.mode)},
                 {"requested_depth",
                  std::to_string(requested_speculative_draft_count)},
                 {"capture_depth",
                  std::to_string(transaction_draft_capacity)}});
            if (materialize_dynamic_generation_loop_this_step)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "dynamic_device_generation_capacity_capture_transactions",
                    1.0,
                    "graph_setup",
                    {},
                    {{"selected_depth",
                      std::to_string(requested_speculative_draft_count)},
                     {"capture_depth",
                      std::to_string(effectiveMTPMaxDraftDepth(mtp))},
                     {"initial_transaction_depth", std::to_string(transaction_draft_capacity)},
                     {"authority", "device_generation_controller"}});
            }
        }
        /*
         * Admit only a transaction that can actually reach grouped verification.
         * A one-token response boundary can collapse the speculative width to
         * zero and return the already device-owned target sample directly. It
         * must not leave behind a resident controller that no graph transaction
         * can complete. This edge remains before every draft sample and verifier
         * launch, so the first compact reducer still consumes an initialized
         * response/controller ledger.
         */
        const bool admit_device_generation_this_step =
            use_grouped_outcome_device_resident_publication_verifier &&
            pre_sample_effective_draft_count > 0;
        if (!admitScalarDeviceResidentGeneration(
                admit_device_generation_this_step,
                first_condition_was_already_emitted
                    ? sampling_math::
                          DeviceGenerationLeadingRowDisposition::AlreadyEmitted
                    : sampling_math::
                          DeviceGenerationLeadingRowDisposition::PendingResponse))
        {
            return fail_after_checkpoint(
                last_error_.empty()
                    ? "GPU grouped MTP generation admission failed"
                    : last_error_);
        }
        constexpr int32_t kDeferredMTPFirstTokenShadow = -3;
        const bool verifier_accepts_device_first_token =
            use_grouped_outcome_device_resident_publication_verifier;
        const bool can_defer_stochastic_first_host_read =
            stochastic_device_verify &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();
        const bool can_defer_greedy_first_host_read =
            !stochastic_verify &&
            use_grouped_outcome_device_resident_publication_verifier &&
            runner_->primaryDeviceId().is_gpu() &&
            pre_sample_effective_draft_count > 0 &&
            stop_tokens_.size() <=
                static_cast<size_t>(
                    sampling_math::kSpeculativeBatchMaxStopTokens) &&
            runner_->supportsMTPDeviceDraftTokenInput();

        int32_t first_token = -1;
        bool first_token_device_target_slot_available = false;
        const bool first_token_is_already_emitted_condition =
            first_condition_was_already_emitted;
        if (use_pending_condition_row)
        {
            first_token = condition_token;
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_pending_condition_rows",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else if (use_ready_logits && ready_condition.has_value() &&
                 ready_condition->host_token.has_value())
        {
            first_token = *ready_condition->host_token;
            PerfStatsCollector::addCounter("mtp", "first_token_ready_cache_hits", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_ready_cache_token",
                1.0,
                "decode",
                {},
                {{"token", std::to_string(first_token)}});
        }
        else if (use_ready_logits && ready_condition.has_value() &&
                 ready_condition->isDeviceResidentOnly())
        {
            /*
             * The terminal generation loop sampled this condition on device
             * after committing the exact previous verifier prefix. Its fused
             * publisher already refreshed canonical target slot zero and the
             * logical-state mailbox. Keep a sentinel only in diagnostic host
             * vectors; every production sidecar/verifier consumer below reads
             * the authenticated device sources.
             */
            if (hosted_dynamic_continuation &&
                !runner_->publishDeviceResidentConditionTokenToTargetSampleSlot(
                    *ready_condition->resident_state,
                    /*request_index=*/0, /*target_sample_slot=*/0))
                return fail_after_checkpoint(
                    "Hosted dynamic MTP continuation could not stage its resident condition in the canonical target slot");
            first_token = kDeferredMTPFirstTokenShadow;
            first_token_device_target_slot_available = true;
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "first_token_resident_continuation_uses",
                1.0,
                {static_cast<uint64_t>(ready_condition->resident_state
                          ->publication_generation)},
                "decode",
                {},
                {{"host_token_materializations", "0"}});
        }
        else
        {
            if (stochastic_verify)
            {
                if (runner_->primaryDeviceId().is_gpu())
                {
                    if (active_sampling_params_.top_k <= 0 ||
                        active_sampling_params_.top_k > 256)
                    {
                        return fail_after_checkpoint(
                            "GPU stochastic MTP sampling requires 1 <= top_k <= 256");
                    }
                    if (!stochastic_device_verify)
                    {
                        return fail_after_checkpoint(
                            "GPU stochastic MTP requires device-resident distribution verification");
                    }
                    if (active_sampling_params_.has_penalties() &&
                        !runner_->applyDeviceOwnedMTPPenaltiesToLogitRows(
                            DeviceLogitsSource::Main,
                            /*row_count=*/1,
                            MTPRequestPenaltyPolicy{
                                .presence_penalty =
                                    active_sampling_params_.presence_penalty,
                                .frequency_penalty =
                                    active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                            }))
                    {
                        return fail_after_checkpoint(
                            "MTP stochastic first-token device-history penalty application failed");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "sample_first_token_stochastic_device",
                            "decode");
                        const int first_token_logical_position =
                            transaction_base_cached_tokens;
                        const float first_token_threshold =
                            sample_threshold_for_position(
                                sampler_,
                                first_token_logical_position);
                        PerfStatsCollector::addCounterWithSequence(
                            "mtp",
                            "first_token_stochastic_draw",
                            1.0,
                            {static_cast<uint64_t>(first_token_logical_position), std::bit_cast<uint64_t>(static_cast<double>(first_token_threshold))},
                            "decode",
                            {},
                            {{"deferred", can_defer_stochastic_first_host_read ? "true" : "false"}});
                        if (!runner_->buildStochasticDistributionOnDevice(
                                DeviceLogitsSource::Main,
                                0,
                                DeviceDistributionBuffer::Target,
                                0,
                                active_sampling_params_,
                                vocab))
                        {
                            return fail_after_checkpoint("MTP stochastic first-token GPU distribution build failed");
                        }
                        if (can_defer_stochastic_first_host_read)
                        {
                            if (!runner_->sampleStochasticDistributionOnDeviceDeferred(
                                    DeviceDistributionBuffer::Target,
                                    0,
                                    first_token_threshold))
                            {
                                return fail_after_checkpoint("MTP stochastic first-token GPU deferred sampling failed");
                            }
                            first_token = kDeferredMTPFirstTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "first_token_stochastic_deferred_host_reads",
                                1.0,
                                "decode");
                        }
                        else
                        {
                            first_token = runner_->sampleStochasticDistributionOnDevice(
                                DeviceDistributionBuffer::Target,
                                0,
                                first_token_threshold);
                        }
                    }
                    if (first_token < 0 &&
                        first_token != kDeferredMTPFirstTokenShadow)
                    {
                        return fail_after_checkpoint("MTP stochastic first-token GPU sampling failed");
                    }
                    first_token_device_target_slot_available = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_stochastic_device_samples",
                        1.0,
                        "decode");
                }
                else
                {
                    const float *main_logits = runner_->logits();
                    if (!main_logits)
                    {
                        return fail_after_checkpoint("No logits available for stochastic MTP first token");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_stochastic", "decode");
                        if (use_serial_sample_equivalent_host_stochastic)
                        {
                            const auto distribution =
                                sampler_.compute_distribution(
                                    main_logits,
                                    static_cast<size_t>(vocab),
                                    active_sampling_params_);
                            const int logical_position =
                                transaction_base_cached_tokens;
                            const float threshold =
                                sample_threshold_for_position(
                                    sampler_,
                                    logical_position);
                            first_token = sampleMTPDistributionWithThreshold(
                                distribution,
                                threshold);
                            PerfStatsCollector::addCounterWithSequence(
                                "mtp",
                                "first_token_stochastic_serial_equivalent_host_samples",
                                1.0,
                                {static_cast<uint64_t>(logical_position), std::bit_cast<uint64_t>(static_cast<double>(threshold)), static_cast<uint64_t>(first_token)},
                                "decode",
                                {},
                                {{"position_owner", "transaction_base_cached_tokens"},
                                 {"source", "cpu_rank_terminal_logits"}});
                        }
                        else
                        {
                            first_token = sampler_.sample(
                                main_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                        }
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_stochastic_samples", 1.0, "decode");
                }
            }
            else
            {
                if (use_sampling_penalties)
                {
                    const bool penalty_ok =
                        runner_->primaryDeviceId().is_gpu()
                            ? runner_->applyDeviceOwnedMTPPenaltiesToLogitRows(
                                  DeviceLogitsSource::Main,
                                  /*row_count=*/1,
                                  MTPRequestPenaltyPolicy{
                                      .presence_penalty =
                                          active_sampling_params_.presence_penalty,
                                      .frequency_penalty =
                                          active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                                  })
                            : runner_->applyPenaltiesOnDevice(
                                  sampler_.compute_penalty_map(
                                      active_sampling_params_,
                                      vocab),
                                  vocab);
                    if (!penalty_ok)
                    {
                        return fail_after_checkpoint(
                            runner_->primaryDeviceId().is_gpu()
                                ? "MTP first-token device-history penalty application failed"
                                : "MTP first-token CPU penalty application failed");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_gpu_penalty_applications",
                        1.0,
                        "decode");
                }
                if (can_defer_greedy_first_host_read)
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "sample_first_token_greedy_device_target_slot",
                        "decode");
                    if (!runner_->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                            /*target_sample_slot=*/0,
                            /*out_token=*/nullptr))
                    {
                        return fail_after_checkpoint(
                            "MTP first-token GPU sampling failed; device target-slot deferred sampling failed; host logits sampling is CPU-only");
                    }
                    first_token = kDeferredMTPFirstTokenShadow;
                    first_token_device_target_slot_available = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_greedy_deferred_host_reads",
                        1.0,
                        "decode");
                }
                else if (verifier_accepts_device_first_token &&
                         runner_->primaryDeviceId().is_gpu() &&
                         runner_->supportsMTPDeviceDraftTokenInput())
                {
                    /*
                     * Keep a persistent device owner even when the request's
                     * remaining output budget clamps speculative depth to zero.
                     * The host shadow is still returned because it is the user
                     * visible result and feeds sampler history, but shifted-MTP
                     * publication and the main-model state advance below both
                     * consume this target slot directly.  Restricting target-slot
                     * sampling to positive draft depth made the final one-token
                     * transaction fall back into the obsolete scalar GPU path.
                     */
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "sample_first_token_greedy_device_target_slot_shadow",
                        "decode");
                    if (!runner_->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                            /*target_sample_slot=*/0,
                            &first_token))
                    {
                        return fail_after_checkpoint(
                            "MTP first-token GPU sampling failed; device target-slot shadow sampling failed; host logits sampling is CPU-only");
                    }
                    first_token_device_target_slot_available = true;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "first_token_greedy_device_target_slot_shadow_samples",
                        1.0,
                        "decode");
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_device", "decode");
                    first_token = runner_->sampleGreedyOnDevice();
                }
                if (first_token < 0 &&
                    first_token != kDeferredMTPFirstTokenShadow)
                {
                    if (use_sampling_penalties)
                    {
                        return fail_after_checkpoint("MTP first-token penalized GPU sampling failed");
                    }
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        return fail_after_checkpoint(
                            "MTP first-token GPU sampling failed; host logits sampling is CPU-only");
                    }
                    PerfStatsCollector::addCounter("mtp", "first_token_cpu_host_samples", 1.0, "decode");
                    const float *main_logits = runner_->logits();
                    if (!main_logits)
                    {
                        return fail_after_checkpoint("No logits available for MTP first draft token");
                    }
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_first_token_host", "decode");
                        first_token = sampleHostLogitsAtLogicalPosition(
                            sampler_,
                            main_logits,
                            static_cast<size_t>(vocab),
                            active_sampling_params_, transaction_base_cached_tokens);
                    }
                }
                else
                {
                    PerfStatsCollector::addCounter("mtp", "first_token_device_samples", 1.0, "decode");
                }
            }
        }

        /**
         * Compose one grouped-verifier token row from device-owned producers.
         *
         * Target sample slot zero is the canonical verifier condition mailbox.
         * The initial sampler writes it directly; accepted-state publication
         * refreshes it in the same captured kernel that publishes a ready bonus
         * or rejected correction token. The verifier therefore has one stable
         * pointer topology across every transaction and cannot reconstruct row
         * zero from a host response shadow. Draft rows always come from the
         * resident draft sample bank beginning at slot zero.
         */
        auto prepare_grouped_gpu_verifier_input_tokens =
            [&](const MTPSpecDecodeVerifierInputPlan &input_plan,
                const char *consumer,
                std::string *error) -> const void *
        {
            auto fail = [&](std::string message) -> const void *
            {
                if (error)
                    *error = std::move(message);
                return nullptr;
            };

            if (!runner_->primaryDeviceId().is_gpu())
            {
                return fail(
                    "Device-resident grouped-verifier token composition requires a GPU runner");
            }
            if (!input_plan.ok ||
                input_plan.total_verifier_input_tokens <= 1)
            {
                return fail(
                    "Device-resident grouped-verifier token composition requires a valid grouped row plan");
            }

            DeviceMTPVerifierInputBatchRequest request{
                .request_id = 0,
                .first_token = -1,
                .first_token_from_device = true,
                .first_target_sample_slot = 0,
                .first_draft_slot = 0,
                .draft_token_count =
                    input_plan.total_verifier_input_tokens - 1,
                .total_verifier_input_tokens =
                    input_plan.total_verifier_input_tokens,
            };

            if (!first_token_device_target_slot_available &&
                (!first_token_resident_state.has_value() ||
                 !first_token_resident_state->valid() ||
                 !first_token_resident_state->coversRequest(0)))
            {
                return fail(
                    "GPU grouped verifier has no device-owned condition transaction for canonical target slot zero");
            }

            const void *tokens =
                runner_->prepareMTPVerifierInputTokenBatchOnDevice(
                    &request,
                    /*request_count=*/1,
                    input_plan.total_verifier_input_tokens);
            if (!tokens)
            {
                return fail(
                    "GPU grouped verifier rejected canonical device target slot zero");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "grouped_verifier_first_token_device_sources",
                1.0,
                "decode",
                {},
                {{"source", "canonical_target_slot"},
                 {"consumer", consumer ? consumer : "unknown"},
                 {"total_tokens",
                  std::to_string(
                      input_plan.total_verifier_input_tokens)}});
            return tokens;
        };

        bool draft_count_budget_limited = false;
        if (decode_step_token_budget_ > 0)
        {
            const int budgeted_speculative_outputs =
                std::max(0, decode_step_token_budget_ -
                                  first_token_output_budget_cost);
            draft_count_budget_limited =
                budgeted_speculative_outputs <
                requested_speculative_draft_count;
            if (draft_count_budget_limited)
            {
                const bool resident_controller_owns_commit_budget =
                    use_grouped_outcome_device_resident_publication_verifier &&
                    runner_->primaryDeviceId().is_gpu() &&
                    budgeted_speculative_outputs > 0;
                if (resident_controller_owns_commit_budget)
                {
                    /*
                     * The device selector names the complete transaction width.
                     * Trimming sidecars here would make host response budget a
                     * second graph-geometry authority and leave captured token
                     * rows narrower than the resident selector. Execute the
                     * selected transaction and let its device commit budget
                     * limit serial-visible output and state publication.
                     */
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "draft_budget_delegated_to_device_controller",
                        1.0,
                        "decode",
                        {},
                        {{"selected_depth",
                          std::to_string(requested_speculative_draft_count)},
                         {"speculative_output_budget",
                          std::to_string(budgeted_speculative_outputs)},
                         {"token_budget",
                          std::to_string(decode_step_token_budget_)}});
                }
                else
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "draft_steps_budget_clamped",
                        1.0,
                        "decode",
                        {},
                        {{"configured", std::to_string(requested_speculative_draft_count)},
                         {"effective", std::to_string(speculative_draft_count)},
                         {"token_budget", std::to_string(decode_step_token_budget_)}});
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "draft_steps_budget_skipped",
                        static_cast<double>(requested_speculative_draft_count - speculative_draft_count),
                        "decode");
                }
            }
        }

        // The compact grouped publisher owns its own history. These direct
        // outputs have no compact outcome, so publish the original resident
        // scalar once before returning it. A carried token explicitly takes
        // the no-write disposition; no response shadow is uploaded or used.
        const auto publish_direct_output_history = [&]() -> bool
        {
            if (!runner_->primaryDeviceId().is_gpu()) return true;
            if (first_token_is_already_emitted_condition)
                return runner_->publishMTPDirectTokenHistory(0,
                    MTPDirectTokenHistoryDisposition::AlreadyPublished);
            if (!first_token_device_target_slot_available)
            {
                if (!first_token_resident_state.has_value() ||
                    !runner_->publishDeviceResidentConditionTokenToTargetSampleSlot(
                        *first_token_resident_state, 0, 0))
                    return false;
                first_token_device_target_slot_available = true;
            }
            return runner_->publishMTPDirectTokenHistory(0,
                MTPDirectTokenHistoryDisposition::NewlyEmitted);
        };

        if (speculative_draft_count == 0)
        {
            PerfStatsCollector::addCounter("mtp", "budget_limited_direct_emits", 1.0, "decode");
            PerfStatsCollector::addCounter("mtp", "output_tokens", 1.0, "decode");

            if (first_token == kDeferredMTPFirstTokenShadow)
            {
                /*
                 * A one-token response budget has no grouped verifier ledger in
                 * which to return this already-sampled condition token. Observe
                 * exactly the compact mailbox row at the public result boundary;
                 * execution below continues to consume the mailbox and target
                 * slot directly, so this host value never becomes an execution
                 * input or a replacement state authority.
                 */
                if (!first_token_resident_state.has_value() ||
                    !first_token_resident_state->coversRequest(0) ||
                    !runner_->observeDeviceResidentNextConditionTokens(
                        *first_token_resident_state,
                        /*request_count=*/1,
                        &first_token) ||
                    first_token < 0 || first_token >= vocab)
                {
                    return fail_after_checkpoint(
                        "MTP GPU budget-limited direct emit could not materialize its terminal resident token");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "budget_limited_direct_emit_terminal_token_materializations",
                    1.0,
                    "decode",
                    runner_->primaryDeviceId().toString(),
                    {{"boundary", "host_result"},
                     {"bytes", std::to_string(sizeof(first_token))},
                     {"execution_authority", "device_mailbox"}});
            }

            const bool first_token_is_stop =
                std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
            if (!publish_direct_output_history())
                return fail_after_checkpoint("Direct MTP output could not publish its original device-token history");
            if (!first_token_is_stop)
            {
                if (auto sequence_error = begin_overlay_graph_sequence(
                        /*draft_depth=*/0,
                        "budget-limited direct main advance"))
                {
                    return fail_after_checkpoint(*sequence_error);
                }
                /*
                 * Direct emit publishes exactly one shifted sidecar row before
                 * the main graph advances with first_token.  Prefix-cache and
                 * device-resident verifier-base paths can expose a host
                 * runner position that is already staged past the sidecar KV
                 * head, so the precondition must be anchored to the snapshot
                 * that owns the terminal hidden row we are about to publish.
                 *
                 * CPU coordinates derive from the checkpoint's model-owned
                 * origin: QFN includes token zero, while successor-only caches
                 * omit it. GPU publication continues to consume only its
                 * device-owned token/position transaction below.
                 */
                bool shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "budget_limited_direct_emit_shifted_commit",
                        "decode");
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        /*
                         * Ready and rejection-correction tokens are owned by the
                         * resident publication mailbox, not by their host response
                         * shadows. Preserve the token in the persistent target
                         * arena before shifted-cache publication consumes that
                         * one-shot mailbox. This is an event-ordered
                         * D2D handoff; the host scalar is never uploaded.
                         */
                        if (first_token_resident_state.has_value() &&
                            !first_token_device_target_slot_available)
                        {
                            if (!runner_
                                     ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                                         *first_token_resident_state,
                                         /*request_index=*/0,
                                         /*target_sample_slot=*/0))
                            {
                                return fail_after_checkpoint(
                                    "MTP GPU budget-limited direct emit could not publish its resident condition token to the device target slot");
                            }
                            first_token_device_target_slot_available = true;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "budget_limited_direct_emit_resident_target_publications",
                                1.0,
                                "decode",
                                {},
                                {{"source",
                                  use_pending_condition_row
                                      ? "pending_condition"
                                      : "ready_token"},
                                 {"transfer", "d2d"}});
                        }
                        if (!first_token_device_target_slot_available)
                        {
                            return fail_after_checkpoint(
                                "MTP GPU budget-limited direct emit has no device-owned target token");
                        }
                        shifted_commit_ok =
                            first_token_resident_state.has_value()
                                ? runner_
                                      ->commitMTPShiftedRowFromDeviceResidentLogicalState(
                                          *first_token_resident_state,
                                          /*request_index=*/0,
                                          /*already_appended_tokens=*/0,
                                          /*allow_speculative_discard=*/true)
                                : runner_
                                      ->commitMTPShiftedRowFromDeviceTargetSample(
                                          /*target_sample_slot=*/0,
                                          /*already_appended_tokens=*/0,
                                          /*allow_speculative_discard=*/true);
                    }
                    else
                    {
                        const int base_sidecar_position = mtpShiftedAppendPosition(
                            snapshotShiftedMTPTokens(verifier_base_checkpoint),
                            verifier_base_checkpoint.mtp_shifted_origin);
                        shifted_commit_ok =
                            runner_->commitMTPShiftedRowFromCurrentTerminalHidden(
                                first_token,
                                /*already_appended_tokens=*/0,
                                /*allow_speculative_discard=*/true,
                                base_sidecar_position);
                    }
                }
                if (!shifted_commit_ok)
                {
                    return fail_after_checkpoint(
                        "MTP budget-limited direct emit shifted-cache commit failed");
                }

                bool advance_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "budget_limited_direct_emit_forward",
                        "decode");
                    /*
                     * Depth-zero MTP still advances the main graph so the
                     * next decode call can consume ready terminal logits.  On
                     * GPU, that next consumer is a device sampler or
                     * distribution builder, so publish the producer stream and
                     * let the consumer preserve ordering without a CPU sync.
                     */
                    runner_->setMTPMainDecodeSyncDeferralEnabled(
                        can_defer_main_decode_sync);
                    if (runner_->primaryDeviceId().is_gpu())
                    {
                        /*
                         * Publish the same target slot consumed by shifted-MTP
                         * maintenance together with the canonical pre-forward KV
                         * position, then consume that typed mailbox in every
                         * participant's captured main graph.
                         */
                        advance_ok =
                            runner_
                                ->advanceMTPMainConditionFromDeviceTargetSample(
                                    first_token,
                                    /*target_sample_slot=*/0,
                                    MTPConditionForwardPurpose::CommittedSerialToken);
                    }
                    else
                    {
                        advance_ok = runner_->forward(&first_token, 1);
                    }
                    if (!advance_ok)
                    {
                        runner_->setMTPMainDecodeSyncDeferralEnabled(false);
                    }
                }
                if (!advance_ok)
                {
                    return fail_after_checkpoint(
                        "MTP budget-limited direct emit state advance failed");
                }
            }

            if (first_token_is_stop)
            {
                if (auto commit_error = commit_mtp_transaction_outputs(
                        "budget_limited_direct_stop",
                        verifier_base_checkpoint,
                        std::vector<int32_t>{first_token},
                        std::nullopt,
                        /*terminal_logits_ready=*/false,
                        /*is_complete=*/true,
                        PrefixStateProvenance::DecodeEquivalent,
                        /*state_advanced=*/false))
                {
                    return fail_after_checkpoint(*commit_error);
                }
            }
            else
            {
                if (auto commit_error = commit_mtp_transaction_outputs(
                        "budget_limited_direct_emit",
                        verifier_base_checkpoint,
                        std::vector<int32_t>{first_token},
                        std::nullopt,
                        /*terminal_logits_ready=*/true,
                        /*is_complete=*/false,
                        PrefixStateProvenance::DecodeEquivalent,
                        /*state_advanced=*/true))
                {
                    return fail_after_checkpoint(*commit_error);
                }
            }
            return result;
        }

        if (hosted_dynamic_continuation)
        {
            const auto mode = stochastic_verify
                ? DeviceGenerationSamplingMode::Stochastic
                : DeviceGenerationSamplingMode::Greedy;
            // The first target scalar above is device-owned in both cases:
            // a carried resident condition or a new sample from ready logits.
            // Preparation binds that source to the retained Full sidecar and
            // authenticates the original native admission ticket (transaction
            // zero). Only its selector may submit the first complete branch.
            if (!first_token_device_target_slot_available)
            {
                // A pending correction may still be owned only by its exact
                // event-published resident handle. Stage that scalar once; a
                // host token shadow is never an admitted execution source.
                if (!first_token_resident_state.has_value() ||
                    !first_token_resident_state->coversRequest(0) ||
                    !runner_->publishDeviceResidentConditionTokenToTargetSampleSlot(
                        *first_token_resident_state,
                        /*request_index=*/0, /*target_sample_slot=*/0))
                    return fail_after_checkpoint(
                        "Hosted dynamic MTP continuation has no device-owned first target scalar");
                first_token_device_target_slot_available = true;
            }
            if (!runner_->prepareDeviceResidentGenerationContinuation(mode) ||
                !runner_->launchDeviceResidentGeneration())
                return fail_after_checkpoint(
                    "Hosted dynamic MTP continuation could not launch its admission-ticket-selected retained family");
            GenerationResult resident_result = finishScalarDeviceResidentGeneration(
                mode, transaction_base_cached_tokens,
                DeviceGenerationFirstTransaction::AdmissionTicket,
                0, 0, std::move(result));
            if (!resident_result.success())
                return fail_after_checkpoint(resident_result.error);
            return resident_result;
        }

        std::optional<PrefixStateSnapshot> verifier_replay_base_checkpoint;
        if (verify_commit_replay_check)
        {
            verifier_replay_base_checkpoint = verifier_base_checkpoint;
        }

        std::string base_position_error;
        const std::optional<int> maybe_base_sidecar_position =
            currentDecodeTransactionPositionForPlanning(
                "speculative sidecar",
                &base_position_error);
        if (!maybe_base_sidecar_position)
            return fail_after_checkpoint(base_position_error);
        const int base_sidecar_position = *maybe_base_sidecar_position;
        bool first_token_is_stop =
            first_token != kDeferredMTPFirstTokenShadow &&
            std::find(stop_tokens_.begin(), stop_tokens_.end(), first_token) != stop_tokens_.end();
        if (first_token_is_stop &&
            !first_token_is_already_emitted_condition)
        {
            /*
             * A host-visible stop token selected from the current target row is
             * already the serial decode answer.  No speculative sidecar row or
             * grouped verifier row may become part of live state, and running
             * them would only add work after the request is known complete.
             * Deferred device-token lanes still flow through the compact
             * outcome reducer because the host cannot inspect the token here.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "first_token_stop_direct_completes",
                1.0,
                "decode",
                {},
                {{"requested_draft_tokens",
                  std::to_string(requested_speculative_draft_count)},
                 {"stochastic_verify", stochastic_verify ? "true" : "false"},
                 {"policy_path",
                  use_grouped_outcome_device_resident_publication_verifier
                      ? "grouped_outcome_device_resident_publication"
                      : "grouped_outcome_host_publication"}});
            PerfStatsCollector::addCounter(
                "mtp",
                "output_tokens",
                1.0,
                "decode");
            if (!publish_direct_output_history())
                return fail_after_checkpoint("Sampled MTP stop could not publish its original device-token history");
            if (auto commit_error = commit_mtp_transaction_outputs(
                    "first_token_stop_direct",
                    verifier_base_checkpoint,
                    std::vector<int32_t>{first_token},
                    std::nullopt,
                    /*terminal_logits_ready=*/false,
                    /*is_complete=*/true,
                    PrefixStateProvenance::DecodeEquivalent,
                    /*state_advanced=*/false))
            {
                return fail_after_checkpoint(*commit_error);
            }
            return result;
        }
        if (auto sequence_error = begin_overlay_graph_sequence(
                speculative_draft_count, "speculative sidecar"))
        {
            return fail_after_checkpoint(*sequence_error);
        }
        std::vector<int32_t> draft_tokens;
        draft_tokens.reserve(static_cast<size_t>(speculative_draft_count) + 1);
        draft_tokens.push_back(first_token);
        Sampler draft_sampler = sampler_;
        if ((stochastic_verify || use_sampling_penalties) &&
            !first_token_is_already_emitted_condition &&
            first_token != kDeferredMTPFirstTokenShadow)
        {
            draft_sampler.record_token(first_token);
        }

        std::vector<PrefixStateSnapshot> sidecar_checkpoints;
        sidecar_checkpoints.reserve(1);
        std::vector<std::vector<SamplingDistributionEntry>> host_mtp_draft_distributions(
            static_cast<size_t>(std::max(0, speculative_draft_count)));
        constexpr int32_t kDeferredMTPDraftTokenShadow = -2;
        std::string mtp_token_sampling_error;
        const bool use_greedy_device_draft_slots =
            use_grouped_outcome_device_resident_publication_verifier &&
            !stochastic_verify &&
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPDeviceDraftTokenInput();

        auto sample_mtp_token = [&](int draft_idx, bool defer_host_read) -> int32_t
        {
            mtp_token_sampling_error.clear();
            int32_t token = -1;
            if (stochastic_verify)
            {
                if (stochastic_device_verify)
                {
                    /*
                     * vLLM's default draft-sample mode is greedy: the draft
                     * side emits only a token, and target-side rejection treats
                     * q as a one-hot distribution. That avoids building full
                     * draft probability rows on every MTP sidecar step while
                     * preserving stochastic target correction semantics.
                     */
                    {
                        PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_stochastic_device", "decode");
                        const float threshold =
                            sample_threshold_for_position(
                                draft_sampler,
                                transaction_base_cached_tokens + 1 + draft_idx);
                        if (defer_host_read)
                        {
                            if (!runner_->sampleStochasticDraftProposalOnDeviceDeferred(
                                    DeviceLogitsSource::MTP,
                                    0,
                                    draft_idx,
                                    active_sampling_params_,
                                    vocab,
                                    threshold))
                            {
                                return -1;
                            }
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_stochastic_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)}});
                            return kDeferredMTPDraftTokenShadow;
                        }
                        token = runner_->sampleStochasticDraftProposalOnDevice(
                            DeviceLogitsSource::MTP,
                            0,
                            draft_idx,
                            active_sampling_params_,
                            vocab,
                            threshold);
                    }
                    if (token < 0)
                    {
                        return -1;
                    }
                    PerfStatsCollector::addCounter("mtp", "mtp_token_stochastic_device_samples", 1.0, "decode");
                    return token;
                }

                const float *mtp_logits = runner_->mtpLogits();
                if (!mtp_logits)
                {
                    return -1;
                }
                auto distribution =
                    draft_sampler.compute_distribution(
                        mtp_logits,
                        static_cast<size_t>(vocab),
                        active_sampling_params_);
                if (draft_idx >= 0 &&
                    draft_idx < static_cast<int>(host_mtp_draft_distributions.size()))
                {
                    host_mtp_draft_distributions[static_cast<size_t>(draft_idx)] =
                        distribution;
                }
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_stochastic", "decode");
                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        const int logical_position =
                            transaction_base_cached_tokens + 1 + draft_idx;
                        const float threshold =
                            sample_threshold_for_position(
                                draft_sampler,
                                logical_position);
                        token = sampleMTPDistributionWithThreshold(
                            distribution,
                            threshold);
                        PerfStatsCollector::addCounterWithSequence(
                            "mtp",
                            "mtp_token_stochastic_position_keyed_host_samples",
                            1.0,
                            {static_cast<uint64_t>(logical_position), std::bit_cast<uint64_t>(static_cast<double>(threshold))},
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                    }
                    else
                    {
                        token = draft_sampler.sample_from_distribution(distribution);
                    }
                }
                PerfStatsCollector::addCounter("mtp", "mtp_token_stochastic_samples", 1.0, "decode");
                return token;
            }

            // Resident greedy slots publish through the captured stage that
            // applies this branch policy before argmax. A separate transform
            // here would double-penalize the first transaction, while its later
            // retained parents see only the captured publication stage.
            if (use_sampling_penalties && !use_greedy_device_draft_slots)
            {
                const bool penalty_ok =
                    runner_->primaryDeviceId().is_gpu()
                        ? runner_->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                              draft_idx,
                              MTPRequestPenaltyPolicy{
                                  .presence_penalty =
                                      active_sampling_params_.presence_penalty,
                                  .frequency_penalty =
                                      active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                              })
                        : runner_->applyPenaltiesToMTPLogitsOnDevice(
                              draft_sampler.compute_penalty_map(
                                  active_sampling_params_,
                                  vocab),
                              vocab);
                if (!penalty_ok)
                {
                    return -1;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "mtp_token_gpu_penalty_applications",
                    1.0,
                    "decode");
            }
            {
                if (use_greedy_device_draft_slots)
                {
                    bool sampled_to_slot = false;
                    {
                        PerfStatsCollector::ScopedTimer timer(
                            "mtp",
                            "sample_mtp_token_greedy_device_slot",
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        int32_t *host_shadow =
                            defer_host_read ? nullptr : &token;
                        sampled_to_slot =
                            runner_->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                                draft_idx,
                                host_shadow);
                    }
                    if (sampled_to_slot && defer_host_read)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "mtp_token_greedy_device_slot_deferred_host_reads",
                            1.0,
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        return kDeferredMTPDraftTokenShadow;
                    }
                    if (sampled_to_slot && token >= 0)
                    {
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "mtp_token_greedy_device_slot_samples",
                            1.0,
                            "decode",
                            {},
                            {{"draft_idx", std::to_string(draft_idx)}});
                        return token;
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "mtp_token_greedy_device_slot_failures",
                        1.0,
                        "decode",
                        {},
                        {{"draft_idx", std::to_string(draft_idx)}});
                    mtp_token_sampling_error =
                        "MTP draft-token GPU sampling failed; device draft-slot sampling failed; host logits sampling is CPU-only";
                    return -1;
                }
                PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_device", "decode");
                token = runner_->sampleGreedyFromMTPLogitsOnDevice();
            }
            if (token >= 0)
            {
                PerfStatsCollector::addCounter("mtp", "mtp_token_device_samples", 1.0, "decode");
                return token;
            }

            if (use_sampling_penalties)
            {
                mtp_token_sampling_error =
                    "MTP draft-token penalized GPU sampling failed";
                return -1;
            }

            if (runner_->primaryDeviceId().is_gpu())
            {
                mtp_token_sampling_error =
                    "MTP draft-token GPU sampling failed; host logits sampling is CPU-only";
                return -1;
            }

            PerfStatsCollector::addCounter("mtp", "mtp_token_cpu_host_samples", 1.0, "decode");
            const float *mtp_logits = runner_->mtpLogits();
            if (!mtp_logits)
            {
                mtp_token_sampling_error = "No MTP logits available";
                return -1;
            }
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sample_mtp_token_host", "decode");
                token = sampleHostLogitsAtLogicalPosition(
                    sampler_,
                    mtp_logits,
                    static_cast<size_t>(vocab),
                    active_sampling_params_, transaction_base_cached_tokens + 1 + draft_idx);
            }
            return token;
        };

        const bool use_sidecar_sample_fusion =
            runner_->supportsMTPSidecarSampleFusion() && !use_sampling_penalties && !stochastic_verify;
        /*
         * Every GPU MTP lane hands sidecar ownership to its next consumer with
         * a stream event. Penalty kernels are ordinary device consumers and
         * republish the same handoff after mutation; they are not a reason to
         * drain the producer stream on the host.
         */
        const bool use_device_resident_sidecar_stream_handoff =
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPSidecarLogitsStreamHandoff();
        const bool use_sidecar_stream_handoff_for_stochastic =
            stochastic_verify &&
            stochastic_device_verify &&
            use_device_resident_sidecar_stream_handoff;
        const bool use_sidecar_stream_handoff_for_grouped_greedy =
            !stochastic_verify &&
            use_grouped_outcome_device_resident_publication_verifier &&
            use_device_resident_sidecar_stream_handoff;
        const bool use_device_draft_token_sidecar =
            runner_->primaryDeviceId().is_gpu() &&
            runner_->supportsMTPDeviceDraftTokenInput() &&
            (stochastic_device_verify || use_greedy_device_draft_slots);
        const bool use_resident_pending_condition_sidecar =
            use_pending_condition_row &&
            pending_condition_has_resident_state &&
            use_device_draft_token_sidecar;
        const bool use_resident_ready_condition_sidecar =
            ready_condition_has_resident_state &&
            use_device_draft_token_sidecar;
        auto prelaunch_matches_resident_state =
            [&](const std::optional<DeviceResidentLogicalSequenceStateHandle> &expected)
        {
            return expected.has_value() &&
                   expected->valid() &&
                   prelaunched_first_sidecar_resident_state.has_value() &&
                   prelaunched_first_sidecar_resident_state->valid() &&
                   prelaunched_first_sidecar_params.has_value() &&
                   samplingParamsEqual(
                       *prelaunched_first_sidecar_params,
                       active_sampling_params_) &&
                   prelaunched_first_sidecar_resident_state->sameMailboxAs(
                       *expected);
        };
        const bool use_prelaunched_first_sidecar =
            (use_sidecar_stream_handoff_for_stochastic ||
             use_sidecar_stream_handoff_for_grouped_greedy) &&
            ((use_resident_ready_condition_sidecar &&
              prelaunch_matches_resident_state(
                  ready_condition->resident_state)) ||
             (use_resident_pending_condition_sidecar &&
              prelaunch_matches_resident_state(pending_condition_resident_state)));
        if (prelaunched_first_sidecar_resident_state.has_value() &&
            !use_prelaunched_first_sidecar)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "stochastic_prelaunched_first_sidecar_dropped",
                1.0,
                "decode",
                {},
                {{"has_ready_resident",
                  use_resident_ready_condition_sidecar ? "true" : "false"},
                 {"has_pending_resident",
                  use_resident_pending_condition_sidecar ? "true" : "false"}});
        }
        const bool can_defer_stochastic_draft_host_reads =
            use_sidecar_stream_handoff_for_stochastic &&
            verifier_accepts_device_first_token;
        const bool can_defer_greedy_draft_host_reads =
            !stochastic_verify &&
            use_greedy_device_draft_slots &&
            verifier_accepts_device_first_token &&
            runner_->primaryDeviceId().is_gpu();
        for (int draft_idx = 0; draft_idx < speculative_draft_count; ++draft_idx)
        {
            bool sidecar_ok = false;
            bool used_prelaunched_first_sidecar = false;
            int32_t mtp_token = -1;
            {
                PerfStatsCollector::ScopedTimer timer("mtp", "sidecar_forward", "decode");
                if (draft_idx == 0)
                {
                    if (use_prelaunched_first_sidecar)
                    {
                        /*
                         * The previous decode step already enqueued this
                         * first sidecar from the same resident mailbox before
                         * flushing host-visible response tokens. Reuse the
                         * pending MTP logits instead of replaying the sidecar
                         * and duplicating shifted-cache work. Greedy lanes
                         * sample those pending logits into the usual device
                         * draft slot below so verifier setup stays device
                         * resident.
                         */
                        sidecar_ok = true;
                        used_prelaunched_first_sidecar = true;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_first_sidecar_prelaunch_reuses",
                            1.0,
                            "decode",
                            {},
                            {{"sampling", stochastic_verify ? "stochastic" : "greedy"}});
                    }
                    else if (use_sidecar_sample_fusion)
                    {
                        const bool defer_fused_sample =
                            can_defer_greedy_draft_host_reads;
                        int32_t *sample_host_shadow =
                            defer_fused_sample ? nullptr : &mtp_token;
                        if (use_resident_ready_condition_sidecar)
                        {
                            sidecar_ok = runner_
                                ->forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(
                                    *ready_condition->resident_state,
                                    /*request_index=*/0,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else if (use_resident_pending_condition_sidecar)
                        {
                            sidecar_ok = runner_
                                ->forwardMTPFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlot(
                                    *pending_condition_resident_state,
                                    /*request_index=*/0,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else if (first_token_device_target_slot_available &&
                                 use_greedy_device_draft_slots)
                        {
                            sidecar_ok =
                                runner_
                                    ->forwardMTPFromDeviceTargetAtLivePositionAndSampleGreedyToDeviceDraftSlot(
                                    /*target_sample_slot=*/0,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else
                        {
                            sidecar_ok =
                                use_greedy_device_draft_slots
                                    ? runner_->forwardMTPAndSampleGreedyToDeviceDraftSlot(
                                          draft_tokens.back(),
                                          draft_idx,
                                          sample_host_shadow)
                                    : runner_->forwardMTPAndSampleGreedy(
                                          draft_tokens.back(),
                                          &mtp_token);
                        }
                        if (sidecar_ok && defer_fused_sample)
                        {
                            mtp_token = kDeferredMTPDraftTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_greedy_device_slot_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)},
                                 {"path", "fused_first_sidecar"}});
                        }
                    }
                    else if (use_sidecar_stream_handoff_for_stochastic)
                    {
                        if (use_resident_ready_condition_sidecar)
                        {
                            /*
                             * The ready token has already been sampled from a
                             * verifier bonus row and will be emitted to the
                             * caller at this step boundary.  The next sidecar
                             * consumes the same token and logical position from
                             * the device mailbox so the hot path does not
                             * round-trip that token through the CPU.
                             */
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                    *ready_condition->resident_state,
                                    /*request_index=*/0);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_resident_ready_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else if (use_resident_pending_condition_sidecar)
                        {
                            /*
                             * The host token is only the response shadow.  The
                             * next sidecar consumes the correction token and
                             * logical position from the mailbox produced by
                             * direct device publication, keeping the fixed
                             * depth path aligned with vLLM's resident
                             * transaction model.
                             */
                            sidecar_ok =
                                runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                    *pending_condition_resident_state,
                                    /*request_index=*/0);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_resident_condition_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else if (first_token_device_target_slot_available)
                        {
                            sidecar_ok =
                                use_device_draft_token_sidecar &&
                                runner_
                                    ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                                        /*target_sample_slot=*/0);
                            if (sidecar_ok)
                            {
                                PerfStatsCollector::addCounter(
                                    "mtp",
                                    "stochastic_first_sidecar_device_target_inputs",
                                    1.0,
                                    "decode");
                            }
                        }
                        else
                        {
                            /*
                             * A GPU sidecar may never reconstruct a device
                             * token/position pair from host shadows. CPU keeps
                             * its ordinary host-owned sidecar contract.
                             */
                            sidecar_ok =
                                runner_->primaryDeviceId().is_gpu()
                                    ? false
                                    : runner_->forwardMTPForDeviceSampling(
                                          draft_tokens.back());
                        }
                    }
                    else if (use_resident_ready_condition_sidecar)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                *ready_condition->resident_state,
                                /*request_index=*/0);
                    }
                    else if (use_resident_pending_condition_sidecar)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                                *pending_condition_resident_state,
                                /*request_index=*/0);
                    }
                    else if (first_token_device_target_slot_available &&
                             use_device_draft_token_sidecar)
                    {
                        sidecar_ok = runner_
                            ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                                /*target_sample_slot=*/0);
                    }
                    else
                    {
                        sidecar_ok =
                            runner_->primaryDeviceId().is_gpu()
                                ? false
                                : runner_->forwardMTP(draft_tokens.back());
                    }
                }
                else
                {
                    if (use_sidecar_sample_fusion)
                    {
                        const bool defer_fused_sample =
                            can_defer_greedy_draft_host_reads;
                        int32_t *sample_host_shadow =
                            defer_fused_sample ? nullptr : &mtp_token;
                        if (use_greedy_device_draft_slots)
                        {
                            /*
                             * The previous draft row was sampled into the
                             * runner-owned device slot with index draft_idx-1.
                             * Consume that slot directly and write the next
                             * proposal to draft_idx.  The compact verifier
                             * outcome will materialize response tokens later,
                             * so no intermediate D2H token copy is required.
                             */
                            sidecar_ok =
                                runner_
                                    ->forwardMTPFromDeviceDraftAtLivePositionAndSampleGreedyToDeviceDraftSlot(
                                    draft_idx - 1,
                                    /*position_offset=*/draft_idx,
                                    draft_idx,
                                    sample_host_shadow);
                        }
                        else
                        {
                            sidecar_ok =
                                use_greedy_device_draft_slots
                                    ? runner_->forwardMTPFromLastDraftAndSampleGreedyToDeviceDraftSlot(
                                          draft_tokens.back(),
                                          base_sidecar_position + draft_idx,
                                          draft_idx,
                                          sample_host_shadow)
                                    : runner_->forwardMTPFromLastDraftAndSampleGreedy(
                                          draft_tokens.back(),
                                          base_sidecar_position + draft_idx,
                                          &mtp_token);
                        }
                        if (sidecar_ok && defer_fused_sample)
                        {
                            mtp_token = kDeferredMTPDraftTokenShadow;
                            PerfStatsCollector::addCounter(
                                "mtp",
                                "mtp_token_greedy_device_slot_deferred_host_reads",
                                1.0,
                                "decode",
                                {},
                                {{"draft_idx", std::to_string(draft_idx)},
                                 {"path", "fused_chained_sidecar"}});
                        }
                    }
                    else if (use_device_draft_token_sidecar)
                    {
                        /*
                         * The previous iteration sampled MTP draft token
                         * draft_idx - 1 into the runner-owned device slot with
                         * the same index. Feed that slot directly into the next
                         * sidecar embedding instead of uploading draft_tokens.back().
                         */
                        sidecar_ok =
                            runner_
                                ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                                draft_idx - 1,
                                /*position_offset=*/draft_idx);
                    }
                    else if (use_sidecar_stream_handoff_for_stochastic)
                    {
                        sidecar_ok =
                            runner_->forwardMTPFromLastDraftForDeviceSampling(
                                draft_tokens.back(),
                                base_sidecar_position + draft_idx);
                    }
                    else
                    {
                        sidecar_ok = runner_->forwardMTPFromLastDraft(
                            draft_tokens.back(),
                            base_sidecar_position + draft_idx);
                    }
                }
            }
            if (!sidecar_ok)
            {
                return fail_after_checkpoint(
                    draft_idx == 0
                        ? "MTP sidecar forward failed"
                        : "Chained MTP sidecar forward failed");
            }
            if (runner_->primaryDeviceId().is_gpu())
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "sidecar_iteration_host_flushes_avoided",
                    1.0,
                    "decode",
                    {},
                    {{"draft_idx", std::to_string(draft_idx)}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "sidecar_iteration_flush",
                    "decode");
                if (!runner_->flushPendingMTPWork())
                {
                    return fail_after_checkpoint("MTP sidecar stream flush failed");
                }
            }
            if (auto fence_error =
                    fence_mpi_mtp_boundary("after_sidecar_iteration_flush"))
            {
                return fail_after_checkpoint(*fence_error);
            }

            if (draft_idx == 0)
            {
                if (use_grouped_decode_equivalent_outcome_verifier &&
                    runner_->supportsMTPSidecarPreservesMainState())
                {
                    /*
                     * Grouped decode-equivalent publication still uses the main
                     * verifier base checkpoint, but graph-native sidecars prove
                     * they do not mutate that base while drafting. Capturing a
                     * post-sidecar payload checkpoint here would only export
                     * hybrid KV/GDN state that grouped publication does not use.
                     */
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "post_sidecar_checkpoint_skipped_sidecar_preserved",
                        1.0,
                        "decode",
                        {},
                        {{"verifier_path",
                          "grouped_decode_equivalent_outcome"}});
                }
                else
                {
                    PerfStatsCollector::ScopedTimer timer("mtp", "capture_post_sidecar_prefix_state", "decode");
                    if (!runner_->ensureMTPCheckpointTerminalHidden())
                    {
                        return fail_after_checkpoint(
                            "MTP decode could not materialize post-sidecar terminal hidden");
                    }
                    std::string position_error;
                    const auto capture_request =
                        current_checkpoint_capture_request(
                            speculative_draft_count,
                            /*preceding_condition_rows=*/0,
                            "post-sidecar checkpoint capture",
                            &position_error);
                    if (!capture_request)
                        return fail_after_checkpoint(position_error);
                    sidecar_checkpoints.push_back(
                        runner_->captureLivePrefixCheckpoint(*capture_request));
                    if (!sidecar_checkpoints.back().valid)
                    {
                        return fail_after_checkpoint("MTP decode could not capture post-sidecar shifted state");
                    }
                }
            }
            else
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "post_sidecar_checkpoint_skipped_speculative",
                    1.0,
                    "decode");
            }

            const bool sidecar_sample_already_done =
                use_sidecar_sample_fusion &&
                !used_prelaunched_first_sidecar;
            if (!sidecar_sample_already_done)
            {
                const bool next_sidecar_needs_host_token =
                    draft_idx + 1 < speculative_draft_count &&
                    !use_device_draft_token_sidecar;
                const bool greedy_verifier_can_consume_device_token =
                    can_defer_greedy_draft_host_reads &&
                    use_device_draft_token_sidecar;
                const bool defer_draft_host_read =
                    (can_defer_stochastic_draft_host_reads &&
                     !next_sidecar_needs_host_token) ||
                    greedy_verifier_can_consume_device_token;
                mtp_token = sample_mtp_token(draft_idx, defer_draft_host_read);
            }
            if (mtp_token < 0)
            {
                if (mtp_token != kDeferredMTPDraftTokenShadow)
                    return fail_after_checkpoint(
                        mtp_token_sampling_error.empty()
                            ? "No MTP logits available"
                            : mtp_token_sampling_error);
            }
            if (sidecar_sample_already_done)
            {
                if (use_greedy_device_draft_slots)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "mtp_token_greedy_device_slot_samples",
                        1.0,
                        "decode",
                        {},
                        {{"draft_idx", std::to_string(draft_idx)}});
                }
                PerfStatsCollector::addCounter("mtp", "mtp_token_device_samples", 1.0, "decode");
            }
            draft_tokens.push_back(mtp_token);
            if ((stochastic_verify || use_sampling_penalties) &&
                mtp_token != kDeferredMTPDraftTokenShadow)
            {
                draft_sampler.record_token(mtp_token);
            }

            ++mtp_stats_.draft_steps;
            PerfStatsCollector::addCounter("mtp", "draft_steps", 1.0, "decode");
        }

        /*
         * vLLM-style stochastic verification can keep every draft token
         * sidecar-resident: the verifier input row is materialized later on
         * the verifier graph stream, and that materialization waits on the
         * target/draft sample-ready events before copying device tokens. In
         * that lane, synchronizing the sidecar stream here is pure host-side
         * performance debt. Host-token and debug-preservation paths still need
         * the flush because they inspect sidecar-produced state immediately.
         */
        const bool can_skip_sidecar_flush_before_verifier =
            runner_->primaryDeviceId().is_gpu() &&
            use_device_resident_sidecar_stream_handoff;
        if (can_skip_sidecar_flush_before_verifier)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_final_flush_skipped_device_verifier_tokens",
                1.0,
                "decode",
                {},
                {{"draft_tokens", std::to_string(draft_tokens.size())},
                 {"first_token_deferred",
                  first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"}});
        }
        else
        {
            PerfStatsCollector::ScopedTimer timer(
                "mtp",
                "sidecar_final_flush_before_verification",
                "decode");
            if (!runner_->flushPendingMTPWork())
            {
                return fail_after_checkpoint("MTP sidecar stream flush failed before verification");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "sidecar_final_flush_required_before_verification",
                1.0,
                "decode",
                {},
                {{"stochastic_verify", stochastic_verify ? "true" : "false"},
                 {"stochastic_device_verify",
                  stochastic_device_verify ? "true" : "false"},
                 {"has_penalties",
                  active_sampling_params_.has_penalties() ? "true" : "false"},
                 {"verifier_path", "grouped_decode_equivalent_outcome"}});
        }
        if (auto fence_error =
                fence_mpi_mtp_boundary("before_target_verifier"))
        {
            return fail_after_checkpoint(*fence_error);
        }

        if (DebugEnv::isTruthyEnv("LLAMINAR_MTP_VERIFY_SIDECAR_PRESERVES_MAIN_STATE"))
        {
            auto join_debug_tokens = [](const std::vector<int32_t> &tokens) -> std::string
            {
                std::ostringstream oss;
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    if (i)
                        oss << ",";
                    oss << tokens[i];
                }
                return oss.str();
            };

            if (!runner_->ensureMTPCheckpointTerminalHidden())
            {
                return fail_after_checkpoint(
                    "MTP sidecar preservation check could not materialize terminal hidden");
            }
            PrefixStateSnapshot sidecar_state = runner_->captureLivePrefixState();
            if (!sidecar_state.valid)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not capture sidecar state");
            }
            /*
             * Row-token equality alone can miss the coherence bug class that
             * motivated Phase 9.5: sidecar replay may leave host-visible rows
             * correct while mutating main KV, GDN/short-conv state, or logical
             * positions.  Compare the runtime state surface that the next
             * verifier and prefix-cache probes observe.  Shifted MTP KV is
             * intentionally excluded here; MoE still publishes accepted shifted
             * rows from verifier rows rather than reusing the sidecar draft row.
             */
            const PrefixRuntimeStateSnapshot sidecar_runtime_state =
                runner_->prefixStateProbe();

            auto verifier_rows_from_current_state = [&]()
                -> std::optional<std::vector<int32_t>>
            {
                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                    return std::nullopt;
                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                    runner_.get(),
                    verifier_input_plan);
                if (!verifier_plan_scope.installed())
                    return std::nullopt;
                const void *verifier_input_tokens_device = nullptr;
                if (runner_->primaryDeviceId().is_gpu())
                {
                    /*
                     * This diagnostic executes the same verifier graph as the
                     * production transaction. GPU verifier metadata therefore
                     * has the same ownership contract: token IDs and their
                     * position row are composed in persistent runner storage,
                     * after waiting for the target/draft sample producers.
                     * Calling the host-token forward entrypoint here leaves the
                     * position row unpublished and no longer represents a legal
                     * GPU verifier invocation.
                     */
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "sidecar_preservation_probe",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        LOG_ERROR("[OrchestrationRunner] "
                                  << preparation_error);
                        return std::nullopt;
                    }
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(true, verifier_row_count))
                    return std::nullopt;
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                MTPVerifierForwardExecutionOptions forward_options;
                forward_options.device_token_ids =
                    verifier_input_tokens_device;
                const MTPVerifierForwardExecutionResult forward_result =
                    executeMTPSpecVerifierForward(
                        *runner_,
                        verifier_input_plan,
                        forward_options);
                if (!forward_result.ok)
                {
                    runner_->setComputeAllPositionLogits(false);
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return std::nullopt;
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                    return std::nullopt;
                std::vector<int32_t> rows(
                    static_cast<size_t>(verifier_row_count),
                    -1);
                if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                        0,
                        static_cast<int>(rows.size()),
                        rows.data()))
                {
                    return std::nullopt;
                }
                return rows;
            };

            std::optional<std::vector<int32_t>> sidecar_rows =
                verifier_rows_from_current_state();
            if (!sidecar_rows)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not sample sidecar verifier rows");
            }
            if (!runner_->restoreLivePrefixState(verifier_base_checkpoint))
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not restore verifier base checkpoint");
            }
            const PrefixRuntimeStateSnapshot base_runtime_state =
                runner_->prefixStateProbe();
            MTPRuntimeSnapshotComparisonOptions sidecar_state_compare;
            sidecar_state_compare.compare_shifted_mtp_kv = false;
            const MTPStateValidationResult sidecar_state_result =
                compareMTPRuntimeStateSnapshots(
                    base_runtime_state,
                    sidecar_runtime_state,
                    sidecar_state_compare);
            if (!sidecar_state_result)
            {
                return fail_after_checkpoint(
                    "MTP sidecar mutated main runtime state: " +
                    sidecar_state_result.reason);
            }
            std::optional<std::vector<int32_t>> base_rows =
                verifier_rows_from_current_state();
            if (!base_rows)
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not sample base verifier rows");
            }
            if (!runner_->restoreLivePrefixState(sidecar_state))
            {
                return fail_after_checkpoint("MTP sidecar preservation check could not restore sidecar state");
            }
            if (*sidecar_rows != *base_rows)
            {
                return fail_after_checkpoint(
                    "MTP sidecar mutated main verifier state: condition_token=" +
                    std::to_string(condition_token) +
                    " first_token=" + std::to_string(first_token) +
                    " draft_tokens=" + join_debug_tokens(draft_tokens) +
                    " sidecar_rows=" + join_debug_tokens(*sidecar_rows) +
                    " base_rows=" + join_debug_tokens(*base_rows) +
                    " used_ready_logits=" + (use_ready_logits ? std::string("true") : std::string("false")));
            }
        }

        auto verify_committed_prefix_replay = [&](
                                                  const char *path,
                                                  const std::vector<int32_t> &tokens_to_replay,
                                                  int32_t expected_next_token,
                                                  int state_advanced_token_count = -1,
                                                  const std::string &debug_context = {})
            -> std::optional<std::string>
        {
            return verifyCommittedPrefixReplay(
                verify_commit_replay_check,
                use_ready_logits,
                mtp,
                condition_token,
                base_sidecar_position,
                verifier_replay_base_checkpoint,
                path,
                tokens_to_replay,
                expected_next_token,
                state_advanced_token_count,
                debug_context);
        };

        const MTPVisibleStateCommitPlan visible_state_commit_plan =
            planMTPVisibleStateCommit(
                static_cast<int>(draft_tokens.size()),
                first_token_is_already_emitted_condition ? 1 : 0,
                decode_step_token_budget_);
        if (!visible_state_commit_plan)
        {
            return fail_after_checkpoint(
                std::string("MTP visible-state commit planning failed: ") +
                visible_state_commit_plan.reason);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "visible_state_commit_plans",
            1.0,
            "decode",
            {},
            {{"verifier_rows",
              std::to_string(visible_state_commit_plan.verifier_input_rows)},
             {"max_state_commit_rows",
              std::to_string(visible_state_commit_plan.max_state_commit_rows)},
             {"output_budget",
              std::to_string(
                  visible_state_commit_plan.remaining_output_budget)},
             {"pending_condition_input",
              first_token_is_already_emitted_condition ? "true" : "false"},
             {"response_boundary_clipped",
              visible_state_commit_plan.response_boundary_clipped
                  ? "true"
                  : "false"}});

        if (use_grouped_outcome_host_publication_verifier)
        {
            constexpr const char *verifier_publication_path =
                "grouped_decode_equivalent_host_publication";
            constexpr const char *verifier_replay_check_path =
                verifier_publication_path;
            const bool sidecar_preserves_main_state =
                runner_->supportsMTPSidecarPreservesMainState();
            bool restored_verifier_base = sidecar_preserves_main_state;
            if (sidecar_preserves_main_state)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "mtp",
                    "all_position_verifier_base_restore_skipped_sidecar_preserved",
                    1.0,
                    {static_cast<uint64_t>(verifier_base_checkpoint.cached_tokens)},
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "all_position_verifier_restore_base_checkpoint",
                    "decode");
                restored_verifier_base =
                    runner_->restoreLivePrefixState(verifier_base_checkpoint);
                if (restored_verifier_base)
                {
                    PerfStatsCollector::addCounterWithSequence(
                        "mtp",
                        "all_position_verifier_base_restores",
                        1.0,
                        {static_cast<uint64_t>(verifier_base_checkpoint.cached_tokens)},
                        "decode",
                        {},
                        {{"draft_tokens", std::to_string(draft_tokens.size())}});
                }
            }
            if (!restored_verifier_base)
            {
                return fail_after_checkpoint(
                    "All-position MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            bool first_token_is_stop =
                first_token != kDeferredMTPFirstTokenShadow &&
                std::find(stop_tokens_.begin(),
                          stop_tokens_.end(),
                          first_token) != stop_tokens_.end();
            /*
             * The first sidecar draft can only be kept as live shifted-MTP
             * state when the backend explicitly proves that its sidecar row is
             * equivalent to the first accepted target row.  Otherwise the
             * shifted cache is published only from accepted verifier rows below.
             * Synthesizing row zero from the host-visible first token is not
             * decode-equivalent for MoE: a rejected correction is an output
             * token, not yet a condition-token state boundary.
             */
            const bool first_shifted_row_available_from_sidecar =
                sidecar_preserves_main_state &&
                runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                !first_token_is_stop;
            if (first_shifted_row_available_from_sidecar)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_reused_sidecar_rows",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"first_token_deferred",
                      first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"}});
            }
            else if (!first_token_is_stop)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_deferred_to_verifier_rows",
                    1.0,
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"first_token_deferred",
                      first_token == kDeferredMTPFirstTokenShadow ? "true" : "false"},
                     {"sidecar_preserves_main_state",
                      sidecar_preserves_main_state ? "true" : "false"}});
            }

            const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                buildSingleRequestVerifierInputPlan(draft_tokens);
            if (!verifier_input_plan.ok)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier input metadata failed: ") +
                    verifier_input_plan.error);
            }
            if (!verifierInputPlanHasCompactRows(verifier_input_plan))
            {
                return fail_after_checkpoint(
                    "All-position MTP verifier row metadata is malformed");
            }

            std::vector<int32_t> sampled_verifier_rows(
                static_cast<size_t>(verifier_input_plan.compact_logit_row_count),
                -1);
            auto apply_all_position_row_penalties_for_history =
                [&](int compare_rows,
                    int bonus_row,
                    const char *counter_name) -> bool
            {
                if (!active_sampling_params_.has_penalties())
                    return true;
                if (first_token == kDeferredMTPFirstTokenShadow)
                    return false;

                /*
                 * Row i is consumed only when every previous speculative row
                 * accepted.  Its sampler history is therefore deterministic:
                 * request base history, first target token, then
                 * draft[1..i-1].  Mutating the row on the verifier stream lets
                 * compact greedy/stochastic reducers see the exact same logits
                 * as serial decode without reading full rows back to host.
                 */
                Sampler row_penalty_sampler = sampler_;
                row_penalty_sampler.record_token(first_token);
                for (int row = 0; row < compare_rows; ++row)
                {
                    auto penalty_map =
                        row_penalty_sampler.compute_penalty_map(
                            active_sampling_params_,
                            vocab);
                    if (!penalty_map.empty() &&
                        !runner_->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                            row,
                            penalty_map,
                            vocab))
                    {
                        return false;
                    }
                    row_penalty_sampler.record_token(
                        draft_tokens[static_cast<size_t>(row + 1)]);
                }

                auto bonus_penalty_map =
                    row_penalty_sampler.compute_penalty_map(
                        active_sampling_params_,
                        vocab);
                if (!bonus_penalty_map.empty() &&
                    !runner_->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                        bonus_row,
                        bonus_penalty_map,
                        vocab))
                {
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    counter_name,
                    static_cast<double>(compare_rows + 1),
                    "decode",
                    {},
                    {{"verifier_path", verifier_publication_path}});
                return true;
            };
            /*
             * This branch is the CPU ownership domain. CUDA and ROCm are
             * admitted only through the grouped device-resident branch below,
             * where verifier publication and continuation remain on device.
             * Keeping a deferred-stream or compact-device option here would
             * recreate the retired host-outcome bridge behind an unreachable
             * condition.
             */
            if (runner_->primaryDeviceId().is_gpu())
            {
                return fail_after_checkpoint(
                    "Grouped host publication is CPU-only");
            }
            if (first_token == kDeferredMTPFirstTokenShadow ||
                std::find(draft_tokens.begin(),
                          draft_tokens.end(),
                          kDeferredMTPDraftTokenShadow) != draft_tokens.end())
            {
                return fail_after_checkpoint(
                    "Grouped host publication received a device-owned token shadow");
            }
            {
                PerfStatsCollector::ScopedTimer verifier_timer(
                    "mtp",
                    "verifier_forward",
                    "decode",
                    {},
                    {{"implementation", verifier_publication_path},
                     {"verifier_path", verifier_publication_path}});
                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                // The verifier forward still consumes every draft token so KV,
                // GDN, and MoE state publication can see the full sequence.
                // Row-indexed logits only shrink the LM-head projection rows.
                ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                    runner_.get(),
                    verifier_input_plan);
                if (!verifier_plan_scope.installed())
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not install row metadata plan");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(
                        true,
                        verifier_row_count))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not enable row-indexed logits");
                }
                if (!runner_->setComputeAllPositionLogits(true))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not enable all-position logits");
                }
                MTPVerifierForwardExecutionOptions verifier_forward_options;
                const MTPVerifierForwardExecutionResult verifier_forward =
                    executeMTPSpecVerifierForward(
                        *runner_,
                        verifier_input_plan,
                        verifier_forward_options);
                if (!verifier_forward.ok)
                {
                    runner_->setComputeAllPositionLogits(false);
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        std::string("All-position MTP verifier forward failed: ") +
                        verifier_forward.error);
                }
                /*
                 * Rank-level LocalTP row sampling consumes the verifier replay
                 * stream through LogitsLocalInfo.  Do that while the
                 * row-indexed all-position logits contract is still active:
                 * disabling the graph mode first is allowed to rebind or clear
                 * the active verifier-logit view, which turns a valid remote
                 * shard winner into the default token 0.  Compact device
                 * outcome reducers are excluded because their captured
                 * penalty-aware argmax already owns row transformation and
                 * sampling on the verifier stream.
                 */
                if (!stochastic_verify && use_sampling_penalties)
                {
                    const int compare_rows =
                        static_cast<int>(draft_tokens.size()) - 1;
                    const int bonus_row = compare_rows;
                    if (!apply_all_position_row_penalties_for_history(
                            compare_rows,
                            bonus_row,
                            "greedy_vllm_penalty_rows_preapplied"))
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "All-position greedy MTP row penalty application failed before cleanup");
                    }
                }
                if (!stochastic_verify)
                {
                    if (!runner_->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                            0,
                            static_cast<int>(sampled_verifier_rows.size()),
                            sampled_verifier_rows.data()))
                    {
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        return fail_after_checkpoint(
                            "All-position MTP verifier could not sample verifier rows before cleanup");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "greedy_verifier_rows_sampled_before_all_position_cleanup",
                        static_cast<double>(sampled_verifier_rows.size()),
                        "decode",
                        {},
                        {{"verifier_path", verifier_publication_path}});
                }
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not disable all-position logits");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier could not disable row-indexed logits");
                }
            }

            MTPDecodeCatchupGreedyRequest catchup_request;
            catchup_request.draft_tokens = draft_tokens;
            catchup_request.stop_tokens = stop_tokens_;
            catchup_request.base_sidecar_position = base_sidecar_position;
            catchup_request.allow_speculative_discard = true;
            catchup_request.verifier_path = verifier_publication_path;
            catchup_request.implementation_name = verifier_publication_path;
            catchup_request.verifier_base_checkpoint = &verifier_base_checkpoint;

            Sampler all_position_stochastic_penalty_sampler = sampler_;
            MTPDecodeCatchupGreedyResult catchup;
            if (stochastic_verify)
            {
                if (!stochastic_host_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped host stochastic MTP requires host distribution verification");
                }
                std::vector<MTPRejectionSampleRowResult> stochastic_rows;
                stochastic_rows.reserve(
                    draft_tokens.size() > 0 ? draft_tokens.size() - 1 : 0);
                bool stochastic_stopped_on_output = false;
                std::optional<int32_t> bonus_ready_token;
                all_position_stochastic_penalty_sampler.record_token(first_token);

                if (std::find(stop_tokens_.begin(),
                              stop_tokens_.end(),
                              first_token) != stop_tokens_.end())
                {
                    stochastic_stopped_on_output = true;
                }

                std::vector<SamplingDistributionEntry> host_target_distribution;
                auto build_all_position_target_distribution =
                    [&](int row) -> bool
                {
                    const float *all_position_logits =
                        runner_->getAllPositionLogits();
                    if (!all_position_logits || row < 0)
                        return false;

                    const float *row_logits =
                        all_position_logits +
                        static_cast<size_t>(row) * static_cast<size_t>(vocab);
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_stochastic_host_target_distribution",
                        "decode",
                        {},
                        {{"implementation",
                          "grouped_decode_equivalent_host_publication"}});
                    host_target_distribution =
                        all_position_stochastic_penalty_sampler
                            .compute_distribution(
                                row_logits,
                                static_cast<size_t>(vocab),
                                active_sampling_params_);
                    return !host_target_distribution.empty();
                };

                for (int draft_idx = 1;
                     !stochastic_stopped_on_output &&
                     draft_idx < static_cast<int>(draft_tokens.size());
                     ++draft_idx)
                {
                    const int row = draft_idx - 1;
                    if (!build_all_position_target_distribution(row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP target distribution build failed");
                    }

                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(draft_idx)];
                    const int logical_position =
                        transaction_base_cached_tokens + draft_idx;
                    MTPRejectionSampleRowResult row_result;
                    if (row < 0 ||
                        row >= static_cast<int>(host_mtp_draft_distributions.size()) ||
                        host_mtp_draft_distributions[static_cast<size_t>(row)].empty() ||
                        host_target_distribution.empty())
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP host verifier missing distributions");
                    }
                    const auto &draft_distribution =
                        host_mtp_draft_distributions[static_cast<size_t>(row)];
                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        const float sample_threshold =
                            sample_threshold_for_position(
                                sampler_,
                                logical_position);
                        row_result = sampleMTPSerialEquivalentTargetRow(
                            host_target_distribution,
                            draft_token,
                            sample_threshold);
                    }
                    else
                    {
                        const float accept_threshold =
                            accept_threshold_for_position(
                                sampler_,
                                logical_position);
                        const float residual_threshold =
                            residual_threshold_for_position(
                                sampler_,
                                logical_position);
                        row_result = sampleMTPRejectionRowFromDistributions(
                            host_target_distribution,
                            draft_distribution,
                            draft_token,
                            accept_threshold,
                            residual_threshold);
                    }
                    if (!row_result.ok)
                    {
                        return fail_after_checkpoint(
                            std::string("All-position stochastic MTP verifier row failed: ") +
                            row_result.error);
                    }

                    ++mtp_stats_.stochastic_accept_tests;
                    PerfStatsCollector::addCounterWithSequence(
                        "mtp",
                        "stochastic_accept_tests",
                        1.0,
                        {static_cast<uint64_t>(draft_token), std::bit_cast<uint64_t>(static_cast<double>(row_result.accept_probability)), std::bit_cast<uint64_t>(static_cast<double>(row_result.accept_threshold))},
                        "decode",
                        {},
                        {{"row", std::to_string(row)},
                         {"device_resident", "false"},
                         {"stochastic_coupling",
                          use_serial_sample_equivalent_host_stochastic
                              ? "serial_sample_equivalent"
                              : "vllm_probability_rejection"},
                         {"verifier_path",
                          "grouped_decode_equivalent_host_publication"}});

                    if (use_serial_sample_equivalent_host_stochastic)
                    {
                        PerfStatsCollector::addCounterWithSequence(
                            "mtp",
                            "stochastic_serial_equivalent_host_verifier_rows",
                            1.0,
                            {static_cast<uint64_t>(logical_position)},
                            "decode",
                            {},
                            {{"row", std::to_string(row)}});
                    }

                    const int32_t output_token = row_result.token;
                    stochastic_rows.push_back(row_result);
                    if (row_result.accepted)
                    {
                        sampled_verifier_rows[static_cast<size_t>(row)] =
                            draft_token;
                        ++mtp_stats_.stochastic_accepts;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "stochastic_accepts",
                            1.0,
                            "decode",
                            {},
                            {{"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
                    }
                    else
                    {
                        if (output_token < 0)
                        {
                            return fail_after_checkpoint(
                                "All-position stochastic MTP verifier produced no correction token");
                        }
                        sampled_verifier_rows[static_cast<size_t>(row)] =
                            output_token;
                        ++mtp_stats_.stochastic_residual_samples;
                        PerfStatsCollector::addCounterWithSequence(
                            "mtp",
                            use_serial_sample_equivalent_host_stochastic
                                ? "stochastic_serial_equivalent_correction_host_samples"
                                : "stochastic_residual_host_samples",
                            1.0,
                            {static_cast<uint64_t>(draft_token), static_cast<uint64_t>(output_token)},
                            "decode",
                            {},
                            {{"row", std::to_string(row)},
                             {"stochastic_coupling",
                              use_serial_sample_equivalent_host_stochastic
                                  ? "serial_sample_equivalent"
                                  : "vllm_probability_rejection"},
                             {"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
                    }

                    all_position_stochastic_penalty_sampler.record_token(output_token);
                    if (std::find(stop_tokens_.begin(),
                                  stop_tokens_.end(),
                                  output_token) != stop_tokens_.end())
                    {
                        stochastic_stopped_on_output = true;
                        break;
                    }
                    if (!row_result.accepted)
                        break;
                }

                const bool all_rows_verified =
                    stochastic_rows.size() + 1 == draft_tokens.size();
                const bool all_rows_accepted =
                    std::all_of(
                        stochastic_rows.begin(),
                        stochastic_rows.end(),
                        [](const MTPRejectionSampleRowResult &row)
                        {
                            return row.accepted;
                        });

                if (!stochastic_stopped_on_output &&
                    all_rows_verified &&
                    all_rows_accepted)
                {
                    const int bonus_row =
                        static_cast<int>(draft_tokens.size()) - 1;
                    if (!build_all_position_target_distribution(bonus_row))
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP bonus distribution build failed");
                    }
                    const int32_t ready_token =
                        sampleMTPDistributionWithThreshold(
                            host_target_distribution,
                            sample_threshold_for_position(
                                sampler_,
                                transaction_base_cached_tokens +
                                    static_cast<int>(draft_tokens.size())));
                    if (ready_token < 0)
                    {
                        return fail_after_checkpoint(
                            "All-position stochastic MTP bonus ready-token sampling failed");
                    }
                    bonus_ready_token = ready_token;
                    sampled_verifier_rows[static_cast<size_t>(bonus_row)] =
                        ready_token;
                    ++mtp_stats_.stochastic_terminal_samples;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "stochastic_terminal_host_samples",
                        1.0,
                        "decode",
                            {},
                            {{"verifier_path",
                              "grouped_decode_equivalent_host_publication"}});
                }
                catchup = buildAllPositionMTPDecodeCatchupStochasticResult(
                    catchup_request,
                    stochastic_rows,
                    bonus_ready_token);
            }
            else
            {
                catchup = buildAllPositionMTPDecodeCatchupGreedyResult(
                    catchup_request,
                    sampled_verifier_rows);
            }
            if (!catchup.ok)
                return fail_after_checkpoint(catchup.error);

            MTPSpecDecodeMetadataShape metadata_shape;
            metadata_shape.max_requests = 1;
            metadata_shape.max_draft_tokens =
                static_cast<int>(draft_tokens.size());
            const int32_t verifier_base_cached_tokens =
                static_cast<int32_t>(verifier_base_checkpoint.cached_tokens);
            MTPSpecTransactionBatchPlan transaction_plan;
            {
                PerfStatsCollector::ScopedTimer transaction_plan_timer(
                    "mtp",
                    "all_position_transaction_plan_build",
                    "decode",
                    {},
                    {{"source", "grouped_host_catchup"}});
                transaction_plan =
                    buildMTPSpecTransactionBatchPlanFromGreedyCatchup(
                        metadata_shape,
                        /*request_id=*/0,
                        vocab,
                        catchup_request,
                        catchup,
                        verifier_base_cached_tokens);
            }
            if (!transaction_plan.ok)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier transaction plan failed: ") +
                    transaction_plan.error);
            }
            MTPSpecStepPlanBatch &step_plans =
                transaction_plan.step_plans;
            if (step_plans.steps.size() != 1)
            {
                return fail_after_checkpoint(
                    std::string("All-position MTP verifier step-plan failed: ") +
                    "missing single-request step");
            }

            MTPSpecStepPlan &mutable_step = step_plans.steps.front();
            const int accepted_state_count =
                std::max(0, mutable_step.accepted_count);
            int shifted_publication_commit_count = 0;
            bool first_shifted_row_available_for_publication =
                first_shifted_row_available_from_sidecar;
            if (!first_shifted_row_available_for_publication &&
                !first_token_is_stop &&
                accepted_state_count > 0)
            {
                if (catchup.accepted_tokens.empty())
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier has no token for initial shifted-cache publication");
                }

                bool initial_shifted_commit_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_initial_shifted_commit",
                        "decode");
                    /*
                     * Non-reusable MoE/TP sidecars cannot keep the speculative
                     * row-zero shifted KV append, but the verifier has not yet
                     * published a hidden row for the accepted first token.  The
                     * only serial-decode source for that shifted row is the
                     * verifier-base terminal hidden captured before sidecar
                     * draft work.  Import just that checkpoint hidden row here;
                     * the main KV/GDN/position publication remains owned by the
                     * verifier-row publisher below.
                     */
                    initial_shifted_commit_ok =
                        runner_->commitMTPShiftedRowFromCheckpointTerminalHidden(
                            verifier_base_checkpoint,
                            catchup.accepted_tokens.front(),
                            /*already_appended_tokens=*/0,
                            /*allow_speculative_discard=*/true,
                            static_cast<int>(verifier_base_checkpoint.cached_tokens));
                }
                if (!initial_shifted_commit_ok)
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier initial shifted-cache commit failed");
                }
                first_shifted_row_available_for_publication = true;
                shifted_publication_commit_count += 1;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_initial_shifted_commits",
                    1.0,
                    "decode",
                    {},
                    {{"source", "verifier_base_checkpoint_terminal_hidden"}});
            }

            mutable_step.reuse_initial_mtp_shifted_kv_row =
                first_shifted_row_available_for_publication;
            const MTPSpecStepPlan &step = mutable_step;
            if (!first_token_is_stop &&
                accepted_state_count > 1)
            {
                if (accepted_state_count >
                    static_cast<int>(catchup.accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier accepted-state publication exceeds committed outputs");
                }
                bool shifted_catchup_ok = false;
                {
                    PerfStatsCollector::ScopedTimer timer(
                        "mtp",
                        "all_position_shifted_prefix_commit",
                        "decode");
                    /*
                     * `already_appended_tokens` is a verifier-row indexing
                     * count.  Row zero hidden is the source for accepted token
                     * one, even when the sidecar row is not a reusable shifted
                     * KV boundary.  Shifted-KV residency is a separate count:
                     * reusable sidecars own one skipped shifted row; restored
                     * MoE/TP verifier-base paths own zero skipped verifier
                     * rows and must be anchored to the verifier-base cached
                     * token count. The DGO projects that absolute condition
                     * position using its cache's immutable model origin:
                     * ordinary pre-commit is max(0,verifier_base-1), while
                     * QFN retains its visible position-zero seed and expects
                     * verifier_base before any additional resident rows.
                     */
                    const int shifted_commit_position_offset =
                        first_shifted_row_available_from_sidecar
                            ? base_sidecar_position
                            : static_cast<int>(verifier_base_checkpoint.cached_tokens);
                    const int already_appended_shifted_kv_tokens =
                        first_shifted_row_available_for_publication ? 1 : 0;
                    shifted_catchup_ok =
                        runner_->commitMTPShiftedRowsFromPartialForward(
                            catchup.accepted_tokens.data(),
                            accepted_state_count,
                            /*already_appended_tokens=*/1,
                            catchup.main_forward_token_count,
                            /*allow_speculative_discard=*/true,
                            shifted_commit_position_offset,
                            already_appended_shifted_kv_tokens);
                }
                if (!shifted_catchup_ok)
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier shifted-cache accepted-prefix commit failed");
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_shifted_prefix_commits",
                    static_cast<double>(accepted_state_count - 1),
                    "decode");
                shifted_publication_commit_count += accepted_state_count - 1;
            }

            std::string publication_error;
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "grouped_outcome_publish_accepted_state_host",
                    "decode");
                if (!runner_->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                        step_plans,
                        &publication_error))
                {
                    return fail_after_checkpoint(
                        std::string("MTP grouped verifier state publication failed: ") +
                        publication_error);
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_host_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"request_count", std::to_string(step_plans.request_count)},
                     {"accepted_state_count", std::to_string(accepted_state_count)},
                     {"stopped_on_output", catchup.stopped_on_output ? "true" : "false"}});
            }

            int correction_forward_count = 0;
            int deferred_correction_condition_count = 0;
            int deferred_correction_start_index = kMTPSpecDecodeInvalidToken;
            if (step.requiresCorrectionReplay())
            {
                const int replay_start = step.correction_replay_start_index;
                const int replay_count = step.correction_replay_count;
                if (replay_start < 0 ||
                    replay_start + replay_count >
                        static_cast<int>(catchup.accepted_tokens.size()))
                {
                    return fail_after_checkpoint(
                        "All-position MTP verifier deferred correction plan is outside committed outputs");
                }
                correction_forward_count = 0;
                deferred_correction_condition_count = replay_count;
                deferred_correction_start_index = replay_start;
                /*
                 * A rejected correction token is host-visible output, but it is
                 * not live model state yet.  Do not append its shifted-MTP KV
                 * row at the rejecting step: doing so leaves the sidecar cache
                 * one row ahead of the accepted verifier prefix.  The next
                 * all-position verifier step consumes the pending condition row
                 * exactly once and lets the normal sidecar graph append the
                 * shifted row at the same boundary as a serial decode.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_deferred_correction_condition_tokens",
                    static_cast<double>(deferred_correction_condition_count),
                    "decode",
                    {},
                    {{"verifier_path",
                      "grouped_decode_equivalent_host_publication"},
                     {"start_index", std::to_string(replay_start)}});
            }

            std::vector<int32_t> accepted_tokens =
                std::move(catchup.accepted_tokens);
            std::vector<int32_t> verifier_tokens =
                std::move(catchup.verifier_tokens);
            const bool all_speculative_accepted =
                catchup.all_speculative_accepted;
            constexpr bool commit_boundary_clipped = false;
            const int accepted_speculative_prefix =
                catchup.accepted_speculative_prefix;
            const int32_t rejected_verified_token =
                catchup.rejected_verified_token;
            const bool stopped_on_output = catchup.stopped_on_output;
            if (!step.requiresCorrectionReplay() &&
                !stopped_on_output)
            {
                const int derived_correction_count =
                    std::max(
                        0,
                        static_cast<int>(accepted_tokens.size()) -
                            accepted_state_count);
                if (derived_correction_count > 0)
                {
                    deferred_correction_start_index = accepted_state_count;
                    deferred_correction_condition_count =
                        derived_correction_count;
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "all_position_derived_deferred_correction_condition_tokens",
                        static_cast<double>(
                            deferred_correction_condition_count),
                        "decode",
                        {},
                        {{"verifier_path",
                          "grouped_decode_equivalent_host_publication"},
                         {"accepted_state_count",
                          std::to_string(accepted_state_count)},
                         {"committed_output_count",
                          std::to_string(accepted_tokens.size())}});
                }
            }
            const int32_t raw_ready_token = catchup.ready_token;
            int32_t ready_token = raw_ready_token;
            const bool has_deferred_correction_condition =
                !stopped_on_output &&
                deferred_correction_condition_count > 0;
            if (has_deferred_correction_condition && ready_token >= 0)
            {
                /*
                 * Some verifier paths can already have sampled a token after
                 * the rejected correction row.  That token is useful evidence
                 * for diagnostics, but it is not a serial decode boundary: the
                 * correction token itself has only been emitted, not consumed
                 * as the next condition row.  Suppress terminal ready state and
                 * let the following decode step consume the pending correction
                 * exactly once.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_deferred_correction_ready_tokens_suppressed",
                    1.0,
                    "decode",
                    {},
                    {{"verifier_path",
                      "grouped_decode_equivalent_host_publication"},
                     {"raw_ready_token", std::to_string(raw_ready_token)},
                     {"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
                ready_token = -1;
            }
            const int main_forward_token_count =
                catchup.main_forward_token_count + correction_forward_count;
            result.is_complete = result.is_complete || stopped_on_output;
            const int emitted_token_start_index =
                first_token_is_already_emitted_condition ? 1 : 0;
            if (emitted_token_start_index >
                static_cast<int>(accepted_tokens.size()))
            {
                return fail_after_checkpoint(
                    "All-position MTP pending-condition commit has no matching committed row");
            }
            const int newly_emitted_token_count =
                static_cast<int>(accepted_tokens.size()) -
                emitted_token_start_index;
            std::optional<int32_t> next_pending_condition_token;
            std::optional<DeviceResidentLogicalSequenceStateHandle>
                next_pending_condition_resident_state;
            std::optional<DeviceResidentLogicalSequenceStateHandle>
                ready_condition_resident_state;

            if (!stopped_on_output &&
                has_deferred_correction_condition)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "all_position_rejection_without_ready_token",
                    1.0,
                    "decode",
                    {},
                    {{"deferred_condition_tokens",
                      std::to_string(deferred_correction_condition_count)}});
                if (deferred_correction_condition_count == 1)
                {
                    const int replay_start = deferred_correction_start_index;
                    if (replay_start < 0 ||
                        replay_start >=
                            static_cast<int>(accepted_tokens.size()))
                    {
                        return fail_after_checkpoint(
                            "All-position MTP pending correction row is outside committed outputs");
                    }
                    next_pending_condition_token =
                        accepted_tokens[static_cast<size_t>(replay_start)];
                }
                else if (deferred_correction_condition_count > 1)
                {
                    return fail_after_checkpoint(
                        "All-position MTP pending-condition fast path supports one correction row");
                }
            }
            std::optional<std::string> tx_error =
                validate_spec_decode_transaction(
                    verifier_replay_check_path,
                    verifier_publication_path,
                    draft_tokens,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    all_speculative_accepted,
                    stopped_on_output,
                    accepted_speculative_prefix,
                    commit_boundary_clipped);
            if (tx_error)
            {
                return fail_after_checkpoint(*tx_error);
            }

            ++mtp_stats_.verifier_runs;
            mtp_stats_.verifier_token_count +=
                static_cast<uint64_t>(main_forward_token_count);
            mtp_stats_.last_transaction_draft_depth =
                speculative_draft_count;
            PerfStatsCollector::addCounter("mtp", "verifier_runs", 1.0, "decode");
            PerfStatsCollector::addCounter(
                "mtp",
                "verifier_tokens",
                static_cast<double>(main_forward_token_count),
                "decode");
            {
                /*
                 * CPU greedy and stochastic verification share the same
                 * decode-equivalent grouped forward and host publication
                 * machinery.  Keep their route telemetry distinct: the
                 * stochastic matrix treats this counter as proof that target
                 * distributions and speculative acceptance were evaluated,
                 * rather than merely observing that some grouped forward ran.
                 */
                const char *grouped_host_verifier_counter =
                    stochastic_verify
                        ? "grouped_decode_equivalent_stochastic_verifier_runs"
                        : "grouped_decode_equivalent_greedy_verifier_runs";
                PerfStatsCollector::addCounter(
                    "mtp",
                    grouped_host_verifier_counter,
                    1.0,
                    "decode",
                    {},
                    {{"verifier_forward_tokens", std::to_string(main_forward_token_count)},
                     {"verifier_rows", std::to_string(sampled_verifier_rows.size())},
                     {"replay_forward_tokens", std::to_string(correction_forward_count)},
                     {"shifted_commits", std::to_string(shifted_publication_commit_count)},
                     {"verify_mode", stochastic_verify ? "stochastic" : "greedy"},
                     {"state_publication", "grouped_host"}});
            }

            if (!all_speculative_accepted && !commit_boundary_clipped)
            {
                ++mtp_stats_.rejected_tokens;
                PerfStatsCollector::addCounter("mtp", "rejected_tokens", 1.0, "decode");
            }

            if (accepted_speculative_prefix > 0)
            {
                mtp_stats_.accepted_tokens +=
                    static_cast<uint64_t>(accepted_speculative_prefix);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_tokens",
                    static_cast<double>(accepted_speculative_prefix),
                    "decode");
                PerfStatsCollector::addCounter(
                    "mtp",
                    "accepted_second_draft_tokens",
                    accepted_speculative_prefix > 0 ? 1.0 : 0.0,
                    "decode");
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "acceptance_trace",
                1.0,
                "decode",
                {},
                {{"draft_step", std::to_string(mtp_stats_.draft_steps)},
                 {"condition_token", std::to_string(condition_token)},
                 {"first_token", std::to_string(first_token)},
                 {"draft_tokens", join_tokens(draft_tokens)},
                 {"verifier_tokens", join_tokens(verifier_tokens)},
                 {"all_position_rows", join_tokens(sampled_verifier_rows)},
                 {"rejected_verified_token", std::to_string(rejected_verified_token)},
                 {"accepted_speculative_prefix", std::to_string(accepted_speculative_prefix)},
                 {"all_speculative_accepted", all_speculative_accepted ? "true" : "false"},
                 {"commit_boundary_clipped",
                  commit_boundary_clipped ? "true" : "false"},
                 {"verifier_state_matches_output", "true"},
                 {"verifier_path", verifier_publication_path},
                 {"catchup_implementation", verifier_publication_path},
                 {"policy_path", "grouped_outcome_host_publication"},
                 {"decode_equivalent_replay_required", "false"},
                 {"correction_replay_tokens", std::to_string(correction_forward_count)},
                 {"deferred_correction_condition_tokens",
                  std::to_string(deferred_correction_condition_count)},
                 {"output_tokens", std::to_string(newly_emitted_token_count)},
                 {"ready_token", std::to_string(ready_token)},
                 {"raw_ready_token", std::to_string(raw_ready_token)},
                 {"pending_condition_input",
                  first_token_is_already_emitted_condition ? "true" : "false"},
                 {"next_pending_condition_token",
                  next_pending_condition_token.has_value()
                      ? std::to_string(*next_pending_condition_token)
                      : std::string("none")},
                 {"used_ready_logits", use_ready_logits ? "true" : "false"}});

            if (!stopped_on_output &&
                (ready_token >= 0 || verify_commit_replay_check))
            {
                std::ostringstream replay_context;
                replay_context
                    << "draft_tokens=" << join_tokens(draft_tokens)
                    << " verifier_tokens=" << join_tokens(verifier_tokens)
                    << " all_position_rows=" << join_tokens(sampled_verifier_rows)
                    << " accepted_state_count=" << step.accepted_count
                    << " target_cached_tokens=" << step.target_cached_tokens
                    << " main_forward_token_count=" << main_forward_token_count
                    << " all_speculative_accepted="
                    << (all_speculative_accepted ? "true" : "false")
                    << " accepted_speculative_prefix="
                    << accepted_speculative_prefix;
                if (auto mismatch = verify_committed_prefix_replay(
                        verifier_replay_check_path,
                        accepted_tokens,
                        ready_token,
                        accepted_state_count,
                        replay_context.str()))
                {
                    return fail_after_checkpoint(*mismatch);
                }
            }
            // Validate the actual publication counts before committing output.
            MTPPublicationWitness publication;
            if (!MTPPublicationWitness::fromCounts(static_cast<int>(accepted_tokens.size()),
                    emitted_token_start_index, accepted_state_count, stopped_on_output, &publication))
                return fail_after_checkpoint("Host MTP committed publication witness is outside its one-row debt contract");
            if (auto commit_error = commit_mtp_transaction_outputs(
                    "grouped_decode_equivalent_host_verifier",
                    verifier_base_checkpoint,
                    accepted_tokens,
                    stopped_on_output || ready_token < 0
                        ? std::optional<int32_t>{}
                        : std::optional<int32_t>{ready_token},
                    /*terminal_logits_ready=*/!stopped_on_output && ready_token >= 0,
                    /*is_complete=*/stopped_on_output,
                    PrefixStateProvenance::VerifierPrefillRowsDecodeEquivalent,
                    /*state_advanced=*/true,
                    /*state_advanced_token_count=*/accepted_state_count,
                    emitted_token_start_index,
                    next_pending_condition_token,
                    next_pending_condition_resident_state,
                    ready_condition_resident_state))
            {
                return fail_after_checkpoint(*commit_error);
            }

            // Main-state publication and response output commit both succeeded.
            // The same source counts author the learner and diagnostic; no
            // acceptance/rollback arithmetic invents either publication debt.
            recordMTPDepthObservation(
                requested_speculative_draft_count, speculative_draft_count, accepted_speculative_prefix,
                draft_count_budget_limited, /*rollback=*/false, publication);
            PerfStatsCollector::addCounter(
                "mtp",
                "output_tokens",
                static_cast<double>(newly_emitted_token_count),
                "decode",
                {},
                {{"source", "committed_host_mtp_transaction"}});

            return result;
        }

        if (use_grouped_decode_equivalent_outcome_verifier)
        {
            const bool grouped_outcome_device_resident_publication =
                use_grouped_outcome_device_resident_publication_verifier;
            if (grouped_outcome_device_resident_publication)
            {
                /*
                 * This is intentionally stricter than "policy selected the
                 * grouped-outcome lane". LocalTP can prove grouped verifier
                 * row math while still lacking a single compact reducer across
                 * all TP shards. Only runners that passed the earlier
                 * device-resident publication gate may enter this hot path.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_publication_uses",
                    1.0,
                    "decode",
                    {},
                    {{"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
                     {"probe_rows", std::to_string(verifier_policy_probe_rows)},
                     {"reason", verifier_policy.reason}});
            }
            if (use_grouped_outcome_host_publication_verifier)
            {
                /*
                 * LocalTP can reduce greedy verifier rows across sharded
                 * logits, but it still lacks one compact device-resident
                 * outcome reducer spanning every TP participant.  This middle
                 * lane keeps the grouped verifier forward and publishes the
                 * accepted rows from a host-visible MTPSpecStepPlanBatch
                 * through the narrow grouped publication API.
                 */
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_host_publication_uses",
                    1.0,
                    "decode",
                    {},
                    {{"model_class", mtpDepthPolicyModelClassName(verifier_model_class)},
                     {"probe_rows", std::to_string(verifier_policy_probe_rows)},
                     {"reason", verifier_policy.reason}});
            }
            const bool sidecar_preserves_main_state =
                runner_->supportsMTPSidecarPreservesMainState();
            const PrefixStateSnapshot *sidecar_checkpoint = nullptr;
            if (!sidecar_preserves_main_state)
            {
                if (sidecar_checkpoints.empty())
                {
                    return fail_after_checkpoint(
                        "Grouped decode-equivalent MTP verifier requires a post-sidecar checkpoint");
                }

                sidecar_checkpoint = &sidecar_checkpoints.front();
                if (!sidecar_checkpoint->valid)
                {
                    return fail_after_checkpoint(
                        "Grouped decode-equivalent MTP verifier received an invalid post-sidecar checkpoint");
                }
            }
            bool restored_verifier_base = sidecar_preserves_main_state;
            if (sidecar_preserves_main_state)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "mtp",
                    "grouped_decode_equivalent_verifier_base_restore_skipped_sidecar_preserved",
                    1.0,
                    {static_cast<uint64_t>(verifier_base_checkpoint.cached_tokens)},
                    "decode",
                    {},
                    {{"draft_tokens", std::to_string(draft_tokens.size())},
                     {"discarded_sidecar_checkpoint",
                      sidecar_checkpoint ? "true" : "false"}});
            }
            else
            {
                PerfStatsCollector::ScopedTimer timer(
                    "mtp",
                    "grouped_decode_equivalent_verifier_restore_base_checkpoint",
                    "decode");
                restored_verifier_base =
                    runner_->restoreLivePrefixState(verifier_base_checkpoint);
                if (restored_verifier_base)
                {
                    PerfStatsCollector::addCounterWithSequence(
                        "mtp",
                        "grouped_decode_equivalent_verifier_base_restores",
                        1.0,
                        {static_cast<uint64_t>(verifier_base_checkpoint.cached_tokens)},
                        "decode",
                        {},
                        {{"draft_tokens", std::to_string(draft_tokens.size())},
                         {"discarded_sidecar_checkpoint",
                          sidecar_checkpoint ? "true" : "false"}});
                }
            }
            if (!restored_verifier_base)
            {
                return fail_after_checkpoint(
                    "Grouped decode-equivalent MTP verifier could not restore verifier base checkpoint after sidecar draft");
            }

            if (!grouped_outcome_device_resident_publication)
            {
                if (stochastic_verify &&
                    runner_->primaryDeviceId().is_gpu() &&
                    runner_->supportsDeviceResidentMTPSpecStatePublication() &&
                    !stochastic_device_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped decode-equivalent stochastic MTP verifier requires device-resident distribution verification");
                }
                return fail_after_checkpoint(
                    "Grouped decode-equivalent MTP verifier has no grouped publication path; serial row replay is diagnostic-only and is not a production fallback");
            }

            if (stochastic_verify && grouped_outcome_device_resident_publication)
            {
                if (!runner_->primaryDeviceId().is_gpu() || !stochastic_device_verify)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires GPU device-resident verification");
                }
                if (draft_tokens.size() <= 1)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires at least one speculative draft row");
                }
                if (stop_tokens_.size() >
                    static_cast<size_t>(
                        sampling_math::kSpeculativeBatchMaxStopTokens))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP has too many stop tokens for the compact device summary");
                }

                const bool first_token_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                const bool first_token_is_stop =
                    !first_token_deferred &&
                    std::find(stop_tokens_.begin(),
                              stop_tokens_.end(),
                              first_token) != stop_tokens_.end();
                const bool use_serial_sample_equivalent_stochastic =
                    active_sampling_params_.seed != 0;
                if (first_token_is_stop)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP stop-on-first-token short-circuit is not implemented");
                }

                const int compare_rows =
                    static_cast<int>(draft_tokens.size()) - 1;
                bool has_deferred_draft_token = false;
                bool has_host_visible_draft_token = false;
                for (int row = 0; row < compare_rows; ++row)
                {
                    const int32_t draft_token =
                        draft_tokens[static_cast<size_t>(row + 1)];
                    if (draft_token == kDeferredMTPDraftTokenShadow)
                        has_deferred_draft_token = true;
                    else if (draft_token >= 0)
                        has_host_visible_draft_token = true;
                    else
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP found an invalid draft token");
                    }
                }
                if (has_deferred_draft_token && has_host_visible_draft_token)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP does not support mixed host/deferred draft-token ownership");
                }
                if (has_host_visible_draft_token)
                {
                    if (!runner_->stageStochasticDraftTokensForDeviceVerification(
                            draft_tokens.data() + 1,
                            compare_rows,
                            /*first_draft_slot=*/0))
                    {
                        return fail_after_checkpoint(
                            "Grouped-outcome stochastic MTP draft-token device staging failed");
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_stochastic_draft_token_stages",
                        static_cast<double>(compare_rows),
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"}});
                }

                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifier_input_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome MTP verifier input metadata failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome MTP verifier row metadata is malformed");
                }

                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                const bool can_defer_grouped_verifier_sync =
                    !first_token_is_stop;
                ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                    runner_.get(),
                    can_defer_grouped_verifier_sync);
                ScopedMTPAllPositionVerifierTransaction verifier_transaction(
                    runner_.get(),
                    verifier_input_plan);
                auto fail_active_verifier_transaction =
                    [&](std::string message) -> GenerationResult
                {
                    std::string cleanup_error;
                    if (!verifier_transaction.close(&cleanup_error))
                    {
                        if (!message.empty())
                            message += "; ";
                        message += cleanup_error;
                    }
                    /*
                     * Prefix restoration is an exclusive logical-state writer.
                     * Retire the deferred verifier reader before entering the
                     * shared rollback helper or the writer would wait on this
                     * same stack frame's reader admission.
                     */
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(message);
                };
                if (!verifier_transaction.ready())
                {
                    return fail_active_verifier_transaction(
                        verifier_transaction.error());
                }
                const void *verifier_input_tokens_device = nullptr;
                {
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "grouped_outcome_stochastic_verifier_forward",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(verifier_row_count)}});
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "grouped_stochastic_verifier",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        return fail_active_verifier_transaction(
                            preparation_error);
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_verifier_device_token_inputs",
                        1.0,
                        "decode",
                        {},
                        {{"total_tokens",
                          std::to_string(
                              verifier_input_plan.total_verifier_input_tokens)}});

                    MTPVerifierForwardExecutionOptions verifier_forward_options;
                    verifier_forward_options.device_token_ids =
                        verifier_input_tokens_device;
                    const MTPVerifierForwardExecutionResult verifier_forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan,
                            verifier_forward_options);
                    if (!verifier_forward.ok)
                    {
                        return fail_active_verifier_transaction(
                            std::string("Grouped-outcome MTP verifier forward failed: ") +
                            verifier_forward.error);
                    }
                }

                const int bonus_row = compare_rows;
                const MTPRequestPenaltyPolicy verifier_penalty_policy{
                    .presence_penalty =
                        active_sampling_params_.presence_penalty,
                    .frequency_penalty =
                        active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                };
                if (!runner_
                         ->buildCapturedStochasticVerifierTargetDistributions(
                             compare_rows + 1,
                             active_sampling_params_,
                             verifier_penalty_policy,
                             vocab))
                {
                    return fail_active_verifier_transaction(
                        "Grouped-outcome stochastic MTP captured penalty/distribution transaction failed");
                }

                Sampler bonus_sampler = sampler_;
                const uint64_t inverse_sample_seed =
                    generation_request_seeds_->values().front();

                DeviceSpeculativeOutcomeHandle outcome_handle;
                bool resident_outcome_ok = false;
                {
                    PerfStatsCollector::ScopedTimer outcome_timer(
                        "mtp",
                        "grouped_outcome_stochastic_device_resident_outcome_enqueue",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(compare_rows)}});
                    // Both verification laws construct the same complete
                    // captured lifecycle. The law is immutable graph identity;
                    // omitting a public seed must not leave the outcome child
                    // unbuilt or capture host RNG draws/positions as constants.
                    {
                        DeviceStochasticBatchOutcomeRequest request;
                        request.request_id = 0;
                        request.first_target_slot = 0;
                        request.first_draft_slot = 0;
                        request.row_count = compare_rows;
                        request.first_token = -1;
                        request.first_token_from_device = true;
                        /*
                         * Captured verifier preparation has already copied the
                         * selected target-slot or resident-mailbox token into
                         * entry zero of the verifier token row. From this point
                         * onward that materialized row is the sole owner of the
                         * condition token consumed by the serial-equivalent
                         * summary kernel. Keeping either original producer named
                         * here would advertise two resident sources for one
                         * logical token and make stream/lifetime ownership
                         * ambiguous.
                         */
                        request.first_target_sample_slot = -1;
                        request.token_row_offset = 0;
                        request.token_row_stride =
                            verifier_input_plan.total_verifier_input_tokens;
                        request.bonus_target_slot = bonus_row;
                        request.bonus_threshold = 0.0f;
                        request.inverse_sample_seed = inverse_sample_seed;
                        request.inverse_sample_first_logical_position = -1;
                        request.derive_thresholds_from_seed = true;
                        request.draw_position_source =
                            DeviceStochasticDrawPositionSource::VerifierBaseSnapshot;
                        request.serial_sample_equivalent =
                            use_serial_sample_equivalent_stochastic;
                        request.use_vllm_probability_rejection =
                            !use_serial_sample_equivalent_stochastic;
                        request.use_device_draft_tokens = true;
                        request.stop_token_count =
                            static_cast<int>(stop_tokens_.size());
                        for (size_t i = 0; i < stop_tokens_.size(); ++i)
                        {
                            request.stop_tokens[i] = stop_tokens_[i];
                        }
                        resident_outcome_ok =
                            runner_->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                                &request,
                                /*request_count=*/1,
                                &outcome_handle);
                    }
                }
                std::string verifier_cleanup_error;
                if (!verifier_transaction.close(&verifier_cleanup_error))
                {
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(verifier_cleanup_error);
                }
                verifier_sync_deferral.close();
                if (!resident_outcome_ok)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP resident device outcome verifier failed");
                }

                MTPDecodeCatchupGreedyRequest catchup_request;
                catchup_request.draft_tokens = draft_tokens;
                catchup_request.stop_tokens = stop_tokens_;
                catchup_request.base_sidecar_position = base_sidecar_position;
                catchup_request.allow_speculative_discard = true;
                catchup_request.verifier_path =
                    "grouped_decode_equivalent_stochastic";
                catchup_request.implementation_name =
                    "device_batch_outcome_device_resident_publication";
                catchup_request.verifier_base_checkpoint =
                    &verifier_base_checkpoint;

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                MTPDecodeCatchupGreedyResult catchup;
                int shifted_publication_commit_count = 0;
                if (plan_.usesLocalTP())
                {
                    const bool first_shifted_row_available_from_sidecar =
                        sidecar_preserves_main_state &&
                        runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                        !first_token_is_stop;
                    const int max_state_commit_rows =
                        static_cast<int>(draft_tokens.size());
                    bool first_shifted_row_available_for_publication =
                        first_shifted_row_available_from_sidecar;
                    if (!first_shifted_row_available_from_sidecar &&
                        !first_token_is_stop &&
                        max_state_commit_rows > 0)
                    {
                        bool initial_shifted_commit_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_stochastic_initial_shifted_device_commit",
                                "decode");
                            initial_shifted_commit_ok =
                                runner_->commitMTPInitialShiftedRowFromDeviceOutcome(
                                    verifier_base_checkpoint,
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!initial_shifted_commit_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP device-resident initial shifted-cache commit failed");
                        }
                        first_shifted_row_available_for_publication = true;
                        shifted_publication_commit_count += 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_stochastic_initial_shifted_device_commits",
                            1.0,
                            "decode",
                            {},
                            {{"source", "device_outcome_checkpoint_terminal_hidden"}});
                    }
                    if (first_shifted_row_available_for_publication &&
                        max_state_commit_rows > 1)
                    {
                        bool shifted_catchup_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_stochastic_shifted_prefix_device_commit",
                                "decode");
                            /*
                             * The sidecar owns row zero.  Rows after that boundary
                             * are prepared from compact device outcome metadata on
                             * the sidecar stream; non-accepted suffix rows are valid
                             * speculative writes and are discarded by shifted-KV
                             * publication's device-derived target count.
                             */
                            shifted_catchup_ok =
                                runner_->commitMTPShiftedRowsFromDeviceOutcome(
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*already_appended_tokens=*/1,
                                    max_state_commit_rows,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!shifted_catchup_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome stochastic MTP device-resident shifted-cache suffix commit failed");
                        }
                        shifted_publication_commit_count +=
                            max_state_commit_rows - 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_stochastic_shifted_prefix_device_commits",
                            static_cast<double>(max_state_commit_rows - 1),
                            "decode",
                            {},
                            {{"rows", std::to_string(max_state_commit_rows - 1)}});
                    }
                }

                if (!runner_->supportsDeviceResidentMTPSpecStatePublication())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP requires device-resident accepted-state publication; replay publication is not an accepted production path");
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.max_state_commit_rows =
                    visible_state_commit_plan.max_state_commit_rows;
                publication_request.publish_mtp_shifted_kv = true;
                publication_request.penalty_policy =
                    MTPRequestPenaltyPolicy{
                        .presence_penalty =
                            active_sampling_params_.presence_penalty,
                        .frequency_penalty =
                            active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                    };
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_publication_uses",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"sampling", "stochastic"},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"physical_verifier_rows",
                      std::to_string(
                          publication_request.physicalVerifierRowsPerRequest())}});

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())}});
                    if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error))
                    {
                        return fail_after_checkpoint(
                            std::string("Grouped-outcome stochastic MTP device-resident state publication failed: ") +
                            publication_error);
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"request_count", "1"},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"shifted_commits",
                      std::to_string(shifted_publication_commit_count)}});
                /*
                 * A GPU grouped transaction is admitted before its first draft
                 * launch. The compact reducer must therefore publish a
                 * controller-owned outcome, and dynamic execution must consume
                 * the same first-transaction admission edge used to size the
                 * complete captured graph family. Any other state is a broken
                 * lifecycle, never permission to reconstruct the transaction on
                 * the host.
                 */
                if (!publication_request.outcome
                         .device_generation_controller_owned)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome stochastic MTP lost device-generation controller ownership after admission");
                }
                if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
                    !materialize_dynamic_generation_loop_this_step)
                {
                    return fail_after_checkpoint(
                        "Dynamic grouped-outcome stochastic MTP did not begin at its admitted prefill boundary");
                }

                GenerationResult resident_result =
                    completeDeviceResidentGeneration(
                        publication_request,
                        DeviceGenerationSamplingMode::Stochastic,
                        transaction_base_cached_tokens,
                        requested_speculative_draft_count,
                        speculative_draft_count,
                        std::move(result));
                if (!resident_result.success())
                    return fail_after_checkpoint(resident_result.error);
                return resident_result;
            }

            if (!stochastic_verify && grouped_outcome_device_resident_publication)
            {
                if (!runner_->primaryDeviceId().is_gpu())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires a GPU device-resident verifier");
                }
                if (!runner_->supportsGreedyAllPositionBatchOutcomeOnDevice())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires compact greedy device outcome support");
                }
                if (!runner_->supportsDeviceResidentMTPSpecStatePublication())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP requires device-resident accepted-state publication");
                }
                if (stop_tokens_.size() >
                    static_cast<size_t>(
                        sampling_math::kSpeculativeBatchMaxStopTokens))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP has too many stop tokens for the compact device summary");
                }
                if (draft_tokens.empty() ||
                    draft_tokens.size() >
                        static_cast<size_t>(
                            effectiveMTPMaxDraftDepth(mtp) + 1))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP draft width exceeds the configured verifier graph capacity");
                }

                const bool first_token_deferred =
                    first_token == kDeferredMTPFirstTokenShadow;
                const bool has_deferred_draft_token =
                    std::find(
                        draft_tokens.begin() + 1,
                        draft_tokens.end(),
                        kDeferredMTPDraftTokenShadow) != draft_tokens.end();
                if (!runner_->supportsMTPDeviceDraftTokenInput())
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy GPU MTP requires device-resident verifier token input");
                }

                const MTPSpecDecodeVerifierInputPlan verifier_input_plan =
                    buildSingleRequestVerifierInputPlan(draft_tokens);
                if (!verifier_input_plan.ok)
                {
                    return fail_after_checkpoint(
                        std::string("Grouped-outcome greedy MTP verifier input metadata failed: ") +
                        verifier_input_plan.error);
                }
                if (!verifierInputPlanHasCompactRows(verifier_input_plan))
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier row metadata is malformed");
                }

                const int verifier_row_count =
                    verifier_input_plan.compact_logit_row_count;
                ScopedMTPAllPositionVerifierSyncDeferral verifier_sync_deferral(
                    runner_.get(),
                    true);
                const void *verifier_input_tokens_device = nullptr;
                {
                    PerfStatsCollector::ScopedTimer verifier_timer(
                        "mtp",
                        "grouped_outcome_greedy_verifier_forward",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(verifier_row_count)}});
                    ScopedMTPSpecVerifierInputPlan verifier_plan_scope(
                        runner_.get(),
                        verifier_input_plan);
                    auto fail_active_greedy_verifier =
                        [&](std::string message) -> GenerationResult
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        verifier_plan_scope.close();
                        /*
                         * Rollback is a logical-state writer. The verifier
                         * stream reader must publish and retire its edge before
                         * the shared failure helper requests writer admission.
                         */
                        verifier_sync_deferral.close();
                        return fail_after_checkpoint(message);
                    };
                    if (!verifier_plan_scope.installed())
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not install row metadata plan");
                    }
                    if (!runner_->setComputeRowIndexedAllPositionLogits(
                            true,
                            verifier_row_count))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not enable row-indexed logits");
                    }
                    if (!runner_->setComputeAllPositionLogits(true))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP verifier could not enable all-position logits");
                    }
                    std::string preparation_error;
                    verifier_input_tokens_device =
                        prepare_grouped_gpu_verifier_input_tokens(
                            verifier_input_plan,
                            "grouped_greedy_verifier",
                            &preparation_error);
                    if (!verifier_input_tokens_device)
                    {
                        return fail_active_greedy_verifier(
                            preparation_error);
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "grouped_outcome_verifier_device_token_inputs",
                        1.0,
                        "decode",
                        {},
                        {{"total_tokens",
                          std::to_string(
                              verifier_input_plan.total_verifier_input_tokens)},
                         {"sampling", "greedy"}});

                    MTPVerifierForwardExecutionOptions verifier_forward_options;
                    verifier_forward_options.device_token_ids =
                        verifier_input_tokens_device;
                    if (runner_->primaryDeviceId().is_gpu() &&
                        !runner_
                             ->prepareGreedyAllPositionBatchOutcomeGraph(
                                 verifier_row_count,
                                 stop_tokens_.data(),
                                 static_cast<int>(
                                     stop_tokens_.size()),
                                 MTPRequestPenaltyPolicy{
                                     .presence_penalty =
                                         active_sampling_params_
                                             .presence_penalty,
                                     .frequency_penalty =
                                         active_sampling_params_
                                             .frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                                 }))
                    {
                        return fail_active_greedy_verifier(
                            "Grouped-outcome greedy MTP could not arm the "
                            "graph-owned outcome transaction");
                    }
                    const MTPVerifierForwardExecutionResult verifier_forward =
                        executeMTPSpecVerifierForward(
                            *runner_,
                            verifier_input_plan,
                            verifier_forward_options);
                    if (!verifier_forward.ok)
                    {
                        return fail_active_greedy_verifier(
                            std::string("Grouped-outcome greedy MTP verifier forward failed: ") +
                            verifier_forward.error);
                    }
                    /*
                     * Keep row-indexed all-position logits alive until the
                     * compact outcome reducer has sampled every verifier row.
                     * Rank-level LocalTP consumes child LogitsLocalInfo after
                     * this forward, and tearing the views down here can make a
                     * rejected row sample from cleared or rebound buffers.
                     */
                }

                const int compare_rows =
                    static_cast<int>(draft_tokens.size()) - 1;

                DeviceSpeculativeOutcomeHandle outcome_handle;
                {
                    PerfStatsCollector::ScopedTimer outcome_timer(
                        "mtp",
                        "grouped_outcome_greedy_device_resident_outcome_enqueue",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"rows", std::to_string(compare_rows)}});
                    if (!runner_->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                            draft_tokens.data(),
                            static_cast<int>(draft_tokens.size()),
                            stop_tokens_.data(),
                            static_cast<int>(stop_tokens_.size()),
                            &outcome_handle))
                    {
                        runner_->setComputeAllPositionLogits(false);
                        runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                        verifier_sync_deferral.close();
                        return fail_after_checkpoint(
                            "Grouped-outcome greedy MTP resident device outcome verifier failed");
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_greedy_verifier_rows_sampled_before_all_position_cleanup",
                    static_cast<double>(compare_rows + 1),
                    "decode",
                    {},
                    {{"policy_path",
                      "grouped_outcome_device_resident_publication"},
                     {"sampling", "greedy"}});
                if (!runner_->setComputeAllPositionLogits(false))
                {
                    runner_->setComputeRowIndexedAllPositionLogits(false, 0);
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier could not disable all-position logits");
                }
                if (!runner_->setComputeRowIndexedAllPositionLogits(false, 0))
                {
                    verifier_sync_deferral.close();
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP verifier could not disable row-indexed logits");
                }
                verifier_sync_deferral.close();

                MTPDecodeCatchupGreedyRequest catchup_request;
                catchup_request.draft_tokens = draft_tokens;
                catchup_request.stop_tokens = stop_tokens_;
                catchup_request.base_sidecar_position = base_sidecar_position;
                catchup_request.allow_speculative_discard = true;
                catchup_request.verifier_path =
                    "grouped_decode_equivalent_greedy";
                catchup_request.implementation_name =
                    "device_batch_outcome_device_resident_publication";
                catchup_request.verifier_base_checkpoint =
                    &verifier_base_checkpoint;

                DeviceSpeculativeVerifyBatchOutcome device_outcome;
                MTPDecodeCatchupGreedyResult catchup;
                int shifted_publication_commit_count = 0;
                if (plan_.usesLocalTP())
                {
                    const bool first_shifted_row_available_from_sidecar =
                        sidecar_preserves_main_state &&
                        runner_->supportsMTPShiftedRowReuseFromSidecar() &&
                        !first_token_is_stop;
                    const int max_state_commit_rows =
                        static_cast<int>(draft_tokens.size());
                    bool first_shifted_row_available_for_publication =
                        first_shifted_row_available_from_sidecar;
                    if (!first_shifted_row_available_from_sidecar &&
                        !first_token_is_stop &&
                        max_state_commit_rows > 0)
                    {
                        bool initial_shifted_commit_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_greedy_initial_shifted_device_commit",
                                "decode");
                            initial_shifted_commit_ok =
                                runner_->commitMTPInitialShiftedRowFromDeviceOutcome(
                                    verifier_base_checkpoint,
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!initial_shifted_commit_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP device-resident initial shifted-cache commit failed");
                        }
                        first_shifted_row_available_for_publication = true;
                        shifted_publication_commit_count += 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_greedy_initial_shifted_device_commits",
                            1.0,
                            "decode",
                            {},
                            {{"source", "device_outcome_checkpoint_terminal_hidden"}});
                    }
                    if (first_shifted_row_available_for_publication &&
                        max_state_commit_rows > 1)
                    {
                        bool shifted_catchup_ok = false;
                        {
                            PerfStatsCollector::ScopedTimer timer(
                                "mtp",
                                "grouped_outcome_greedy_shifted_prefix_device_commit",
                                "decode");
                            shifted_catchup_ok =
                                runner_->commitMTPShiftedRowsFromDeviceOutcome(
                                    outcome_handle,
                                    /*request_index=*/0,
                                    /*already_appended_tokens=*/1,
                                    max_state_commit_rows,
                                    /*main_forward_token_count=*/max_state_commit_rows,
                                    /*allow_speculative_discard=*/true);
                        }
                        if (!shifted_catchup_ok)
                        {
                            return fail_after_checkpoint(
                                "Grouped-outcome greedy MTP device-resident shifted-cache suffix commit failed");
                        }
                        shifted_publication_commit_count +=
                            max_state_commit_rows - 1;
                        PerfStatsCollector::addCounter(
                            "mtp",
                            "grouped_outcome_greedy_shifted_prefix_device_commits",
                            static_cast<double>(max_state_commit_rows - 1),
                            "decode",
                            {},
                            {{"rows", std::to_string(max_state_commit_rows - 1)}});
                    }
                }

                DeviceSpeculativePublicationRequest publication_request;
                publication_request.outcome = outcome_handle;
                publication_request.max_state_commit_rows =
                    visible_state_commit_plan.max_state_commit_rows;
                publication_request.publish_mtp_shifted_kv = true;
                publication_request.penalty_policy =
                    MTPRequestPenaltyPolicy{
                        .presence_penalty =
                            active_sampling_params_.presence_penalty,
                        .frequency_penalty =
                            active_sampling_params_.frequency_penalty,
                    .repetition_penalty = active_sampling_params_.repetition_penalty,
                    };

                std::string publication_error;
                {
                    PerfStatsCollector::ScopedTimer direct_publish_timer(
                        "mtp",
                        "grouped_outcome_publish_accepted_state_device_resident",
                        "decode",
                        {},
                        {{"policy_path", "grouped_outcome_device_resident_publication"},
                         {"request_count", "1"},
                         {"logical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .logicalVerifierRowsPerRequest())},
                         {"physical_verifier_rows",
                          std::to_string(
                              publication_request
                                  .physicalVerifierRowsPerRequest())},
                         {"sampling", "greedy"}});
                    if (!runner_->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                            publication_request,
                            &publication_error))
                    {
                        return fail_after_checkpoint(
                            std::string("Grouped-outcome greedy MTP device-resident state publication failed: ") +
                            publication_error);
                    }
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "grouped_outcome_device_resident_state_publications",
                    1.0,
                    "decode",
                    {},
                    {{"policy_path", "grouped_outcome_device_resident_publication"},
                     {"request_count", "1"},
                     {"logical_verifier_rows",
                      std::to_string(
                          publication_request.logicalVerifierRowsPerRequest())},
                     {"shifted_commits",
                      std::to_string(shifted_publication_commit_count)},
                     {"sampling", "greedy"}});

                /*
                 * Greedy and stochastic grouped publication share the same
                 * controller-owned terminal contract. The reducer has already
                 * committed the first transaction; the selected CUDA/HIP
                 * execution policy owns every remaining graph transaction and
                 * surfaces only the terminal response ledger.
                 */
                if (!publication_request.outcome
                         .device_generation_controller_owned)
                {
                    return fail_after_checkpoint(
                        "Grouped-outcome greedy MTP lost device-generation controller ownership after admission");
                }
                if (mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic &&
                    !materialize_dynamic_generation_loop_this_step)
                {
                    return fail_after_checkpoint(
                        "Dynamic grouped-outcome greedy MTP did not begin at its admitted prefill boundary");
                }

                GenerationResult resident_result =
                    completeDeviceResidentGeneration(
                        publication_request,
                        DeviceGenerationSamplingMode::Greedy,
                        transaction_base_cached_tokens,
                        requested_speculative_draft_count,
                        speculative_draft_count,
                        std::move(result));
                if (!resident_result.success())
                    return fail_after_checkpoint(resident_result.error);
                return resident_result;
            }

        }

        return fail_after_checkpoint(
            std::string("MTP verifier policy selected unsupported path: ") +
            verifier_policy.reason);
    }

} // namespace llaminar2
