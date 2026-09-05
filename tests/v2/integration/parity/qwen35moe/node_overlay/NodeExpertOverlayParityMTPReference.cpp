/**
 * @file NodeExpertOverlayParityMTPReference.cpp
 * @brief MTPReference implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Compare the live grouped-MTP graph with recursive HF checkpoints.
     *
     * The classic decode parity loop is deliberately teacher forced so every
     * main-model row follows the exact Hugging Face trajectory. That loop does
     * not execute speculative sidecars. This check first records a serial
     * production trajectory by constraining the public decodeStep boundary to
     * one token. When both commands hold the same authenticated residency
     * epoch, grouped MTP
     * must reproduce that trajectory exactly. Dynamic and LLEP requests may
     * legitimately publish a new placement between those independent requests;
     * such rows are instead compared directly with their Hugging Face main-model
     * checkpoints and the epoch mismatch is retained in the diagnostic CSV.
     *
     * Sidecar tensors remain compared directly with Hugging Face whenever the
     * serial production prefix still names the same main-model row. Recursive
     * predictors select branch-qualified reference tensors using the proposal
     * tokens observed from the device authority. This keeps every checkpoint
     * mathematically comparable even when a narrow quantized-logit tie sends
     * production down a different draft branch from canonical HF argmax. The
     * two diagnostic CSVs complement—never replace—the six canonical
     * prefill/decode artifacts.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::runMTPHuggingFaceCheckpointParity() -> void
    {
        if (!isQwen122ProductionTest() || !activeMTPEnabled() ||
            !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const std::vector<int> expected_tokens =
            readDecodeTokensFromMetadata();
        ASSERT_GE(expected_tokens.size(), 2u)
            << "MTP parity needs a sampled prefill token and one sidecar transaction";

        const auto result_dir = ensureResultsDir();
        const auto token_csv_path =
            result_dir / "mtp_sidecar_token_trace.csv";
        const auto snapshot_csv_path =
            result_dir / "mtp_sidecar_snapshot_breakdown.csv";
        const auto failure_values_csv_path =
            result_dir / "mtp_sidecar_failure_values.csv";
        std::ofstream token_csv(token_csv_path, std::ios::trunc);
        std::ofstream snapshot_csv(snapshot_csv_path, std::ios::trunc);
        std::ofstream failure_values_csv(
            failure_values_csv_path, std::ios::trunc);
        ASSERT_TRUE(token_csv.is_open()) << token_csv_path;
        ASSERT_TRUE(snapshot_csv.is_open()) << snapshot_csv_path;
        ASSERT_TRUE(failure_values_csv.is_open()) << failure_values_csv_path;
        token_csv
            << "call,reference_step,selected_depth,emitted_tokens,"
               "serial_expected_tokens,hf_expected_tokens,hf_branch_compatible,"
               "serial_epoch_compatible,serial_movement_epoch,"
               "grouped_movement_epoch_begin,grouped_movement_epoch_end,"
               "serial_execution_epoch,grouped_execution_epoch,"
               "serial_trajectory_epoch,grouped_trajectory_epoch,"
               "production_mtp0_top1,hf_mtp0_top1,recursive_branch_compatible,"
               "verifier_identity_transaction_count,verifier_identity_depth,"
               "production_verifier_draft_tokens,"
               "draft_steps,verifier_runs,accepted,rejected,commits,"
               "rollbacks,validation_failures,current_position\n";
        snapshot_csv
            << "call,reference_step,reference_depth,production_key,reference_key,"
               "elements,cosine,max_abs_diff,kl,exact_indices,routing_overlap,"
               "routing_top1_match,finite,passed\n";
        failure_values_csv
            << "call,reference_step,reference_depth,stage,index,production,reference\n"
            << std::setprecision(std::numeric_limits<float>::max_digits10);

        const std::array<std::string_view, 21> required_stages = {
            "TERMINAL_HIDDEN_ROW_SELECT",
            "EMBEDDING",
            "NORM_HIDDEN",
            "CONCAT",
            "FC",
            "ATTENTION_NORM",
            "Q_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "FINAL_NORM",
            "LM_HEAD",
            "V_PROJECTION",
        };

        const auto join_tokens = [](const std::vector<int32_t> &tokens)
        {
            std::ostringstream out;
            for (size_t index = 0; index < tokens.size(); ++index)
            {
                if (index != 0)
                    out << ';';
                out << tokens[index];
            }
            return out.str();
        };
        const std::array<std::string_view, 21> main_verifier_stage_suffixes = {
            "ATTENTION_NORM",
            "QKV_PROJECTION",
            "Q_PROJECTION",
            "GDN_Z_PROJECTION",
            "ATTENTION_CONTEXT",
            "ATTENTION_OUTPUT",
            "ATTENTION_OUTPUT_ALLREDUCED",
            "FFN_NORM_RESIDUAL_OUT",
            "FFN_NORM",
            "MOE_ROUTER_OUTPUT",
            "MOE_ROUTING_INDICES",
            "MOE_ROUTING_WEIGHTS",
            "MOE_EXPERT_OUTPUT",
            "MOE_SHARED_EXPERT_OUTPUT",
            "MOE_SHARED_GATE_OUTPUT",
            "MOE_COMBINED_OUTPUT",
            "FFN_RESIDUAL",
            "GDN_CONV1D_OUTPUT",
            "GDN_DELTA_RULE_OUTPUT",
            "GDN_NORM_GATE_OUTPUT",
            "GDN_OUTPUT",
        };
        const auto is_main_verifier_diagnostic_key =
            [&](const std::string &key)
        {
            if (key == "EMBEDDING" || key == "FINAL_NORM" ||
                key == "LM_HEAD" || key == "LM_HEAD_ROWS_SELECT")
            {
                return true;
            }
            if (key.rfind("layer", 0) != 0)
                return false;
            return std::any_of(
                main_verifier_stage_suffixes.begin(),
                main_verifier_stage_suffixes.end(),
                [&](std::string_view suffix)
                {
                    return std::string_view(key).ends_with(suffix);
                });
        };
        const auto capture_main_verifier_diagnostics = [&]
        {
            std::map<std::string, std::vector<float>> snapshots;
            for (const auto &key : activeSnapshotKeys())
            {
                if (!is_main_verifier_diagnostic_key(key))
                    continue;
                size_t elements = 0;
                const float *const data = activeSnapshot(key, elements);
                if (data && elements > 0)
                    snapshots.emplace(key, std::vector<float>(data, data + elements));
            }
            return snapshots;
        };
        const auto placement_trajectory_after_prefill =
            [&](uint64_t movement_epoch_begin,
                uint64_t movement_epoch_end)
        {
            MTPParityPlacementEpochTrajectory trajectory{
                .epoch = movement_epoch_begin,
            };
            const PrefixRuntimeStateSnapshot state = activePrefixStateProbe();
            const bool restored_prefix =
                state.prefix_request.hit || state.prefix_request.partial_hit;
            if (restored_prefix && isDynamicResidencyProductionTest())
            {
                /*
                 * Placement-agnostic production prefix reuse is numerically
                 * valid, but its restored recurrent bytes may have been
                 * produced before a later expert migration. The cache does not
                 * claim byte-identical placement provenance, so a dynamic hit
                 * cannot seed the stricter grouped-vs-serial oracle.
                 */
                trajectory.invalidate();
                return trajectory;
            }

            uint64_t execution_epoch = movement_epoch_end;
            if (!restored_prefix)
            {
                const auto placement = pinnedDevicePlacementEpochEvidence();
                EXPECT_TRUE(placement.has_value())
                    << "Executed ExpertOverlay prefill omitted its "
                       "device-authenticated placement evidence";
                execution_epoch =
                    placement.has_value() ? placement->epoch : 0u;
            }
            trajectory.observe(
                movement_epoch_begin,
                movement_epoch_end,
                execution_epoch);
            return trajectory;
        };
        /*
         * Record the serial production oracle through the same public surface
         * used by the HTTP server. A one-token response budget cannot enter a
         * speculative transaction, but it still advances the captured main
         * graph and its shifted-MTP state exactly as a normal request does.
         *
         * Residency movement was already proved before numerical parity. Do
         * not submit an additional test-driven maintenance epoch here. Normal
         * Dynamic/LLEP requests can still publish histogram work between these
         * independent serving calls, so each captured row records its actual
         * movement epoch. Only requests whose entire inherited state trajectory
         * stayed in one matching epoch form a valid strict serial-equivalence
         * experiment; all grouped rows retain the direct HF oracle below.
         */
        activeClearSnapshots();
        activeClearCache();
        const uint64_t serial_prefill_movement_epoch_begin =
            orch_runner_->moeRuntimeMovementEpoch();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();
        const uint64_t serial_prefill_movement_epoch_end =
            orch_runner_->moeRuntimeMovementEpoch();
        MTPParityPlacementEpochTrajectory serial_trajectory =
            placement_trajectory_after_prefill(
                serial_prefill_movement_epoch_begin,
                serial_prefill_movement_epoch_end);
        const uint64_t mtp_certification_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        /** @brief One serial row plus the residency epoch that executed it. */
        struct SerialVerifierOracle
        {
            std::map<std::string, std::vector<float>> snapshots;
            uint64_t movement_epoch_begin = 0;
            uint64_t movement_epoch_end = 0;
            uint64_t execution_epoch = 0;
            std::optional<uint64_t> trajectory_epoch;

            /** @return Whether this row inherited a complete history in @p epoch. */
            bool trajectoryInEpoch(uint64_t epoch) const
            {
                return epoch != 0u && trajectory_epoch == epoch;
            }
        };
        std::vector<int32_t> serial_tokens;
        const size_t serial_oracle_token_count = std::max(
            expected_tokens.size(),
            usesDynamicMTPDepth()
                ? static_cast<size_t>(activeMTPDraftDepth() + 1)
                : expected_tokens.size());
        std::vector<SerialVerifierOracle>
            serial_oracles_by_output_count(serial_oracle_token_count + 1u);
        while (serial_tokens.size() < serial_oracle_token_count)
        {
            activeClearSnapshots();
            orch_runner_->setDecodeStepTokenBudget(1);
            const uint64_t movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult serial_step = orch_runner_->decodeStep();
            const uint64_t movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(serial_step.success()) << serial_step.error;
            ASSERT_EQ(serial_step.tokens.size(), 1u)
                << "A one-token serial oracle boundary returned a grouped response";
            const auto serial_placement =
                pinnedDevicePlacementEpochEvidence();
            ASSERT_TRUE(serial_placement.has_value())
                << "Serial ExpertOverlay command omitted its device-authenticated "
                   "execution epoch";
            serial_trajectory.observe(
                movement_epoch_begin,
                movement_epoch_end,
                serial_placement->epoch);
            serial_tokens.push_back(serial_step.tokens.front());
            serial_oracles_by_output_count[serial_tokens.size()] = {
                .snapshots = capture_main_verifier_diagnostics(),
                .movement_epoch_begin = movement_epoch_begin,
                .movement_epoch_end = movement_epoch_end,
                .execution_epoch = serial_placement->epoch,
                .trajectory_epoch = serial_trajectory.epoch,
            };
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        ASSERT_EQ(serial_tokens.size(), serial_oracle_token_count);
        ASSERT_GE(serial_tokens.size(), expected_tokens.size());
        ASSERT_EQ(serial_tokens.front(), expected_tokens.front())
            << "The prefill boundary must agree exactly before MTP branch comparison";

        /* Start a fresh request so the first grouped sidecar is decode_step0. */
        activeClearSnapshots();
        activeClearCache();
        const uint64_t grouped_prefill_movement_epoch_begin =
            orch_runner_->moeRuntimeMovementEpoch();
        ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
            << orch_runner_->lastError();
        const uint64_t grouped_prefill_movement_epoch_end =
            orch_runner_->moeRuntimeMovementEpoch();
        MTPParityPlacementEpochTrajectory grouped_trajectory =
            placement_trajectory_after_prefill(
                grouped_prefill_movement_epoch_begin,
                grouped_prefill_movement_epoch_end);

        std::vector<int32_t> emitted;
        const auto initial_state = activePrefixStateProbe();
        int speculative_calls = 0;
        int compared_stages = 0;
        int compared_recursive_stages = 0;
        int deferred_recursive_contexts = 0;
        int compared_main_stages = 0;
        int compared_main_lm_heads = 0;
        int failed_main_lm_heads = 0;
        double grouped_main_cosine_sum = 0.0;
        size_t grouped_main_cosine_count = 0u;
        int serial_epoch_compatible_calls = 0;
        int call = 0;
        while (emitted.size() < expected_tokens.size())
        {
            activeClearSnapshots();
            const auto before = activePrefixStateProbe();
            const int remaining = static_cast<int>(
                expected_tokens.size() - emitted.size());
            int selected_depth_before = usesDynamicMTPDepth()
                                            ? before.mtp_current_depth
                                            : activeMTPDraftDepth();
            if (selected_depth_before <= 0)
                selected_depth_before = activeMTPDraftDepth();
            ASSERT_GE(selected_depth_before, 1);
            ASSERT_LE(selected_depth_before, activeMTPDraftDepth());
            /*
             * A two-token response boundary is the smallest public serving
             * request that admits speculative execution. A rejection can let
             * one public call retire more than one device transaction, so the
             * counter fold below validates every retired transaction. The
             * row-zero snapshot comparison is enabled only when the counters
             * prove that this call has one unambiguous verifier identity.
             */
            orch_runner_->setDecodeStepTokenBudget(
                std::min(remaining, 2));
            const uint64_t grouped_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const GenerationResult step = orch_runner_->decodeStep();
            const uint64_t grouped_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(step.success()) << step.error;
            ASSERT_FALSE(step.tokens.empty());
            ASSERT_LE(step.tokens.size(), static_cast<size_t>(remaining));
            const auto after = activePrefixStateProbe();
            const auto grouped_placement =
                pinnedDevicePlacementEpochEvidence();
            ASSERT_TRUE(grouped_placement.has_value())
                << "Grouped ExpertOverlay command omitted its device-authenticated "
                   "execution epoch";
            const uint64_t grouped_execution_epoch =
                grouped_placement->epoch;
            grouped_trajectory.observe(
                grouped_movement_epoch_begin,
                grouped_movement_epoch_end,
                grouped_execution_epoch);
            const size_t output_begin = emitted.size();

            bool serial_epoch_compatible =
                grouped_trajectory.epoch.has_value();
            uint64_t serial_movement_epoch =
                grouped_movement_epoch_begin;
            uint64_t serial_execution_epoch = 0u;
            uint64_t serial_trajectory_epoch = 0u;
            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t oracle_index = output_begin + offset + 1u;
                ASSERT_LT(oracle_index, serial_oracles_by_output_count.size());
                const auto &oracle =
                    serial_oracles_by_output_count[oracle_index];
                serial_epoch_compatible =
                    serial_epoch_compatible &&
                    oracle.trajectoryInEpoch(grouped_execution_epoch) &&
                    grouped_trajectory.matches(grouped_execution_epoch);
                if (offset == 0)
                {
                    serial_movement_epoch = oracle.movement_epoch_begin;
                    serial_execution_epoch = oracle.execution_epoch;
                    serial_trajectory_epoch =
                        oracle.trajectory_epoch.value_or(0u);
                }
            }
            if (serial_epoch_compatible)
                ++serial_epoch_compatible_calls;

            for (size_t offset = 0; offset < step.tokens.size(); ++offset)
            {
                const size_t output_index = emitted.size() + offset;
                ASSERT_LT(output_index, expected_tokens.size());
                if (serial_epoch_compatible)
                {
                    EXPECT_EQ(step.tokens[offset], serial_tokens[output_index])
                        << "Grouped MTP diverged from the serial production token "
                           "trajectory at output "
                        << output_index;
                }
            }

            const MTPParityTransactionCounters before_counters{
                .draft_steps = before.mtp_draft_steps,
                .verifier_runs = before.mtp_verifier_runs,
            };
            const MTPParityTransactionCounters after_counters{
                .draft_steps = after.mtp_draft_steps,
                .verifier_runs = after.mtp_verifier_runs,
            };
            const auto transaction_activity =
                classifyMTPParityTransactionActivity(
                    before_counters,
                    after_counters);
            EXPECT_NE(
                transaction_activity,
                MTPParityTransactionActivity::Inconsistent)
                << "MTP draft/verifier counters advanced inconsistently: before "
                << "drafts=" << before.mtp_draft_steps
                << " verifiers=" << before.mtp_verifier_runs
                << " after drafts=" << after.mtp_draft_steps
                << " verifiers=" << after.mtp_verifier_runs;
            const bool speculative =
                transaction_activity ==
                MTPParityTransactionActivity::Speculative;
            const uint64_t executed_transaction_count =
                mtpParityExecutedTransactionCount(
                    before_counters,
                    after_counters);
            const uint64_t attempted_draft_tokens =
                mtpParityAttemptedDraftTokenCount(
                    before_counters,
                    after_counters);

            int reference_step = -1;
            int selected_depth = 0;
            int production_mtp0_top1 = -1;
            int hf_mtp0_top1 = -1;
            bool recursive_branch_compatible = false;
            bool hf_branch_compatible = false;
            if (speculative)
            {
                ++speculative_calls;
                reference_step =
                    mtpParityReferenceStepForConditionPosition(
                        before.mtp_next_condition_position,
                        static_cast<int>(config_.token_ids.size()));
                ASSERT_GE(reference_step, 0)
                    << "Device MTP controller omitted its live condition position";
                selected_depth = after.mtp_last_transaction_draft_depth;
                ASSERT_GE(selected_depth, 1);
                ASSERT_LE(selected_depth, activeMTPDraftDepth());
                EXPECT_EQ(selected_depth, selected_depth_before)
                    << "The device-owned transaction selected a different "
                       "depth than its pre-submission controller state";
                const size_t reference_prefix_length =
                    static_cast<size_t>(reference_step + 1);
                std::vector<int32_t> grouped_visible_prefix = emitted;
                grouped_visible_prefix.insert(
                    grouped_visible_prefix.end(),
                    step.tokens.begin(),
                    step.tokens.end());
                hf_branch_compatible =
                    reference_prefix_length <= expected_tokens.size() &&
                    reference_prefix_length <= grouped_visible_prefix.size() &&
                    std::equal(
                        grouped_visible_prefix.begin(),
                        grouped_visible_prefix.begin() + reference_prefix_length,
                        expected_tokens.begin());
                ASSERT_GE(executed_transaction_count, 1u);
                ASSERT_GE(attempted_draft_tokens, executed_transaction_count)
                    << "Every retired verifier transaction must attempt a draft";
                ASSERT_LE(
                    attempted_draft_tokens,
                    executed_transaction_count *
                        static_cast<uint64_t>(activeMTPDraftDepth()))
                    << "The device controller attempted more drafts than its "
                       "captured transaction capacity";
                ASSERT_EQ(
                    after.mtp_observed_verifier_transaction_count,
                    static_cast<int>(executed_transaction_count))
                    << "The durable verifier identity must be committed by "
                       "the same device transaction(s) reported by production";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_depth,
                    selected_depth)
                    << "The durable verifier identity names a different "
                       "dynamic-depth branch than the committed controller";
                ASSERT_EQ(
                    after.mtp_observed_verifier_draft_tokens.size(),
                    static_cast<size_t>(selected_depth))
                    << "Every committed draft token must have an exact, "
                       "response-visible verifier identity";

                const std::string first_sidecar_prefix = std::string(
                    mtpParityCheckpointContextPrefix(
                        before.mtp_verifier_runs == 0
                            ? MTPParityCheckpointContext::
                                  DeviceTargetTokenLivePosition
                            : MTPParityCheckpointContext::
                                  DeviceResidentLogicalState));
                size_t production_lm_head_size = 0;
                const float *production_lm_head = activeSnapshot(
                    first_sidecar_prefix + "MTP0_LM_HEAD",
                    production_lm_head_size);
                const std::vector<float> hf_mtp0_lm_head =
                    loadPyTorchSnapshot(
                        "decode_step" + std::to_string(reference_step) +
                        "_MTP0_LM_HEAD");
                ASSERT_NE(production_lm_head, nullptr);
                ASSERT_EQ(production_lm_head_size, hf_mtp0_lm_head.size());
                production_mtp0_top1 = static_cast<int>(std::distance(
                    production_lm_head,
                    std::max_element(
                        production_lm_head,
                        production_lm_head + production_lm_head_size)));
                hf_mtp0_top1 = static_cast<int>(std::distance(
                    hf_mtp0_lm_head.begin(),
                    std::max_element(
                        hf_mtp0_lm_head.begin(), hf_mtp0_lm_head.end())));
                recursive_branch_compatible =
                    production_mtp0_top1 == hf_mtp0_top1;

                struct ActiveContext
                {
                    std::string prefix;
                    int reference_depth = 0;
                };
                std::vector<ActiveContext> contexts;
                contexts.push_back({
                    .prefix = std::string(
                        mtpParityCheckpointContextPrefix(
                            before.mtp_verifier_runs == 0
                                ? MTPParityCheckpointContext::DeviceTargetTokenLivePosition
                                : MTPParityCheckpointContext::DeviceResidentLogicalState)),
                    .reference_depth = 0,
                });
                const size_t recursive_condition_tokens =
                    selected_depth > 1
                        ? static_cast<size_t>(selected_depth - 1)
                        : 0u;
                const bool has_recursive_proposal_identity =
                    recursive_condition_tokens > 0u &&
                    after.mtp_observed_verifier_draft_tokens.size() >=
                        recursive_condition_tokens &&
                    std::all_of(
                        after.mtp_observed_verifier_draft_tokens.begin(),
                        after.mtp_observed_verifier_draft_tokens.begin() +
                            static_cast<ptrdiff_t>(recursive_condition_tokens),
                        [](int32_t token) { return token >= 0; });
                bool recursive_reference_deferred = false;
                std::vector<int32_t> recursive_reference_condition_tokens;

                /**
                 * @brief Decide whether an observed proposal follows the
                 *        canonical Hugging Face recursive branch.
                 *
                 * Canonical recursive snapshots have no `_BRANCH_...`
                 * qualifier. Additive snapshots use that qualifier only after
                 * production selects a condition token different from the
                 * corresponding canonical HF predictor. Compare the complete
                 * preceding predictor chain so a depth-two checkpoint is
                 * canonical only when both condition tokens agree.
                 *
                 * @param reference_depth Number of recursive condition tokens
                 *        consumed by the requested checkpoint.
                 * @return True when the unqualified HF checkpoint is the exact
                 *         oracle for the observed production branch.
                 */
                const auto follows_canonical_hf_branch =
                    [&](int reference_depth) -> bool
                {
                    if (reference_depth < 0 ||
                        after.mtp_observed_verifier_draft_tokens.size() <
                            static_cast<size_t>(reference_depth))
                    {
                        ADD_FAILURE()
                            << "Cannot resolve recursive HF branch depth "
                            << reference_depth << " from "
                            << after.mtp_observed_verifier_draft_tokens.size()
                            << " observed proposal tokens";
                        return false;
                    }
                    for (int proposal_depth = 0;
                         proposal_depth < reference_depth;
                         ++proposal_depth)
                    {
                        const std::vector<float> canonical_logits =
                            loadPyTorchSnapshot(
                                "decode_step" +
                                std::to_string(reference_step) + "_MTP" +
                                std::to_string(proposal_depth) +
                                "_LM_HEAD");
                        if (canonical_logits.empty())
                        {
                            ADD_FAILURE()
                                << "Hugging Face pack omitted the canonical MTP"
                                << proposal_depth
                                << " LM-head oracle at decode step "
                                << reference_step;
                            return false;
                        }
                        const int32_t canonical_token = static_cast<int32_t>(
                            std::distance(
                                canonical_logits.begin(),
                                std::max_element(
                                    canonical_logits.begin(),
                                    canonical_logits.end())));
                        if (after.mtp_observed_verifier_draft_tokens[
                                static_cast<size_t>(proposal_depth)] !=
                            canonical_token)
                        {
                            return false;
                        }
                    }
                    return true;
                };
                if (has_recursive_proposal_identity)
                {
                    const int recursive_reference_depth = selected_depth - 1;
                    recursive_branch_compatible =
                        follows_canonical_hf_branch(
                            recursive_reference_depth);
                    if (!recursive_branch_compatible)
                    {
                        recursive_reference_condition_tokens.assign(
                            after.mtp_observed_verifier_draft_tokens.begin(),
                            after.mtp_observed_verifier_draft_tokens.begin() +
                                recursive_reference_depth);
                        recursive_reference_deferred =
                            hf_branch_compatible &&
                            !hasHuggingFaceMTPBranchReference(
                                reference_step,
                                recursive_reference_condition_tokens);
                    }
                    contexts.push_back({
                        .prefix = std::string(
                            mtpParityCheckpointContextPrefix(
                                MTPParityCheckpointContext::DeviceChainedTokenLivePosition)),
                        .reference_depth = recursive_reference_depth,
                    });
                }

                const auto sidecar_moe = getMoEConfig();
                for (const auto &context : contexts)
                {
                    ASSERT_TRUE(validatePinnedMTPSidecarRouteEvidence(
                        context.prefix + "MTP0_",
                        grouped_execution_epoch,
                        sidecar_moe.top_k,
                        sidecar_moe.num_experts))
                        << "Live speculative sidecar omitted its exact "
                           "ExpertOverlay route authority at reference depth "
                        << context.reference_depth;
                    if (context.reference_depth > 0 &&
                        recursive_reference_deferred)
                    {
                        ASSERT_TRUE(deferHuggingFaceMTPBranchReference(
                            call,
                            reference_step,
                            context.reference_depth,
                            recursive_reference_condition_tokens,
                            context.prefix,
                            required_stages,
                            snapshot_csv_path))
                            << "Could not preserve the live recursive MTP "
                               "checkpoint set before runner retirement";
                        ++deferred_recursive_contexts;
                        continue;
                    }
                    /*
                     * A quantized router may exchange only the lowest-weight
                     * top-k boundary expert while retaining the top-1 expert
                     * and k-1 set overlap. The routing checkpoint below proves
                     * that bounded discrete difference explicitly. Remember
                     * whether it occurred so the immediately dependent raw
                     * routed sum uses the established intermediate-tensor
                     * threshold; exact routes retain the stricter decode floor.
                     * Downstream combined output and LM-head checks are never
                     * relaxed by this state.
                     */
                    bool routing_indices_exact = true;
                    bool context_finite = true;
                    bool context_routing_top1_match = true;
                    float context_routing_overlap = 1.0f;
                    bool context_routing_weights_equivalent = false;
                    bool context_routed_expert_output_equivalent = false;
                    bool context_lm_head_passed = false;
                    ReferenceTopKContainmentResult context_lm_head_topk;
                    MoERoutingBoundaryResult context_routing_boundary;
                    double context_router_symmetric_kl =
                        std::numeric_limits<double>::infinity();
                    double context_numerical_cosine_sum = 0.0;
                    size_t context_numerical_stage_count = 0u;
                    for (const std::string_view stage : required_stages)
                    {
                        const std::string production_key =
                            context.prefix +
                            (stage == "TERMINAL_HIDDEN_ROW_SELECT"
                                 ? "MTP_TERMINAL_HIDDEN_ROW_SELECT"
                                 : "MTP0_" + std::string(stage));
                        std::string reference_prefix =
                            "decode_step" + std::to_string(reference_step);
                        if (context.reference_depth > 0)
                        {
                            ASSERT_GE(
                                after.mtp_observed_verifier_draft_tokens.size(),
                                static_cast<size_t>(context.reference_depth))
                                << "The proposal authority omitted tokens "
                                   "needed to identify recursive HF depth "
                                << context.reference_depth;
                            if (!follows_canonical_hf_branch(
                                    context.reference_depth))
                            {
                                reference_prefix += "_BRANCH";
                                for (int branch_depth = 0;
                                     branch_depth < context.reference_depth;
                                     ++branch_depth)
                                {
                                    reference_prefix += "_" + std::to_string(
                                        after.mtp_observed_verifier_draft_tokens[
                                            static_cast<size_t>(branch_depth)]);
                                }
                            }
                        }
                        const std::string reference_key =
                            reference_prefix + "_MTP" +
                            std::to_string(context.reference_depth) + "_" +
                            std::string(stage);
                        if (!hf_branch_compatible)
                            continue;

                        size_t actual_size = 0;
                        const float *actual =
                            activeSnapshot(production_key, actual_size);
                        const std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        ASSERT_NE(actual, nullptr)
                            << "Live sidecar omitted " << production_key;
                        ASSERT_GT(actual_size, 0u)
                            << "Live sidecar published an empty " << production_key;
                        ASSERT_FALSE(reference.empty())
                            << "Hugging Face pack omitted " << reference_key;

                        const float *expected = reference.data();
                        size_t expected_size = reference.size();
                        if (expected_size > actual_size && actual_size > 0 &&
                            expected_size % actual_size == 0)
                        {
                            expected += expected_size - actual_size;
                            expected_size = actual_size;
                        }
                        ASSERT_EQ(actual_size, expected_size)
                            << production_key << " versus " << reference_key;

                        bool finite = true;
                        bool exact_indices = true;
                        float routing_overlap = 1.0f;
                        bool routing_top1_match = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0; index < actual_size; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(expected[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(actual[index]) -
                                    static_cast<double>(expected[index])));
                            if (stage == "MOE_ROUTING_INDICES")
                            {
                                exact_indices = exact_indices &&
                                                actual[index] == expected[index];
                            }
                        }
                        float cosine = computeCosineSimilarity(
                            actual, expected, actual_size);
                        float kl = 0.0f;
                        bool passed = finite;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            /*
                             * Quantized router logits may swap one expert at the
                             * low-weight top-k boundary even when the routing
                             * distribution and routed value remain numerically
                             * equivalent. Preserve the highest-weight expert and
                             * require at least k-1 set overlap, matching the
                             * production parity suite's discrete-routing model.
                             */
                            std::set<int> actual_experts;
                            std::set<int> reference_experts;
                            for (size_t index = 0; index < actual_size; ++index)
                            {
                                actual_experts.insert(
                                    static_cast<int>(actual[index]));
                                reference_experts.insert(
                                    static_cast<int>(expected[index]));
                            }
                            size_t intersection = 0;
                            for (const int expert : actual_experts)
                            {
                                if (reference_experts.contains(expert))
                                    ++intersection;
                            }
                            routing_overlap = actual_size > 0
                                                  ? static_cast<float>(intersection) /
                                                        static_cast<float>(actual_size)
                                                  : 0.0f;
                            routing_top1_match =
                                actual_size > 0 && actual[0] == expected[0];
                            routing_indices_exact = exact_indices;
                            cosine = routing_overlap;
                            max_abs_diff = 1.0 - routing_overlap;
                            const float minimum_boundary_overlap =
                                sidecar_moe.top_k > 0
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(
                                                  sidecar_moe.top_k)
                                    : 1.0f;
                            size_t actual_router_size = 0u;
                            const float *const actual_router = activeSnapshot(
                                context.prefix +
                                    "MTP0_MOE_ROUTER_OUTPUT",
                                actual_router_size);
                            const std::vector<float> reference_router =
                                loadPyTorchSnapshot(
                                    reference_prefix + "_MTP" +
                                    std::to_string(context.reference_depth) +
                                    "_MOE_ROUTER_OUTPUT");
                            ASSERT_NE(actual_router, nullptr);
                            ASSERT_EQ(
                                actual_router_size,
                                reference_router.size());
                            context_routing_boundary =
                                compareMoERoutingBoundarySelections(
                                    expected,
                                    actual,
                                    actual_size,
                                    reference_router.data(),
                                    reference_router.size(),
                                    actual_router,
                                    actual_router_size,
                                    static_cast<size_t>(
                                        sidecar_moe.top_k));
                            passed = passed && routing_top1_match &&
                                     routing_overlap >=
                                         minimum_boundary_overlap &&
                                     context_routing_boundary.evaluated &&
                                     context_routing_boundary.equivalent;
                            context_routing_top1_match =
                                context_routing_top1_match &&
                                routing_top1_match;
                            context_routing_overlap = std::min(
                                context_routing_overlap,
                                routing_overlap);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            /*
                             * Ordered weight vectors can look identical even
                             * when their low-weight entries name different
                             * experts. Reconstruct sparse expert-ID vectors,
                             * exactly as the canonical decode campaign does,
                             * so the routing proof measures contribution mass
                             * rather than array position.
                             */
                            size_t actual_indices_size = 0u;
                            const float *const actual_indices = activeSnapshot(
                                context.prefix +
                                    "MTP0_MOE_ROUTING_INDICES",
                                actual_indices_size);
                            const std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    reference_prefix + "_MTP" +
                                    std::to_string(context.reference_depth) +
                                    "_MOE_ROUTING_INDICES");
                            ASSERT_NE(actual_indices, nullptr);
                            ASSERT_EQ(actual_indices_size, actual_size);
                            ASSERT_EQ(reference_indices.size(), actual_size);
                            const StageComparisonResult routing_result =
                                compareRoutingWeights(
                                    actual,
                                    reference,
                                    actual_indices,
                                    reference_indices,
                                    actual_size,
                                    sidecar_moe.top_k,
                                    sidecar_moe.num_experts,
                                    std::string(stage));
                            cosine = routing_result.cosine_similarity;
                            max_abs_diff = routing_result.max_abs_diff;
                            routing_overlap =
                                routing_result.routing_overlap;
                            passed = finite && routing_result.passed;
                            context_routing_weights_equivalent = passed;
                        }
                        else
                        {
                            const float numerical_threshold =
                                stage == "MOE_EXPERT_OUTPUT" &&
                                        !routing_indices_exact
                                    ? config_.cosine_threshold
                                    : config_.decode_cosine_threshold;
                            passed = passed &&
                                     cosine >= numerical_threshold;
                            if (stage == "MOE_ROUTER_OUTPUT")
                            {
                                context_router_symmetric_kl =
                                    symmetricProbabilityKLDivergence(
                                        expected,
                                        actual,
                                        actual_size,
                                        static_cast<size_t>(
                                            sidecar_moe.num_experts));
                                kl = static_cast<float>(
                                    context_router_symmetric_kl);
                                passed = passed &&
                                    context_router_symmetric_kl <=
                                        config_.mtp_kl_threshold.value_or(
                                            config_.kl_threshold);
                            }
                            if (stage == "MOE_EXPERT_OUTPUT")
                            {
                                context_routed_expert_output_equivalent =
                                    passed;
                            }
                        }
                        if (stage == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                actual,
                                expected,
                                actual_size,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed &&
                                     kl < config_.mtp_kl_threshold.value_or(
                                              config_.kl_threshold);
                            context_lm_head_topk =
                                evaluateReferenceTopKContainment(
                                    actual,
                                    expected,
                                    actual_size,
                                    actual_size,
                                    config_.pytorch_top1_in_topk);
                            passed = passed && context_lm_head_topk.passed;
                            context_lm_head_passed = passed;

                            /*
                             * A recurrent LM-head miss can originate either in
                             * the incoming hidden row or in the projection
                             * itself. Preserve the compact 3,072-value operand
                             * only on failure so an offline exact-codebook
                             * projection can attribute those two errors without
                             * making every successful matrix cell emit a full
                             * vocabulary vector.
                             */
                            if (!passed)
                            {
                                size_t hidden_size = 0u;
                                const float *const hidden = activeSnapshot(
                                    context.prefix + "MTP0_FINAL_NORM",
                                    hidden_size);
                                std::vector<float> reference_hidden =
                                    loadPyTorchSnapshot(
                                        reference_prefix + "_MTP" +
                                        std::to_string(
                                            context.reference_depth) +
                                        "_FINAL_NORM");
                                ASSERT_NE(hidden, nullptr);
                                ASSERT_EQ(hidden_size, reference_hidden.size());
                                for (size_t index = 0u;
                                     index < hidden_size;
                                     ++index)
                                {
                                    failure_values_csv
                                        << call << ',' << reference_step << ','
                                        << context.reference_depth
                                        << ",FINAL_NORM," << index << ','
                                        << hidden[index] << ','
                                        << reference_hidden[index] << '\n';
                                }
                            }
                        }

                        context_finite = context_finite && finite;
                        /*
                         * Terminal hidden is an input-provenance diagnostic,
                         * not another model-layer result. Gate it individually
                         * above, but do not double-count the previous sidecar's
                         * output in this sidecar's numerical aggregate.
                         */
                        if (stage != "TERMINAL_HIDDEN_ROW_SELECT" &&
                            parityStageContributesToLayerCosine(
                                stage,
                                routing_indices_exact))
                        {
                            context_numerical_cosine_sum += cosine;
                            ++context_numerical_stage_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ','
                            << context.reference_depth << ','
                            << production_key << ',' << reference_key << ','
                            << actual_size << ',' << cosine << ','
                            << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << routing_overlap << ','
                            << (routing_top1_match ? 1 : 0) << ','
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_stages;
                        if (context.reference_depth > 0)
                            ++compared_recursive_stages;
                    }

                    if (!hf_branch_compatible)
                    {
                        /*
                         * A residency epoch can move the independent Dynamic
                         * request onto a different quantized-logit branch. The
                         * token CSV retains that fact, but no HF tensor from
                         * the canonical branch is an oracle for this context.
                         */
                        continue;
                    }

                    /*
                     * Apply the same route-aware numerical contract as
                     * runDecodeParity(): every checkpoint remains in the CSV,
                     * routing uses dedicated discrete/sparse metrics, raw
                     * expert sums from different legal boundary routes do not
                     * enter an elementwise cosine, and the remaining semantic
                     * tensors form one sidecar-layer aggregate. The LM-head
                     * still independently proves cosine, KL, and the typed
                     * reference top-K containment, so aggregation cannot hide a wrong
                     * token distribution.
                     */
                    ASSERT_GT(context_numerical_stage_count, 0u);
                    const double context_numerical_cosine =
                        context_numerical_cosine_sum /
                        static_cast<double>(
                            context_numerical_stage_count);
                    EXPECT_TRUE(context_finite)
                        << "Recursive MTP context published a non-finite "
                           "checkpoint at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    EXPECT_TRUE(context_routing_top1_match)
                        << "Recursive MTP context changed its highest-weight "
                           "expert at reference depth "
                        << context.reference_depth << "\nCSV: "
                        << snapshot_csv_path;
                    const MoERoutedContributionResult routed_contribution =
                        adjudicateMoERoutedContribution({
                            .routing_boundary =
                                context_routing_boundary,
                            .sparse_routing_weights_equivalent =
                                context_routing_weights_equivalent,
                            .routed_expert_output_equivalent =
                                context_routed_expert_output_equivalent,
                            .router_symmetric_kl =
                                context_router_symmetric_kl,
                            .maximum_router_symmetric_kl =
                                config_.mtp_kl_threshold.value_or(
                                    config_.kl_threshold),
                        });
                    const bool routed_contribution_equivalent =
                        routed_contribution.evaluated &&
                        routed_contribution.equivalent;
                    EXPECT_TRUE(routed_contribution_equivalent)
                        << "Recursive MTP context changed both sparse routed "
                           "mass and the resulting expert value at reference depth "
                        << context.reference_depth
                        << " router_symmetric_kl="
                        << context_router_symmetric_kl
                        << " boundary_gap="
                        << context_routing_boundary.maximum_boundary_gap
                        << " boundary_error_limit="
                        << context_routing_boundary.maximum_error_limit
                        << " authority="
                        << static_cast<int>(routed_contribution.authority)
                        << "\nCSV: "
                        << snapshot_csv_path;
                    const bool numerical_aggregate_passed =
                        context.reference_depth > 0
                            ? productionRecursiveMTPAggregatePasses(
                                  context_numerical_cosine,
                                  config_.decode_cosine_threshold,
                                  config_
                                      .mtp_recursive_aggregate_cosine_floor
                                      .value_or(
                                          kMinimumProductionRecursiveMTPAggregateCosine))
                            : context_numerical_cosine >=
                                  static_cast<double>(
                                      config_.decode_cosine_threshold);
                    EXPECT_TRUE(numerical_aggregate_passed)
                        << "Recursive MTP context numerical aggregate failed "
                           "at reference depth "
                        << context.reference_depth
                        << " cosine=" << context_numerical_cosine
                        << " required="
                        << (context.reference_depth > 0
                                ? productionRecursiveMTPAggregateRequiredCosine(
                                      config_.decode_cosine_threshold,
                                      config_
                                          .mtp_recursive_aggregate_cosine_floor
                                          .value_or(
                                              kMinimumProductionRecursiveMTPAggregateCosine))
                                : static_cast<double>(
                                      config_.decode_cosine_threshold))
                        << "\nCSV: " << snapshot_csv_path;
                    EXPECT_TRUE(context_lm_head_passed)
                        << "Recursive MTP context LM-head/KL/top-k proof "
                           "failed at reference depth "
                        << context.reference_depth
                        << " configured_top_k="
                        << context_lm_head_topk.configured_top_k
                        << " reference_top1_in_production="
                        << context_lm_head_topk
                               .reference_top1_in_production
                        << " production_top1_in_reference="
                        << context_lm_head_topk
                               .production_top1_in_reference
                        << "\nCSV: "
                        << snapshot_csv_path;
                }

                /*
                 * Compare grouped verifier row zero directly with the canonical
                 * HF main-model row while the production-visible prefix names
                 * that reference branch. This is the authoritative numerical
                 * oracle when a Dynamic/LLEP publication makes the separately
                 * executed serial request a different residency experiment.
                 */
                if (hf_branch_compatible && executed_transaction_count == 1u)
                {
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    const auto gdn = getGDNHeadConfig();
                    const auto moe = getMoEConfig();
                    for (const auto &key : activeSnapshotKeys())
                    {
                        if (!is_main_verifier_diagnostic_key(key) ||
                            key == "LM_HEAD_ROWS_SELECT")
                        {
                            continue;
                        }

                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || grouped_elements == 0u ||
                            grouped_elements % verifier_rows != 0u)
                        {
                            continue;
                        }
                        const size_t row_elements =
                            grouped_elements / verifier_rows;
                        const std::string reference_key =
                            "decode_step" + std::to_string(reference_step) +
                            "_" + key;
                        std::vector<float> reference =
                            loadPyTorchSnapshot(reference_key);
                        if (reference.empty() ||
                            reference.size() < row_elements ||
                            reference.size() % row_elements != 0u)
                        {
                            continue;
                        }
                        if (reference.size() > row_elements)
                        {
                            reference.erase(
                                reference.begin(),
                                reference.end() -
                                    static_cast<ptrdiff_t>(row_elements));
                        }

                        std::string stage = key;
                        if (key.rfind("layer", 0) == 0)
                        {
                            const size_t delimiter = key.find('_');
                            if (delimiter != std::string::npos)
                                stage = key.substr(delimiter + 1u);
                        }
                        const std::vector<float> permuted =
                            applyGDNHeadPermutation(
                                grouped, row_elements, stage, gdn);
                        const float *const actual = permuted.empty()
                                                        ? grouped
                                                        : permuted.data();

                        StageComparisonResult result;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            result = compareRoutingIndices(
                                actual,
                                reference,
                                row_elements,
                                moe.top_k,
                                stage);
                        }
                        else if (stage == "MOE_ROUTING_WEIGHTS")
                        {
                            const std::string index_key =
                                key.substr(
                                    0,
                                    key.size() -
                                        std::string("MOE_ROUTING_WEIGHTS").size()) +
                                "MOE_ROUTING_INDICES";
                            size_t grouped_index_elements = 0;
                            const float *const grouped_indices =
                                activeSnapshot(
                                    index_key,
                                    grouped_index_elements);
                            std::vector<float> reference_indices =
                                loadPyTorchSnapshot(
                                    "decode_step" +
                                    std::to_string(reference_step) + "_" +
                                    index_key);
                            if (reference_indices.size() > row_elements &&
                                reference_indices.size() % row_elements == 0u)
                            {
                                reference_indices.erase(
                                    reference_indices.begin(),
                                    reference_indices.end() -
                                        static_cast<ptrdiff_t>(row_elements));
                            }
                            if (grouped_indices &&
                                grouped_index_elements ==
                                    verifier_rows * row_elements &&
                                reference_indices.size() == row_elements)
                            {
                                result = compareRoutingWeights(
                                    actual,
                                    reference,
                                    grouped_indices,
                                    reference_indices,
                                    row_elements,
                                    moe.top_k,
                                    moe.num_experts,
                                    stage);
                            }
                            else
                            {
                                result = compareTensors(
                                    actual, reference, row_elements, stage);
                            }
                        }
                        else
                        {
                            result = compareTensors(
                                actual, reference, row_elements, stage);
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        for (size_t index = 0; index < row_elements; ++index)
                        {
                            finite = finite && std::isfinite(actual[index]) &&
                                     std::isfinite(reference[index]);
                            exact_indices = exact_indices &&
                                            actual[index] == reference[index];
                        }
                        float kl = 0.0f;
                        bool passed = finite && result.passed;
                        if (stage == "MOE_ROUTING_INDICES")
                        {
                            const float minimum_overlap =
                                row_elements > 0u
                                    ? 1.0f -
                                          1.0f /
                                              static_cast<float>(row_elements)
                                    : 1.0f;
                            passed = finite &&
                                     result.routing_top1_match >= 1.0f &&
                                     result.routing_overlap >= minimum_overlap;
                        }
                        if (key == "LM_HEAD")
                        {
                            ++compared_main_lm_heads;
                            kl = computeKLDivergence(
                                actual,
                                reference.data(),
                                row_elements,
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            const ReferenceTopKContainmentResult topk =
                                evaluateReferenceTopKContainment(
                                    actual,
                                    reference.data(),
                                    row_elements,
                                    row_elements,
                                    config_.pytorch_top1_in_topk);
                            passed = finite &&
                                     result.cosine_similarity >=
                                         config_.decode_cosine_threshold &&
                                     kl < config_.kl_threshold &&
                                     topk.passed;
                            if (!passed)
                                ++failed_main_lm_heads;
                        }

                        /*
                         * Match runDecodeParity's established aggregation:
                         * routing has its own set/sparse-vector metrics, while
                         * numerical tensors contribute to the mean-cosine
                         * gate. Individual low-energy intermediates remain
                         * visible through their CSV `passed` field without
                         * letting an ill-conditioned cosine replace the
                         * end-to-end logit/KL/top-k proof.
                         */
                        if (!result.is_routing_stage)
                        {
                            grouped_main_cosine_sum +=
                                result.cosine_similarity;
                            ++grouped_main_cosine_count;
                        }

                        snapshot_csv
                            << call << ',' << reference_step << ",-2,"
                            << key << ',' << reference_key << ','
                            << row_elements << ','
                            << result.cosine_similarity << ','
                            << result.max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ','
                            << (result.is_routing_stage
                                    ? result.routing_overlap
                                    : 1.0f)
                            << ','
                            << (result.is_routing_stage
                                    ? result.routing_top1_match
                                    : 1.0f)
                            << ',' << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        ++compared_main_stages;
                    }
                }

                /*
                 * The first transaction of a two-token public boundary has an
                 * unambiguous grouped-verifier row-zero oracle: the serial
                 * request after consuming the same first visible token. Copy
                 * every compact M-row checkpoint against that serial M=1 row.
                 * This turns a wrong correction token into an earliest-stage
                 * CSV diagnosis instead of leaving only the final token ID.
                 */
                /*
                 * The verifier row is identified by the device-owned live
                 * condition position, not by the number of tokens returned by
                 * this public call. An accepted draft can make output_begin
                 * advance farther than the next verifier condition; mapping by
                 * response count then compares different autoregressive rows.
                 */
                const size_t serial_snapshot_index =
                    static_cast<size_t>(reference_step + 1);
                if (serial_epoch_compatible &&
                    executed_transaction_count == 1u &&
                    serial_snapshot_index <
                        serial_oracles_by_output_count.size())
                {
                    const auto &serial_snapshots =
                        serial_oracles_by_output_count[serial_snapshot_index]
                            .snapshots;
                    const size_t verifier_rows = static_cast<size_t>(
                        activeMTPPhysicalVerifierRows());
                    for (const auto &[key, serial_values] : serial_snapshots)
                    {
                        size_t grouped_elements = 0;
                        const float *const grouped =
                            activeSnapshot(key, grouped_elements);
                        if (!grouped || serial_values.empty() ||
                            grouped_elements !=
                                verifier_rows * serial_values.size())
                        {
                            continue;
                        }

                        bool finite = true;
                        bool exact_indices = true;
                        double max_abs_diff = 0.0;
                        for (size_t index = 0;
                             index < serial_values.size();
                             ++index)
                        {
                            finite = finite &&
                                     std::isfinite(grouped[index]) &&
                                     std::isfinite(serial_values[index]);
                            max_abs_diff = std::max(
                                max_abs_diff,
                                std::abs(
                                    static_cast<double>(grouped[index]) -
                                    static_cast<double>(serial_values[index])));
                            exact_indices = exact_indices &&
                                            grouped[index] ==
                                                serial_values[index];
                        }
                        const float cosine = computeCosineSimilarity(
                            grouped,
                            serial_values.data(),
                            serial_values.size());
                        const bool routing_indices =
                            std::string_view(key).ends_with(
                                "MOE_ROUTING_INDICES");
                        float kl = 0.0f;
                        bool passed = finite &&
                                      (routing_indices
                                           ? exact_indices
                                           : cosine >=
                                                 config_.decode_cosine_threshold);
                        if (key == "LM_HEAD")
                        {
                            kl = computeKLDivergence(
                                grouped,
                                serial_values.data(),
                                serial_values.size(),
                                static_cast<size_t>(orch_runner_->vocabSize()));
                            passed = passed && kl < config_.kl_threshold;
                        }
                        snapshot_csv
                            << call << ',' << reference_step << ",-1,"
                            << key << ",serial_output" << serial_snapshot_index
                            << '_' << key << ',' << serial_values.size() << ','
                            << cosine << ',' << max_abs_diff << ',' << kl << ','
                            << (exact_indices ? 1 : 0) << ",1,1,"
                            << (finite ? 1 : 0) << ','
                            << (passed ? 1 : 0) << '\n';
                        if (!passed)
                        {
                            for (size_t index = 0;
                                 index < serial_values.size();
                                 ++index)
                            {
                                failure_values_csv
                                    << call << ',' << reference_step
                                    << ",-1," << key << ',' << index << ','
                                    << grouped[index] << ','
                                    << serial_values[index] << '\n';
                            }
                        }
                        EXPECT_TRUE(passed)
                            << "Grouped verifier row zero diverged from serial "
                               "decode at "
                            << key << " cosine=" << cosine
                            << " max_abs_diff=" << max_abs_diff
                            << " kl=" << kl << "\nCSV: "
                            << snapshot_csv_path;
                    }
                }
            }

            const size_t output_end = output_begin + step.tokens.size();
            const std::vector<int32_t> serial_expected(
                serial_tokens.begin() + output_begin,
                serial_tokens.begin() + output_end);
            const std::vector<int32_t> hf_expected(
                expected_tokens.begin() + output_begin,
                expected_tokens.begin() + output_end);
            token_csv
                << call << ',' << reference_step << ',' << selected_depth
                << ',' << join_tokens(step.tokens)
                << ',' << join_tokens(serial_expected)
                << ',' << join_tokens(hf_expected)
                << ',' << (hf_branch_compatible ? 1 : 0) << ','
                << (serial_epoch_compatible ? 1 : 0) << ','
                << serial_movement_epoch << ','
                << grouped_movement_epoch_begin << ','
                << grouped_movement_epoch_end << ','
                << serial_execution_epoch << ','
                << grouped_execution_epoch << ','
                << serial_trajectory_epoch << ','
                << grouped_trajectory.epoch.value_or(0u) << ','
                << production_mtp0_top1 << ',' << hf_mtp0_top1 << ','
                << (recursive_branch_compatible ? 1 : 0) << ','
                << after.mtp_observed_verifier_transaction_count << ','
                << after.mtp_observed_verifier_draft_depth << ','
                << join_tokens(after.mtp_observed_verifier_draft_tokens)
                << ','
                << after.mtp_draft_steps << ',' << after.mtp_verifier_runs
                << ',' << after.mtp_accepted_tokens << ','
                << after.mtp_rejected_tokens << ','
                << after.mtp_transaction_commits << ','
                << after.mtp_transaction_rollbacks << ','
                << after.mtp_transaction_validation_failures << ','
                << after.current_position << '\n';
            emitted.insert(emitted.end(), step.tokens.begin(), step.tokens.end());
            ++call;
        }
        orch_runner_->setDecodeStepTokenBudget(0);

        if (usesDynamicMTPDepth())
        {
            /*
             * The two-token boundaries above deliberately isolate one
             * verifier identity for checkpoint diagnosis. At the configured
             * maximum depth of fifteen, those requests are budget-limited and
             * must not contaminate the adaptive controller's economy window.
             * Submit one ordinary full-width serving request so the
             * device-owned controller sees a complete real-weight
             * transaction and can make an evidence-backed depth decision.
             */
            activeClearSnapshots();
            activeClearCache();
            ASSERT_TRUE(orch_runner_->prefill(config_.token_ids))
                << orch_runner_->lastError();
            const auto policy_before = activePrefixStateProbe();
            const uint64_t policy_movement_epoch_begin =
                orch_runner_->moeRuntimeMovementEpoch();
            const int policy_response_budget = activeMTPDraftDepth() + 1;
            orch_runner_->setDecodeStepTokenBudget(policy_response_budget);
            const GenerationResult policy_step = orch_runner_->decodeStep();
            orch_runner_->setDecodeStepTokenBudget(0);
            const uint64_t policy_movement_epoch_end =
                orch_runner_->moeRuntimeMovementEpoch();
            ASSERT_TRUE(policy_step.success()) << policy_step.error;
            ASSERT_EQ(
                policy_step.tokens.size(),
                static_cast<size_t>(policy_response_budget))
                << "The dynamic-depth proof did not retire its admitted "
                   "full-width response budget";
            const auto policy_after = activePrefixStateProbe();
            EXPECT_GT(
                policy_after.mtp_depth_policy_windows,
                policy_before.mtp_depth_policy_windows)
                << "A non-budget-limited real-model verifier transaction did "
                   "not evaluate the device-owned depth policy";
            const auto promotions =
                policy_after.mtp_depth_policy_promotions -
                policy_before.mtp_depth_policy_promotions;
            const auto demotions =
                policy_after.mtp_depth_policy_demotions -
                policy_before.mtp_depth_policy_demotions;
            const auto updates =
                policy_after.mtp_depth_policy_updates -
                policy_before.mtp_depth_policy_updates;
            EXPECT_EQ(updates, promotions + demotions)
                << "The device-owned depth policy did not account for its "
                   "evaluated decision exactly once";
            if (demotions > 0u)
            {
                EXPECT_LT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth demotion did not reduce the selected width";
            }
            else if (promotions > 0u)
            {
                EXPECT_GT(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "A recorded depth promotion did not increase the selected width";
            }
            else
            {
                EXPECT_EQ(
                    policy_after.mtp_current_depth,
                    policy_before.mtp_current_depth)
                    << "An evidence-backed hold changed depth without recording an update";
            }

            if (!isDynamicResidencyProductionTest() &&
                policy_movement_epoch_begin == policy_movement_epoch_end)
            {
                ASSERT_GE(serial_tokens.size(), policy_step.tokens.size());
                for (size_t index = 0; index < policy_step.tokens.size(); ++index)
                {
                    EXPECT_EQ(policy_step.tokens[index], serial_tokens[index])
                        << "Dynamic-depth policy proof diverged from serial "
                           "decode at output "
                        << index;
                }
            }
        }

        const uint64_t final_movement_epoch =
            orch_runner_->moeRuntimeMovementEpoch();
        if (!isDynamicResidencyProductionTest())
        {
            EXPECT_EQ(final_movement_epoch, mtp_certification_movement_epoch)
                << "Static MTP certification crossed a residency epoch";
            EXPECT_GT(serial_epoch_compatible_calls, 0)
                << "Static MTP produced no same-epoch serial equivalence proof";
        }

        const auto final_state = activePrefixStateProbe();
        EXPECT_GT(speculative_calls, 0)
            << "No grouped MTP transaction executed";
        EXPECT_GT(compared_stages, 0)
            << "No live MTP checkpoint was compared";
        if (activeMTPDraftDepth() > 1)
        {
            EXPECT_GT(
                compared_recursive_stages + deferred_recursive_contexts,
                0)
                << "No recursive MTP checkpoint had an observed proposal "
                   "identity or a deferred exact HF branch proof";
        }
        EXPECT_GT(compared_main_stages, 0)
            << "No grouped main-model checkpoint was compared with Hugging Face";
        ASSERT_GT(grouped_main_cosine_count, 0u)
            << "Grouped main-model comparison produced no numerical rows";
        EXPECT_GE(
            grouped_main_cosine_sum /
                static_cast<double>(grouped_main_cosine_count),
            static_cast<double>(config_.decode_cosine_threshold))
            << "Grouped main-model aggregate cosine failed against Hugging Face; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(compared_main_lm_heads, 0)
            << "Grouped main-model comparison omitted LM_HEAD";
        EXPECT_EQ(failed_main_lm_heads, 0)
            << "Grouped main-model LM_HEAD failed cosine/KL/configured-mutual-top-K parity; CSV: "
            << snapshot_csv_path;
        EXPECT_GT(
            final_state.mtp_accepted_tokens,
            initial_state.mtp_accepted_tokens)
            << "The real 122B MTP request accepted no draft tokens";
        EXPECT_GT(
            final_state.mtp_transaction_commits,
            initial_state.mtp_transaction_commits);
        EXPECT_EQ(final_state.mtp_transaction_rollbacks, 0u);
        EXPECT_EQ(final_state.mtp_transaction_validation_failures, 0u);
        EXPECT_EQ(final_state.mtp_max_depth, activeMTPDraftDepth())
            << "MTP runtime capacity did not match the cell's admitted maximum";
        if (usesDynamicMTPDepth())
        {
            EXPECT_GT(final_state.mtp_depth_policy_windows, 0u)
                << "Dynamic-depth policy observed no completed verifier window";
            EXPECT_EQ(final_state.mtp_min_depth, 1);
        }
        token_csv.flush();
        snapshot_csv.flush();
        EXPECT_TRUE(token_csv.good());
        EXPECT_TRUE(snapshot_csv.good());
    }

}
