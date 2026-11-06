/**
 * @file DeviceGraphOrchestratorServingGraphFamily.cpp
 * @brief Participant-local preparation of the retained serving graph family.
 *
 * The existing orchestrator owns admission, buffers, workspace generations and
 * executable lifetimes. This translation unit records the exact prefill,
 * ordinary decode and MTP specializations without launching model arithmetic.
 * It preserves maximum-width preparation, sparse-query policy, snapshot
 * identity and the ordered reset before sealing the family for request reuse.
 */
#include "DeviceGraphOrchestrator.h"
#include "MTPGraphPolicyScopes.h"
#include "PipelineForwardGraphEdges.h"
#include "backends/BackendManager.h"
#include "execution/local_execution/engine/PrefillBucketUtils.h"
#include "execution/mtp/MTPCheckpointPolicy.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include "transfer/MappedTransferProgressEpoch.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"
#include <chrono>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    using dgo_detail::ScopedGroupedMTPVerifierGraphPolicy;
    using dgo_detail::ScopedLiveMTPRequestBatchConditionPolicy;

    namespace
    {
        /**
         * @brief Records one setup-only serving-graph materialization phase.
         *
         * Serving-family construction runs concurrently on every LocalTP
         * participant. A process-wide runtime or library bottleneck can leave
         * the rank coordinator waiting without revealing which device or graph
         * family member owns the long tail. This scope publishes the exact
         * phase boundary at INFO level and records its elapsed time in
         * PerfStats. It never observes device state, synchronizes a stream, or
         * participates in correctness decisions.
         */
        class ServingGraphMaterializationPhaseTrace final
        {
        public:
            /**
             * @brief Begin timing one named phase for one exact participant.
             * @param device Immutable participant identity.
             * @param phase Stable setup phase name used by logs and PerfStats.
             */
            ServingGraphMaterializationPhaseTrace(
                DeviceId device,
                std::string phase)
                : device_(device.toString()),
                  phase_(std::move(phase)),
                  started_(std::chrono::steady_clock::now())
            {
                LOG_INFO(
                    "[ServingGraphMaterialization] phase begin device="
                    << device_ << " phase=" << phase_);
            }

            /**
             * @brief Publish elapsed host setup time when the phase exits.
             *
             * "Exited" deliberately does not claim success: the owning method
             * emits the authoritative error when an operation fails. If a
             * process is externally terminated, the unmatched begin line names
             * the operation that was still active.
             */
            ~ServingGraphMaterializationPhaseTrace() noexcept
            {
                const auto elapsed = std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started_);
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "serving_graph_family_setup_phase",
                    static_cast<std::uint64_t>(elapsed.count()),
                    "setup",
                    device_,
                    {{"phase", phase_}});
                LOG_INFO(
                    "[ServingGraphMaterialization] phase exit device="
                    << device_ << " phase=" << phase_
                    << " elapsed_ms="
                    << (static_cast<double>(elapsed.count()) / 1.0e6));
            }

            ServingGraphMaterializationPhaseTrace(
                const ServingGraphMaterializationPhaseTrace &) = delete;
            ServingGraphMaterializationPhaseTrace &operator=(
                const ServingGraphMaterializationPhaseTrace &) = delete;

        private:
            std::string device_; ///< Stable participant name for diagnostics.
            std::string phase_;  ///< Stable setup phase name.
            std::chrono::steady_clock::time_point started_; ///< Host timer origin.
        };

    } // namespace

    bool DeviceGraphOrchestrator::
        materializeMTPServingForwardExecutablesWithoutLaunch(
            TensorBase *pipeline_hidden_input,
            void *publication_stream)
    {
        if (!graph_builder_ || !retainsMTPGraphCapacity(graph_builder_->config().mtp))
            return true;
        // Retain both condition purposes and both terminal outcome topologies
        // symmetrically: every participant must enter the same capture waves.
        for (const auto purpose : kMTPConditionForwardPurposes)
            if (!executeMTPMainForward(purpose, pipeline_hidden_input, publication_stream,
                    ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch))
                return false;
        const auto family = mtpVerifierGraphFamily();
        const auto retained = resolveMTPServingForwardCaptureGeometry(graph_builder_->config().mtp, mtp_max_verifier_rows_, family);
        for (const auto &geometry : resolveMTPServingForwardCaptureFamily(graph_builder_->config().mtp, mtp_max_verifier_rows_, family))
        for (const auto outcome : {MTPVerifierOutcomeGraphMode::Greedy, MTPVerifierOutcomeGraphMode::Disabled})
            if (!executeMTPMainForward(MTPVerifierForwardPolicy(outcome, geometry.verifier_rows, retained), pipeline_hidden_input, publication_stream,
                    ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch))
                return false;
        // These are the same two learned scalar owners used at runtime, not
        // depth clones. A main-model-only follower owns neither predictor nor
        // its inputs. Capacity, rather than request one's Off/fixed/dynamic
        // choice, declares the actual finite source-free draft family.
        if (mtp_graph_owner_plan_.ownsControllerExecutables())
        {
            const auto hidden=state_.extension_buffers.find(BufferId::MTP_HIDDEN);
            if (!state_.prefix_terminal_hidden || hidden==state_.extension_buffers.end() || !hidden->second ||
                !mtp_sidecar_condition_token_dev_ || !mtp_sidecar_position_ids_dev_)
                throw std::logic_error("Retained predictor setup lost its original full/chained input roots");
            for (const auto role : {MTPSidecarCaptureRole::Full,MTPSidecarCaptureRole::Chained})
            {
                if(role==MTPSidecarCaptureRole::Chained && resolveMTPRetainedDraftCapacity(graph_builder_->config().mtp)<=1)
                    continue;
                const bool chained=role==MTPSidecarCaptureRole::Chained;
                const int slot=mtp_sidecar_capture_layout_.conditionTokenSlot(role,1);
                const auto* condition=static_cast<const int32_t*>(mtp_sidecar_condition_token_dev_)+
                    slot*mtp_sidecar_condition_token_slot_width_;
                if(!executeMTPDepth0Batched(nullptr,1,chained?hidden->second.get():state_.prefix_terminal_hidden.get(),
                    0,chained?"mtp_decode_chained":"mtp_decode_sidecar",false,
                    chained?BufferId::MTP_HIDDEN:BufferId::PREFIX_TERMINAL_HIDDEN,true,
                    condition,-1,false,1,nullptr,mtp_sidecar_position_ids_dev_,nullptr,0,nullptr,0,-1,0,nullptr,
                    1,1,1,0,0,DeviceGraphExecutor::GraphInitialSubmissionPolicy::MaterializeWithoutLaunch))
                    return false;
            }
        }
        return true;
    }

    bool DeviceGraphOrchestrator::executeMTPMainForward(
        const MTPMainForwardPolicy &policy, TensorBase *pipeline_hidden_input,
        void *publication_stream, ForwardGraphSubmissionIntent submission, std::string *error)
    {
        const auto fail = [&](const char *reason) {
            if (error) *error = reason;
            LOG_ERROR("[DeviceGraphOrchestrator] Resident MTP main forward: " << reason);
            return false;
        };
        if (error) error->clear();
        const bool setup = submission == ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
        if (!state_.device_id.is_gpu() || !publication_stream || !graph_builder_ ||
            (!graph_builder_->config().mtp.enabled &&
             !(setup && retainsMTPGraphCapacity(graph_builder_->config().mtp))) ||
            (submission != ForwardGraphSubmissionIntent::Execute && !setup))
            return fail("requires one MTP GPU participant, exact stream and explicit submission policy");
        if (!setup && (!pp_stage_config_ || pp_stage_config_->has_lm_head || !pipeline_forward_edges_))
            return fail("live main-forward submission here belongs only to a frozen pipeline follower");
        const auto &logical = device_resident_logical_sequence_state_storage_;
        if (!logical.validFor(1) || !mtp_verifier_input_tokens_dev_ ||
            !mtp_verifier_position_ids_dev_ || !mtp_verifier_request_lengths_dev_)
            return fail("persistent condition or verifier rows are incomplete");

        ForwardInput input;
        input.batch_size = 1;
        input.execution_phase = ForwardExecutionPhase::Decode;
        input.graph_submission_intent = submission;
        input.position_policy = ForwardPositionPolicy::ExplicitRows;
        input.device_state_publication_stream = publication_stream;
        input.device = state_.device_id;
        input.kv_cache = state_.kv_cache.get();
        input.external_hidden_state = pipeline_hidden_input;
        input.moe_overlay_collective_generation_id = 1;

        // These scopes install immutable graph policy only. The collective
        // receive, not this host code, supplies each follower's live row data.
        std::optional<ScopedLiveMTPRequestBatchConditionPolicy> condition_scope;
        std::optional<ScopedGroupedMTPVerifierGraphPolicy> verifier_scope;
        const auto *purpose = std::get_if<MTPConditionForwardPurpose>(&policy);
        if (purpose)
        {
            if (*purpose != MTPConditionForwardPurpose::SpeculativeContinuation &&
                *purpose != MTPConditionForwardPurpose::CommittedSerialToken)
                return fail("unknown condition commit purpose");
            condition_scope.emplace(graph_builder_.get(), &live_mtp_request_batch_condition_, true);
            if (!condition_scope->ready()) return fail("cannot install condition topology");
            input.execution_role = ForwardExecutionRole::MTPCondition;
            input.state_transaction = *purpose == MTPConditionForwardPurpose::CommittedSerialToken
                ? ForwardStateTransaction::CommittedMTPCondition : ForwardStateTransaction::Ordinary;
            input.token_ids_device = logical.next_condition_tokens_device;
            input.position_ids_device = logical.target_cached_tokens_device;
            input.sequence_lengths_device = logical.target_cached_tokens_device;
            input.seq_len = input.real_seq_len = 1;
        }
        else
        {
            const auto &verifier = std::get<MTPVerifierForwardPolicy>(policy);
            const auto mode = verifier.outcome();
            if (mode != MTPVerifierOutcomeGraphMode::Greedy && mode != MTPVerifierOutcomeGraphMode::Disabled)
                return fail("unknown verifier outcome topology");
            auto geometry = resolveMTPServingForwardCaptureGeometry(graph_builder_->config().mtp,
                mtp_max_verifier_rows_, mtpVerifierGraphFamily());
            if (!geometry.enabled || !geometry.valid() || geometry.verifier_rows <= 0 ||
                geometry.verifier_rows > mtp_max_verifier_rows_ || verifier.physicalRows() > geometry.verifier_rows)
                return fail("configured verifier envelope is invalid");
            // Physical bucket identity is shared with the terminal. Active
            // token/position extents still arrive through device-owned metadata.
            geometry.verifier_rows = verifier.physicalRows();
            geometry.draft_depth = geometry.verifier_rows - 1;
            verifier_scope.emplace(graph_builder_.get(), &compute_all_position_logits_,
                &compute_row_indexed_all_position_logits_, &row_indexed_all_position_logits_row_count_,
                &mtp_verifier_outcome_graph_mode_, geometry.verifier_rows, mode);
            if (!verifier_scope->ready()) return fail("cannot install grouped verifier topology");
            input.execution_role = ForwardExecutionRole::GroupedMTPVerifier;
            input.state_transaction = ForwardStateTransaction::Ordinary;
            input.token_ids_device = mtp_verifier_input_tokens_dev_;
            input.position_ids_device = mtp_verifier_position_ids_dev_;
            input.sequence_lengths_device = static_cast<const int32_t *>(mtp_verifier_request_lengths_dev_);
            input.seq_len = geometry.verifier_rows;
            input.real_seq_len = geometry.draft_depth + 1;
            input.bucket_seq_len = input.real_seq_len < input.seq_len ? input.seq_len : 0;
            input.position_offset = input.token_offset = 1;
            input.moe_overlay_collective_step_id = 1;
            input.moe_overlay_mtp_depth = geometry.draft_depth;
        }

        TensorBase *logits = state_.logits.get();
        TensorBase *logits_local = state_.logits_local.get();
        if (!bindAllPositionLogitsOutputs(input.execution_role, input.seq_len, logits, logits_local))
            return fail("cannot bind main-model outputs for this participant");
        ModelBuffers buffers = state_.toModelBuffers();
        buffers.logits = logits;
        buffers.logits_local = logits_local;
        setBuffers(buffers);
        if (!bindShiftedMTPPrefillTransaction(input, 1) || !bindMTPMainTerminalHiddenTransaction(input, 1))
            return fail("cannot bind the local model-state transaction");
        ForwardOutput output;
        output.logits = logits;
        output.hidden = state_.hidden.get();
        std::optional<ForwardGraphSignature> materialized;
        if (!executeForward(input, output, setup ? &materialized : nullptr))
            return fail("canonical captured forward execution failed");
        if (setup)
        {
            const auto boundary=graph_builder_->retainedStorageBoundary(state_.device_id);
            if (boundary && boundary->retainedParticipantSource())
            {
                const auto retained=materialized
                    ? forward_engine_->retainedDecodeGraph(*materialized,error) : std::nullopt;
                // The existing executor seals the original source plan during
                // capture. Both M1 condition purposes and every verifier width
                // must export that actual composed parent before live dispatch.
                if (!retained || !retained->graph_owner || !retained->composed_parent ||
                    retained->segmented || retained->device!=state_.device_id ||
                    !retained->graph_owner->retainedParticipantSourcePlan() ||
                    !retained->graph_owner->retainedParticipantSourcePlan()->captureIdentity() ||
                    !retained->capture_generation || !retained->capture_variant_signature ||
                    retained->workspace_generation!=workspaceGeneration(state_.device_id))
                    return fail("setup MTP forward lost its exact original retained participant source parent");
            }
        }
        if (setup && !purpose)
        {
            if (!materialized || !forward_engine_->retainedDecodeGraph(*materialized, error))
                return fail("setup verifier has no exact retained native signature");
            const auto &verifier = std::get<MTPVerifierForwardPolicy>(policy);
            const auto outcome_index = verifier.outcome() == MTPVerifierOutcomeGraphMode::Greedy ? 0u : 1u;
            mtp_retained_verifier_signatures_.at(sparseQueryExecutionProgramIndex(sparseQueryGraphProgram()))
                .at(static_cast<std::size_t>(verifier.physicalRows() - 2)).at(outcome_index) = *materialized;
        }
        if (setup)
            PerfStatsCollector::addCounter("forward_graph", "serving_graph_family_materializations",
                1.0, "setup", state_.device_id.toString(),
                {{"role", purpose ? "mtp_condition" : "mtp_grouped_verifier"},
                 {"sparse_query_program", std::to_string(sparseQueryExecutionProgramIndex(sparseQueryGraphProgram()))},
                 {"physical_rows", std::to_string(input.seq_len)},
                 {"commit_purpose", purpose ? std::to_string(static_cast<int>(*purpose)) : "none"},
                 {"draft_depth", std::to_string(input.moe_overlay_mtp_depth)},
                 {"outcome_mode", purpose ? "none" :
                    std::to_string(static_cast<int>(std::get<MTPVerifierForwardPolicy>(policy).outcome()))},
                 {"submission", "materialized_unlaunched"}});
        return true;
    }

    std::vector<RetainedForwardSourceDefinition>
    DeviceGraphOrchestrator::retainedExpertSourceFamilyDefinitions() const
    {
        if (!forward_engine_ || serving_graph_family_lifecycle_ != ServingGraphFamilyLifecycle::Sealed)
            throw std::logic_error("Expert storage attachment requires the sealed original continuation family");
        auto result = forward_engine_->retainedSourceGraphDefinitions(executor_.snapshotConfigurationEpoch());
        const auto append_sidecar = [&](const MTPSidecarGraphCache& cache)
        {
            if (!cache.valid || !cache.graph || !cache.segment_cache.initialized)
                return;
            const auto& native = cache.segment_cache;
            if (native.needs_capture || !native.successful_capture_count ||
                !native.capture_variant_signature || cache.seq_len <= 0 || cache.batch_size <= 0)
                throw std::logic_error("Expert storage attachment found an incomplete original sidecar");
            if (cache.graph->retainedParticipantSourcePlan() ||
                cache.graph->retainedParticipantDefinitionDomain() || native.participant_preparation_stream)
                throw std::logic_error("Learned sidecar storage requires its explicit original source-family association");
            const auto rows = static_cast<int64_t>(cache.seq_len) * cache.batch_size;
            if (rows <= 0 || rows > std::numeric_limits<int>::max())
                throw std::logic_error("Expert storage attachment sidecar row geometry overflows its original finite parent");
            result.push_back(RetainedForwardSourceDefinition::checked(std::nullopt,
                MoEOverlayInferenceGraphRole::MTPDraft,static_cast<int>(rows),*cache.graph,
                native.capture_variant_signature,{}));
        };
        for (const auto* cache : {&mtp_sidecar_depth0_cache_, &mtp_sidecar_depth0_device_token_cache_,
                &mtp_sidecar_depth0_chained_cache_, &mtp_sidecar_depth0_chained_device_token_cache_})
            append_sidecar(*cache);
        return result;
    }

    bool DeviceGraphOrchestrator::materializeServingGraphFamilyWithoutLaunch(
        const ServingGraphFamilyMaterializationPlan &requested_plan)
    {
        auto plan = requested_plan;
        if (pipeline_forward_edges_ && pipeline_forward_edges_->hasHeterogeneousBoundary())
            plan.pipeline_hidden_input = pp_stage_config_ && !pp_stage_config_->has_embedding
                ? state_.hidden.get() : nullptr;
        ScopedDeviceLog device_log(state_.device_id);
        LOG_INFO(
            "[ServingGraphMaterialization] participant begin device="
            << state_.device_id.toString());

        const std::vector<int> normalized_buckets =
            normalizePrefillGraphBuckets(plan.prefill_bucket_rows);
        if (!plan.valid() || normalized_buckets != plan.prefill_bucket_rows)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup requires a "
                "strictly increasing, duplicate-free physical bucket ladder");
            return false;
        }
        if (!state_.isInitialized() || !hasGlobalWeights())
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup requires a "
                "fully initialized runner with model weights");
            return false;
        }
        const uint64_t snapshot_identity =
            executor_.snapshotConfigurationEpoch();
        if (snapshot_identity == 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup has an invalid zero snapshot topology identity");
            return false;
        }
        if (serving_graph_family_plan_ &&
            !serving_graph_family_plan_->sameCaptureIdentity(plan))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph variant changed the sealed capture plan");
            return false;
        }
        if (serving_graph_family_lifecycle_ ==
                ServingGraphFamilyLifecycle::Sealed &&
            materialized_serving_snapshot_identities_.contains(
                snapshot_identity))
        {
            /*
             * Re-selecting a setup-certified diagnostic topology is an
             * idempotent readiness check. Return before inspecting the installed
             * request coordinator: no capture, allocation, reset, or ticket
             * publication is needed after inference admission begins.
             */
            return true;
        }
        if (serving_graph_family_lifecycle_ ==
                ServingGraphFamilyLifecycle::Sealed &&
            (moe_overlay_inference_transaction_coordinator_ ||
             moe_overlay_inference_transaction_participant_index_ >= 0 ||
             moe_overlay_inference_completion_fence_))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] A new serving graph topology cannot be materialized after request-ticket authority is installed");
            return false;
        }
        if (serving_graph_family_lifecycle_ !=
                ServingGraphFamilyLifecycle::Sealed &&
            (moe_overlay_inference_transaction_coordinator_ ||
            moe_overlay_inference_transaction_participant_index_ >= 0 ||
             moe_overlay_inference_completion_fence_))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup must complete "
                "before request ticket authority is installed");
            return false;
        }
        if (!serving_graph_family_plan_)
            serving_graph_family_plan_ = plan;
        if (pp_stage_config_.has_value() &&
            !pp_stage_config_->has_embedding &&
            !plan.pipeline_hidden_input)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Non-embedding PP serving graph setup requires its stable previous-stage activation owner");
            return false;
        }
        if (normalized_buckets.back() > residentGraphRows() ||
            (state_.device_id.is_gpu() &&
             normalized_buckets.back() > prefill_chunk_row_capacity_))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup bucket exceeds "
                "the memory-plan-admitted resident graph capacity"
                << " largest_bucket=" << normalized_buckets.back()
                << " resident_rows=" << residentGraphRows()
                << " chunk_rows=" << prefill_chunk_row_capacity_);
            return false;
        }
        if (state_.device_id.is_gpu() &&
            (!cache_config_.enabled ||
             !debugEnv().execution.gpu_graphs))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Native device serving graph setup requires graph caching and GPU graphs to be enabled");
            return false;
        }

        bool prefix_cache_ready = true;
        if (graph_builder_->config().prefix_cache.enabled)
        {
            ServingGraphMaterializationPhaseTrace phase(
                state_.device_id, "prefix_cache_ready");
            prefix_cache_ready = ensurePrefixCacheReady();
        }
        if (!prefix_cache_ready)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup could not "
                "reserve its configured prefix-cache storage topology: "
                << prefix_cache_bypass_reason_);
            return false;
        }

        /*
         * Immutable model/topology metadata belongs to one prepared builder
         * recipe. Prepare it before the first workspace declaration, so that
         * declaration and every retained executable specialize the same source
         * authority. Mutable buffers, policies and stages remain per graph.
         */
        {
            ServingGraphMaterializationPhaseTrace phase(
                state_.device_id, "forward_template_preparation");
            graph_builder_->prepareForwardGraphTemplate(state_.device_id);
        }

        /*
         * Captured request/chunk rows are arena-owned, but every stage may also
         * retain WorkspaceAllocator addresses. Publish the complete serial
         * family before deriving any capture identity so setup cannot embed a
         * zero or partial workspace generation. This performs graph building
         * only; no model arithmetic or request state is executed.
         */
        bool workspace_family_materialized = false;
        {
            ServingGraphMaterializationPhaseTrace phase(
                state_.device_id, "workspace_family_declaration");
            workspace_family_materialized = materializeForwardGraphForShape(
                /*seq_len=*/1,
                state_.batch_size,
                plan.pipeline_hidden_input);
        }
        if (!workspace_family_materialized)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup could not materialize its complete workspace family");
            return false;
        }

        if (state_.device_id.is_cpu())
        {
            /*
             * Host execution has no native executable to capture. The common
             * workspace-family declaration above nevertheless builds every
             * exact CPU stage topology (including grouped MTP and MoE),
             * validates the frozen maximum-row envelope, and publishes its
             * stable workspace allocation. Construct the long-lived eager
             * engine now so admission certifies the same runtime owner that
             * will build and execute request graphs. No model arithmetic, KV
             * mutation, ticket publication, or synthetic inference occurs.
             */
            {
                ServingGraphMaterializationPhaseTrace phase(
                    state_.device_id, "host_forward_engine");
                ensureForwardEngine();
            }
            serving_graph_family_lifecycle_ =
                ServingGraphFamilyLifecycle::Sealed;
            materialized_serving_snapshot_identities_.insert(
                snapshot_identity);
            PerfStatsCollector::addCounter(
                "forward_graph",
                "serving_graph_family_participant_completions",
                1.0,
                "setup",
                state_.device_id.toString(),
                {{"prefill_buckets",
                  std::to_string(normalized_buckets.size())},
                 {"main_decode_graphs", "1"},
                 {"preparation_kind", "eager_host_graph"},
                 {"request_state_mutations", "0"},
                 {"executable_launches", "0"}});
            LOG_INFO(
                "[ServingGraphMaterialization] participant complete device="
                << state_.device_id.toString()
                << " preparation=eager_host_graph");
            return true;
        }

        ModelBuffers model_buffers = state_.toModelBuffers();
        model_buffers.logits = state_.logits.get();
        model_buffers.logits_local = state_.logits_local.get();
        setBuffers(model_buffers);

        void *const publication_stream =
            explicitGPUStreamForOperation(
                "serving_graph_family_setup_publication");
        if (!publication_stream)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup could not "
                "resolve an explicit device publication stream");
            return false;
        }

        {
            ServingGraphMaterializationPhaseTrace phase(
                state_.device_id, "native_forward_engine");
            ensureForwardEngine();
        }

        if (mtpVerifierGraphFamily() != resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole()))
            throw std::logic_error("Serving verifier capture ownership differs from its admitted native owner family");
        const MTPServingForwardCaptureGeometry mtp_capture_geometry =
            graph_builder_
                ? resolveMTPServingForwardCaptureGeometry(
                      graph_builder_->config().mtp,
                      mtp_max_verifier_rows_, mtpVerifierGraphFamily())
                : MTPServingForwardCaptureGeometry{};
        if (!mtp_capture_geometry.valid() ||
            (mtp_capture_geometry.enabled &&
             (mtp_capture_geometry.verifier_rows <= 0 ||
              mtp_capture_geometry.verifier_rows >
                  mtp_max_verifier_rows_)))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup could not "
                "resolve its retained MTP verifier geometry"
                << " draft_depth="
                << mtp_capture_geometry.draft_depth
                << " physical_rows="
                << mtp_capture_geometry.verifier_rows
                << " maximum_rows=" << mtp_max_verifier_rows_);
            return false;
        }
        if (mtp_capture_geometry.enabled && state_.device_id.is_gpu() &&
            deviceGenerationExecutionPolicy(graph_builder_->config().mtp.depth_policy.mode == MTPDepthPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth : DeviceGenerationLoopTopology::FixedDepth) ==
                DeviceGenerationExecutionPolicy::Unsupported)
            throw std::logic_error("Serving MTP family declaration has no complete native capture implementation");
        const bool mtp_wide_checkpoint_lane_first =
            materializeMTPWideCheckpointLaneFirst(
                snapshot_enabled_,
                normalized_buckets.back(),
                mtp_capture_geometry);

        /*
         * Prefill buckets and grouped verification share one wide checkpoint
         * lane. This closure is invoked before decode for ordinary/shallow MTP
         * families and after grouped-verifier capture when the verifier owns
         * at least as many physical rows. In either ordering the largest graph
         * freezes the address before a smaller alternative can capture it.
         */
        const auto materialize_prefill_family = [&]() -> bool
        {
            ModelBuffers prefill_buffers = state_.toModelBuffers();
            prefill_buffers.logits = state_.logits.get();
            prefill_buffers.logits_local = state_.logits_local.get();
            setBuffers(prefill_buffers);

            for (auto bucket_it = normalized_buckets.rbegin();
                 bucket_it != normalized_buckets.rend();
                 ++bucket_it)
            {
                const int bucket_rows = *bucket_it;
                if (!prepareServingGraphSetupInputs(
                        bucket_rows,
                        publication_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving prefill graph setup could not publish its setup-only request frontier for bucket "
                        << bucket_rows);
                    return false;
                }

                ForwardInput input;
                input.token_ids = nullptr;
                input.token_ids_device = prefill_chunk_token_ids_dev_;
                input.position_ids = nullptr;
                input.position_ids_device = prefill_chunk_position_ids_dev_;
                input.position_policy = ForwardPositionPolicy::ExplicitRows;
                input.execution_role = ForwardExecutionRole::MainInference;
                input.execution_phase = ForwardExecutionPhase::Prefill;
                input.graph_submission_intent =
                    ForwardGraphSubmissionIntent::
                        MaterializeExecutableWithoutLaunch;
                input.batch_size = 1;
                input.seq_len = bucket_rows;
                input.real_seq_len = bucket_rows;
                input.bucket_seq_len = bucket_rows;
                input.position_offset = 0;
                input.token_offset = 0;
                /*
                 * Generation one is a setup-only dynamic value. It is never
                 * published to a follower and never becomes part of cache
                 * identity; the first admitted request overwrites it before the
                 * retained executable is launched.
                 */
                input.moe_overlay_collective_generation_id = 1;
                input.moe_overlay_collective_step_id = 0;
                input.device_state_publication_stream = publication_stream;
                input.device = state_.device_id;
                input.kv_cache = state_.kv_cache.get();
                input.sequence_lengths = nullptr;
                input.sequence_lengths_device = prefill_chunk_real_rows_dev_;
                input.rehydrate_prefix_runtime_on_device = false;
                input.external_hidden_state = plan.pipeline_hidden_input;
                input.device_prefill_chunk =
                    makeDevicePrefillChunkGraphBinding(
                        bucket_rows,
                        plan.prefill_pad_token_id);
                if (!input.device_prefill_chunk)
                    return false;
                if (!bindShiftedMTPPrefillTransaction(
                        input, /*request_count=*/1))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving prefill graph setup "
                        "could not bind its shifted-MTP transaction for bucket "
                        << bucket_rows);
                    return false;
                }
                if (!bindMTPMainTerminalHiddenTransaction(
                        input, /*request_count=*/1))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving prefill graph setup "
                        "could not bind its MTP terminal publication contract");
                    return false;
                }

                ForwardOutput output;
                output.logits = state_.logits.get();
                output.hidden = state_.hidden.get();
                bool bucket_materialized = false;
                {
                    ServingGraphMaterializationPhaseTrace phase(
                        state_.device_id,
                        "prefill_bucket_" + std::to_string(bucket_rows));
                    bucket_materialized = executeForward(input, output);
                }
                if (!bucket_materialized)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Failed to capture and "
                        "instantiate serving prefill bucket "
                        << bucket_rows << " without launch");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "serving_graph_family_materializations",
                    1.0,
                    "setup",
                    state_.device_id.toString(),
                    {{"role", "main_prefill"},
                     {"physical_rows", std::to_string(bucket_rows)},
                     {"submission", "materialized_unlaunched"},
                     {"capacity_authority", "orchestration_memory_plan"}});
            }
            return true;
        };

        if (!mtp_wide_checkpoint_lane_first &&
            !materialize_prefill_family())
        {
            return false;
        }

        /*
         * The first decode arrives only after prompt prefill, so its cache
         * identity is the history-bearing serial variant. External token rows
         * enter REQUEST_TOKEN_IDS once; every native consumer, including PLE
         * and embedding, retains that same bank. The captured root derives
         * position from the canonical device KV count. Setup binds permanent
         * pointers and a history-bearing scalar without publishing live tokens.
         * Request state remains untouched because MaterializeWithoutLaunch
         * captures and instantiates without running the live-state prelude or
         * any model arithmetic.
         */
        if (plan.main_decode_graph !=
                ServingMainDecodeGraphKind::HistoryBearingSerial ||
            (!request_token_ids_dev_ || !request_position_ids_dev_))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Serving graph setup is missing "
                "its history-bearing serial decode inventory or persistent "
                "request rows");
            return false;
        }

        for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
        {
            ServingProgramBufferScope buffer_scope(*this, model_buffers);
            SparseQueryProgramCompilationScope program_scope(*this,
                program == 0 ? SparseQueryExecutionProgram::RankingPossible : SparseQueryExecutionProgram::ProvenNoRanking);
            ForwardInput decode_input;
            decode_input.token_ids = nullptr;
            decode_input.token_ids_device = request_token_ids_dev_;
            decode_input.position_ids = nullptr;
            decode_input.position_ids_device = request_position_ids_dev_;
            decode_input.device_decode_position = DeviceDecodePositionBinding{
                .backend = getBackendFor(state_.device_id),
                .cached_tokens = state_.kv_cache->deviceSequenceCachedTokenCountPtr(0),
                .position = static_cast<int32_t *>(request_position_ids_dev_)};
            decode_input.position_policy = ForwardPositionPolicy::ExplicitRows;
            decode_input.execution_role = ForwardExecutionRole::MainInference;
            decode_input.execution_phase = ForwardExecutionPhase::Decode;
            decode_input.graph_submission_intent =
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch;
            decode_input.batch_size = 1;
            decode_input.seq_len = 1;
            decode_input.real_seq_len = 1;
            decode_input.bucket_seq_len = 0;
            decode_input.position_offset = 1;
            decode_input.token_offset = 1;
            decode_input.moe_overlay_collective_generation_id = 1;
            decode_input.moe_overlay_collective_step_id = 1;
            decode_input.device_state_publication_stream = publication_stream;
            decode_input.device = state_.device_id;
            decode_input.kv_cache = state_.kv_cache.get();
            decode_input.sequence_lengths = nullptr;
            decode_input.sequence_lengths_device = nullptr;
            decode_input.rehydrate_prefix_runtime_on_device = false;
            decode_input.external_hidden_state = plan.pipeline_hidden_input;
            if (!bindShiftedMTPPrefillTransaction(
                    decode_input, /*request_count=*/1))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Serving decode graph setup could "
                    "not bind its production MTP contract");
                return false;
            }
            if (!bindMTPMainTerminalHiddenTransaction(
                    decode_input, /*request_count=*/1))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Serving decode graph setup could "
                    "not bind its graph-integrated MTP terminal publication");
                return false;
            }

            ForwardOutput decode_output;
            decode_output.logits = state_.logits.get();
            decode_output.hidden = state_.hidden.get();
            bool decode_materialized = false;
            {
                ServingGraphMaterializationPhaseTrace phase(
                    state_.device_id, "history_bearing_serial_decode");
                decode_materialized = executeForward(
                    decode_input, decode_output);
            }
            if (!decode_materialized)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to capture and instantiate "
                    "the history-bearing serial decode executable without launch");
                return false;
            }
            PerfStatsCollector::addCounter(
                "forward_graph",
                "serving_graph_family_materializations",
                1.0,
                "setup",
                state_.device_id.toString(),
                {{"role", "main_decode"},
                 {"physical_rows", "1"},
                 {"decode_history", "present"},
                 {"submission", "materialized_unlaunched"},
                 {"capacity_authority", "orchestration_memory_plan"}});

            {
                // The complete generation controller consumes a device-owned
                // condition token. Its input address and pipeline ingress differ
                // from the host-token serial graph above. Retain that exact
                // forward now; OrdinaryGenerationGraphPlan already prices its
                // independent executable. No request or sampler is admitted here.
                ServingGraphMaterializationPhaseTrace phase(
                    state_.device_id, "device_resident_ordinary_forward");
                std::string error;
                if (!materializeOrdinaryGenerationForward(
                        device_resident_logical_sequence_state_storage_.next_condition_tokens_device,
                        ordinaryForwardRetentionKind(), &error))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Serving graph setup could not retain "
                              "its device-resident ordinary forward: " << error);
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph", "serving_graph_family_materializations", 1.0,
                    "setup", state_.device_id.toString(),
                    {{"role", "device_resident_main_decode"}, {"physical_rows", "1"},
                     {"submission", "materialized_unlaunched"},
                     {"capacity_authority", "orchestration_memory_plan"}});
            }

            if (graph_builder_ && retainsMTPGraphCapacity(graph_builder_->config().mtp))
            {
                /*
                 * A one-token suffix following prefix restore shares decode math
                 * with the ordinary history-bearing graph, but its terminal node
                 * additionally advances shifted depth-zero MTP KV. Capture that
                 * complete transaction as a distinct typed family member. The
                 * device token/geometry addresses are the same request-admission
                 * owners used at runtime; no synthetic inference is launched.
                 */
                if (!device_resident_logical_sequence_state_storage_.validFor(
                        /*request_count=*/1))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving graph setup has no "
                        "persistent MTP logical-state mailbox storage");
                    return false;
                }

                TensorBase *bridge_logits_output = state_.logits.get();
                TensorBase *bridge_logits_local_output =
                    state_.logits_local.get();
                if (!bindAllPositionLogitsOutputs(
                        ForwardExecutionRole::MTPCondition,
                        /*rows=*/1,
                        bridge_logits_output,
                        bridge_logits_local_output))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving graph setup could not "
                        "bind the restored-prefix MTP condition logits surface");
                    return false;
                }
                ModelBuffers bridge_model_buffers = state_.toModelBuffers();
                bridge_model_buffers.logits = bridge_logits_output;
                bridge_model_buffers.logits_local =
                    bridge_logits_local_output;
                setBuffers(bridge_model_buffers);

                ForwardInput bridge_input = decode_input;
                bridge_input.token_ids = nullptr;
                bridge_input.token_ids_device =
                    device_resident_logical_sequence_state_storage_
                        .next_condition_tokens_device;
                bridge_input.position_ids_device =
                    device_resident_logical_sequence_state_storage_
                        .target_cached_tokens_device;
                bridge_input.sequence_lengths_device =
                    device_resident_logical_sequence_state_storage_
                        .target_cached_tokens_device;
                bridge_input.device_decode_position.reset();
                bridge_input.position_offset = 0;
                bridge_input.token_offset = 0;
                bridge_input.execution_role =
                    ForwardExecutionRole::MTPCondition;
                bridge_input.state_transaction =
                    ForwardStateTransaction::RestoredPrefixMTPDecodeBridge;
                bridge_input.shifted_mtp_prefill.reset();
                bridge_input.mtp_main_terminal_hidden.reset();
                if (!bindShiftedMTPPrefillTransaction(
                        bridge_input, /*request_count=*/1) ||
                    !bindMTPMainTerminalHiddenTransaction(
                        bridge_input, /*request_count=*/1))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving graph setup could not "
                        "bind its restored-prefix MTP decode bridge");
                    return false;
                }

                ForwardOutput bridge_output;
                bridge_output.logits = bridge_logits_output;
                bridge_output.hidden = state_.hidden.get();
                ScopedLiveMTPRequestBatchConditionPolicy condition_scope(
                    graph_builder_.get(),
                    &live_mtp_request_batch_condition_,
                    /*enable=*/true);
                if (!condition_scope.ready())
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving graph setup could not "
                        "install its restored-prefix MTP condition policy");
                    return false;
                }
                bool bridge_materialized = false;
                {
                    ServingGraphMaterializationPhaseTrace phase(
                        state_.device_id,
                        "restored_prefix_mtp_decode_bridge");
                    bridge_materialized = executeForward(
                        bridge_input, bridge_output);
                }
                if (!bridge_materialized)
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Failed to capture and "
                        "instantiate the restored-prefix MTP decode bridge");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "serving_graph_family_materializations",
                    1.0,
                    "setup",
                    state_.device_id.toString(),
                    {{"role", "restored_prefix_mtp_decode_bridge"},
                     {"physical_rows", "1"},
                     {"decode_history", "present"},
                     {"submission", "materialized_unlaunched"},
                     {"capacity_authority", "orchestration_memory_plan"}});
            }

            {
                ServingGraphMaterializationPhaseTrace phase(
                    state_.device_id, "mtp_forward_executable_family");
                if (!materializeMTPServingForwardExecutablesWithoutLaunch(
                        plan.pipeline_hidden_input,
                        publication_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphOrchestrator] Serving graph setup could not "
                        "materialize its complete MTP forward family");
                    return false;
                }
            }

        }

        if (mtp_wide_checkpoint_lane_first &&
            !materialize_prefill_family())
        {
            return false;
        }

        // This helper owns no model work or payload. Its one native executable
        // is charged by the canonical controller inventory before setup and
        // remains ready across every request/reset, including sampled EOS.
        if (state_.device_id.is_gpu() && mtp_graph_owner_plan_.directTokenHistoryGraphSlots() != 0)
        {
            std::string history_error;
            if (!prepareMTPDirectTokenHistoryPublication(publication_stream, &history_error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Direct MTP history setup failed: " << history_error);
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "serving_graph_family_participant_completions",
            1.0,
            "setup",
            state_.device_id.toString(),
            {{"prefill_buckets", std::to_string(normalized_buckets.size())},
             {"main_decode_graphs", "2"},
             {"request_state_mutations", "0"},
             {"executable_launches", "0"}});

        /*
         * Stream capture does not launch model arithmetic, but stateful stage
         * methods also maintain host-side logical KV/GDN metadata. Recording a
         * prefill executable can therefore advance those mirrors even though
         * its device kernels remain unlaunched. Close setup through the same
         * complete typed state transaction used by serving boundaries. It
         * resets contents behind stable addresses, publishes one exact reset
         * event, and keeps every executable captured above alive. The first
         * admitted request consequently starts at logical position zero and is
         * a replay, never a synthetic warmup transaction.
         */
        {
            ServingGraphMaterializationPhaseTrace phase(
                state_.device_id, "serving_graph_state_reset");
            resetInferenceState(
                InferenceStateResetRequest::servingGraphSetupBoundary(
                    "serving_graph_family_materialization"));
        }
        PerfStatsCollector::addCounter(
            "forward_graph",
            "serving_graph_setup_state_scrubs",
            1.0,
            "setup",
            state_.device_id.toString(),
            {{"owners", "kv_gdn_mtp_model_runtime_logical_sequence"},
             {"captured_executables_preserved", "true"},
             {"request_admitted", "false"}});

        if (moe_overlay_transfer_progress_epoch_)
        {
            const auto progress_stats =
                moe_overlay_transfer_progress_epoch_->stats();
            LOG_DEBUG(
                "[DeviceGraphOrchestrator] Serving graph family sealed without transfer-progress joins"
                << " device=" << state_.device_id.toString()
                << " background_copy_submissions="
                << progress_stats.copy_submissions);
        }
        serving_graph_family_lifecycle_ =
            ServingGraphFamilyLifecycle::Sealed;
        materialized_serving_snapshot_identities_.insert(snapshot_identity);
        LOG_INFO(
            "[ServingGraphMaterialization] participant complete device="
            << state_.device_id.toString()
            << " preparation=native_device_executable_family");
        return true;
    }

} // namespace llaminar2
