/**
 * @file ForwardExecutionEngineDecodeCache.cpp
 * @brief ForwardExecutionEngine construction, cache management, decode graph cache, and cache-hit replay.
 */
#include "ForwardExecutionEngineInternal.h"

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    ForwardExecutionEngine::ForwardExecutionEngine(Config config, DeviceGraphExecutor &executor)
        : config_(std::move(config)), executor_(executor)
    {
        if (config_.sparse_query_program_count == 0u || config_.sparse_query_program_count > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Forward engine requires the original model's closed sparse-query owner roster");
    }

    // =========================================================================
    // Cache Management
    // =========================================================================

    ForwardExecutionEngine::PrefillChunkRuntimePlan ForwardExecutionEngine::prepareSinglePrefillChunkRuntimePlan(
        const ForwardInput &input,
        const std::vector<int> &bucket_sizes,
        int pad_token_id,
        bool allow_padded_execution)
    {
        PrefillChunkRuntimePlan plan;

        if (!input.token_ids && !input.token_ids_device)
        {
            plan.error = "bucketed prefill requires one authoritative token source";
            return plan;
        }
        if (input.token_ids && input.token_ids_device)
        {
            plan.error =
                "bucketed prefill rejects simultaneous host and device token authorities";
            return plan;
        }
        if (input.token_ids_device && !input.device.is_gpu())
        {
            plan.error =
                "bucketed prefill device token input requires a GPU device";
            return plan;
        }
        if (input.batch_size != 1)
        {
            plan.error = "bucketed prefill chunk runner currently supports batch_size=1";
            return plan;
        }

        const int real_seq_len = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
        plan.selection = selectPrefillGraphBucket(real_seq_len, bucket_sizes);
        if (!plan.selection)
        {
            plan.error = plan.selection.error;
            return plan;
        }

        const int token_offset = effectiveTokenOffset(input);

        plan.padding_required = !plan.selection.exact;
        plan.position_policy =
            input.position_ids_device
                ? ForwardPositionPolicy::ExplicitRows
                : input.device.is_gpu()
                      ? ForwardPositionPolicy::ContiguousOffset
                      : ForwardPositionPolicy::ExplicitRows;
        plan.chunk.token_offset = token_offset;
        plan.chunk.real_count = real_seq_len;
        plan.chunk.bucket_seq_len = plan.selection.bucket_seq_len;
        plan.chunk.token_authority = input.token_ids_device
                                         ? PrefillChunkTokenAuthority::DeviceResidentRows
                                         : PrefillChunkTokenAuthority::HostPaddedRows;
        if (plan.chunk.token_authority ==
            PrefillChunkTokenAuthority::HostPaddedRows)
        {
            plan.chunk.token_ids = padPrefillTokensToBucket(
                input.token_ids,
                real_seq_len,
                plan.selection.bucket_seq_len,
                pad_token_id);
        }
        if (plan.position_policy == ForwardPositionPolicy::ExplicitRows &&
            !input.position_ids_device)
        {
            plan.chunk.position_ids = buildPrefillChunkPositionIds(
                real_seq_len,
                plan.selection.bucket_seq_len,
                token_offset,
                input.batch_size);
        }

        if ((plan.chunk.token_authority ==
                 PrefillChunkTokenAuthority::HostPaddedRows &&
             plan.chunk.token_ids.empty()) ||
            (plan.position_policy == ForwardPositionPolicy::ExplicitRows &&
             !input.position_ids_device &&
             plan.chunk.position_ids.empty()))
        {
            plan.error = "failed to prepare bucketed prefill chunk buffers";
            return plan;
        }

        if (plan.padding_required && !allow_padded_execution)
        {
            plan.error = "Bucketed prefill with padding requires caller opt-in to padded execution: real_seq_len=" +
                         std::to_string(real_seq_len) +
                         " bucket_seq_len=" + std::to_string(plan.selection.bucket_seq_len);
            return plan;
        }

        plan.chunk.ok = true;
        plan.ok = true;
        return plan;
    }

    ForwardExecutionEngine::PrefillChunkRuntimeSchedule ForwardExecutionEngine::preparePrefillChunkRuntimeSchedule(
        const ForwardInput &input,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        PrefillChunkRuntimeSchedule runtime_schedule;

        if ((!input.token_ids && !input.token_ids_device) ||
            (input.token_ids && input.token_ids_device))
        {
            runtime_schedule.error =
                "bucketed prefill schedule requires exactly one token authority";
            return runtime_schedule;
        }
        if (input.token_ids_device && !input.device.is_gpu())
        {
            runtime_schedule.error =
                "bucketed prefill device token input requires a GPU device";
            return runtime_schedule;
        }
        if (input.batch_size != 1)
        {
            runtime_schedule.error = "bucketed prefill schedule currently supports batch_size=1";
            return runtime_schedule;
        }

        runtime_schedule.schedule = planPrefillChunkSchedule(policy);
        if (!runtime_schedule.schedule)
        {
            runtime_schedule.error = runtime_schedule.schedule.error;
            return runtime_schedule;
        }
        if (input.token_ids_device &&
            runtime_schedule.schedule.chunks.size() > 1 &&
            !input.device_prefill_chunk.has_value())
        {
            runtime_schedule.error =
                "multi-chunk device prefill requires a captured device chunk materializer";
            return runtime_schedule;
        }

        const int input_real_tokens = effectiveRealSeqLen(input);
        const int input_start = effectiveTokenOffset(input);
        if (input_real_tokens <= 0)
        {
            runtime_schedule.error = "input real token count must be positive";
            return runtime_schedule;
        }
        if (input_start < 0)
        {
            runtime_schedule.error = "input token offset must be non-negative";
            return runtime_schedule;
        }
        const int input_end = input_start + input_real_tokens;

        runtime_schedule.chunks.reserve(runtime_schedule.schedule.chunks.size());
        for (const auto &chunk : runtime_schedule.schedule.chunks)
        {
            if (chunk.token_offset < input_start ||
                chunk.token_offset + chunk.real_count > input_end)
            {
                runtime_schedule.error = "scheduled chunk range is outside input token range";
                runtime_schedule.chunks.clear();
                return runtime_schedule;
            }

            ForwardInput chunk_input = input;
            const int relative_offset = chunk.token_offset - input_start;
            chunk_input.token_ids =
                input.token_ids ? input.token_ids + relative_offset : nullptr;
            chunk_input.token_ids_device = input.token_ids_device;
            chunk_input.seq_len = chunk.real_count;
            chunk_input.real_seq_len = chunk.real_count;
            chunk_input.bucket_seq_len = 0;
            chunk_input.token_offset = chunk.token_offset;
            chunk_input.position_offset = chunk.token_offset;

            /*
             * The scheduler is the sole authority for physical graph width.
             * Prepare against that one selected bucket instead of repeating
             * selection across the policy's complete bucket set.  Repeating
             * the policy decision here allowed a short final chunk to choose a
             * smaller graph even when fixed_chunk_real_tokens deliberately
             * established one immutable capture width for the transaction.
             */
            auto runtime_plan = prepareSinglePrefillChunkRuntimePlan(
                chunk_input,
                std::vector<int>{chunk.bucket_seq_len},
                pad_token_id,
                allow_padded_execution);
            runtime_plan.chunk_index = chunk.chunk_index;
            runtime_plan.rebalance_allowed_after = chunk.rebalance_allowed_after;
            runtime_plan.rebalance_required_after = chunk.rebalance_required_after;

            if (!runtime_plan)
            {
                runtime_schedule.error = runtime_plan.error;
                runtime_schedule.chunks.push_back(std::move(runtime_plan));
                return runtime_schedule;
            }
            if (runtime_plan.chunk.bucket_seq_len != chunk.bucket_seq_len)
            {
                runtime_schedule.error = "runtime chunk bucket does not match scheduler bucket";
                runtime_schedule.chunks.clear();
                return runtime_schedule;
            }

            runtime_schedule.chunks.push_back(std::move(runtime_plan));
        }

        runtime_schedule.ok = true;
        return runtime_schedule;
    }

    bool ForwardExecutionEngine::runPrefillChunk(
        const ForwardInput &base_input,
        const PrefillChunkRuntimePlan &plan,
        ForwardOutput &output,
        IForwardExecutionHost &host)
    {
        if (!plan)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk plan: " << plan.error);
            return false;
        }
        if (!plan.chunk)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk input: " << plan.chunk.error);
            return false;
        }
        if (plan.padding_required &&
            (!config_.cache_config.enabled ||
             !base_input.device.is_gpu() ||
             !debugEnv().execution.gpu_graphs ||
             !debugEnv().execution.prefill_graph_buckets))
        {
            LOG_ERROR("[ForwardExecutionEngine] Padded prefill chunk execution requires GPU graph bucket preflight: real_count="
                      << plan.chunk.real_count << " bucket_seq_len=" << plan.chunk.bucket_seq_len
                      << " device=" << base_input.device.toString());
            return false;
        }

        // The runtime input owns padded token/position buffers, whereas the
        // host-facing hooks deliberately consume the stable scheduler contract.
        // Materialize that contract once so snapshots and chunk maintenance
        // observe identical real-row and bucket geometry.
        const PrefillChunkPlan scheduled_chunk = makeMaintenanceChunkPlan(plan);
        ScopedPrefillChunkSnapshotDiagnostics snapshot_scope(host, scheduled_chunk);

        // Preserve the caller's execution context and swap in the prepared
        // chunk buffers plus fixed-bucket metadata for the delegated launch.
        ForwardInput chunk_input = makeBucketedPrefillInput(base_input, plan);

        std::unique_ptr<IPrefillChunkGraphSubmissionLease> submission_lease;
        std::string submission_error;
        if (!host.beginPrefillChunkGraphSubmission(
                chunk_input,
                scheduled_chunk,
                &submission_lease,
                &submission_error))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Prefill chunk graph submission "
                "admission failed for chunk "
                << scheduled_chunk.chunk_index << ": "
                << (submission_error.empty()
                        ? "external graph authority rejected the chunk"
                        : submission_error));
            return false;
        }

        const bool execution_succeeded = execute(chunk_input, output, host);
        if (submission_lease &&
            !submission_lease->finish(
                execution_succeeded, &submission_error))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Prefill chunk graph submission "
                "could not publish its terminal state for chunk "
                << scheduled_chunk.chunk_index << ": "
                << (submission_error.empty()
                        ? "external graph authority rejected the terminal state"
                        : submission_error));
            return false;
        }
        if (!execution_succeeded)
            return false;

        const PrefillChunkMaintenanceState maintenance_state =
            host.prefillChunkMaintenanceState(scheduled_chunk);
        const PrefillChunkMaintenanceDecision maintenance_decision =
            evaluatePrefillChunkMaintenance(scheduled_chunk, maintenance_state);

        const bool maintenance_was_requested =
            maintenance_state.rebalance_requested ||
            scheduled_chunk.rebalance_required_after;
        if (maintenance_was_requested && !maintenance_decision)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance blocked after chunk "
                      << scheduled_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }
        if (maintenance_decision.can_run &&
            !host.onPrefillChunkMaintenance(scheduled_chunk, maintenance_decision))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance failed after chunk "
                      << scheduled_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }

        return true;
    }

    /**
     * @brief Publish one bucketed prefill chunk's host wall and its preceding gap.
     * @param input Chunk input; supplies the participant identity.
     * @param plan Executed chunk plan; supplies bucket geometry and real rows.
     * @param wall_ns Host wall time of runPrefillChunk for this chunk.
     * @param gap_ns Host time since the previous chunk returned; absent for the first.
     *
     * Bucket length and real rows are bounded admitted geometry, so they are
     * dimensions; the chunk ordinal and token range are sequence words.
     */
    static void recordPrefillChunkHostTiming(
        const ForwardInput &input,
        const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan,
        uint64_t wall_ns,
        std::optional<uint64_t> gap_ns)
    {
        const std::initializer_list<uint64_t> words = {
            static_cast<uint64_t>(plan.chunk_index),
            static_cast<uint64_t>(plan.chunk.token_offset),
            static_cast<uint64_t>(plan.chunk.token_offset + plan.chunk.real_count)};
        const PerfStatsCollector::Tags tags = {
            {"bucket_seq_len", std::to_string(plan.chunk.bucket_seq_len)},
            {"real_token_count", std::to_string(plan.chunk.real_count)},
            {"timing_scope", "host_wall"},
            {"sync_scope", "none"}};
        PerfStatsCollector::recordTimingNsWithSequence(
            "forward_graph", "prefill_chunk_host_wall", wall_ns, words,
            "prefill", input.device.toString(), tags);
        if (gap_ns)
            PerfStatsCollector::recordTimingNsWithSequence(
                "forward_graph", "prefill_inter_chunk_gap", *gap_ns, words,
                "prefill", input.device.toString(), tags);
    }

    bool ForwardExecutionEngine::runPrefillChunkSchedule(
        const ForwardInput &base_input,
        const PrefillChunkRuntimeSchedule &schedule,
        ForwardOutput &output,
        IForwardExecutionHost &host,
        std::span<const ForwardInput> chunk_inputs)
    {
        if (!schedule)
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid prefill chunk schedule: "
                      << schedule.error);
            return false;
        }
        if (schedule.chunks.empty())
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule is empty");
            return false;
        }

        if (!chunk_inputs.empty() &&
            (!base_input.device_prefill_chunk ||
             chunk_inputs.size() != schedule.chunks.size()))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill bucket family requires one complete binding per chunk and one request owner");
            return false;
        }

        // Validate the complete immutable family before submitting any work.
        // Shape selection is request-admission geometry, not a host cursor:
        // each captured materializer still reads the very same device KV count.
        for (size_t index = 0; index < schedule.chunks.size(); ++index)
        {
            const auto &declaration = chunk_inputs.empty() ? base_input : chunk_inputs[index];
            // Validate every request-derived choice before the first chunk can
            // append KV/state. A later foreign declaration must not partially
            // execute the original accepted request.
            if (declaration.sparse_query_prefill_proof != base_input.sparse_query_prefill_proof ||
                declaration.sparse_query_execution_program != base_input.sparse_query_execution_program)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill family changed its original admitted query proof/program");
                return false;
            }
            if (base_input.sparse_query_prefill_proof) {
                const auto &proof = *base_input.sparse_query_prefill_proof;
                if (proof.isMaterialization() || base_input.execution_phase != ForwardExecutionPhase::Prefill ||
                    proof.program(qwen4exp::QwenSparseFirstRankingLogicalRow) != base_input.sparse_query_execution_program)
                    return false;
                try {
                    const auto chunk_proof = proof.chunk(schedule.chunks[index].chunk.token_offset,
                        schedule.chunks[index].chunk.real_count);
                    if (sparseQueryExecutionProgramIndex(chunk_proof.program(qwen4exp::QwenSparseFirstRankingLogicalRow)) >=
                        config_.sparse_query_program_count) return false;
                } catch (const std::invalid_argument &error) {
                    LOG_ERROR("[ForwardExecutionEngine] Invalid admitted prefill chunk: " << error.what());
                    return false;
                }
            } else if (base_input.sparse_query_execution_program != SparseQueryExecutionProgram::RankingPossible) {
                LOG_ERROR("[ForwardExecutionEngine] Prefill family omitted the required original query proof");
                return false;
            }
            if (!base_input.device_prefill_chunk)
                continue;
            const auto &owner = *base_input.device_prefill_chunk;
            if (!declaration.device_prefill_chunk ||
                declaration.device != base_input.device ||
                declaration.batch_size != base_input.batch_size ||
                declaration.device_state_publication_stream != base_input.device_state_publication_stream ||
                declaration.shifted_mtp_prefill.has_value() != base_input.shifted_mtp_prefill.has_value())
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill bucket family has incomplete device or MTP graph ownership at chunk " << index);
                return false;
            }
            const auto &binding = *declaration.device_prefill_chunk;
            if (!binding.valid() ||
                binding.bucket_seq_len != schedule.chunks[index].chunk.bucket_seq_len ||
                binding.backend != owner.backend ||
                binding.request_token_ids_device != owner.request_token_ids_device ||
                binding.request_position_ids_device != owner.request_position_ids_device ||
                binding.request_total_rows_device != owner.request_total_rows_device ||
                binding.cached_tokens_device != owner.cached_tokens_device ||
                binding.chunk_token_ids_device != owner.chunk_token_ids_device ||
                binding.chunk_position_ids_device != owner.chunk_position_ids_device ||
                binding.chunk_real_rows_device != owner.chunk_real_rows_device ||
                binding.chunk_row_stride_device != owner.chunk_row_stride_device ||
                binding.request_row_capacity != owner.request_row_capacity ||
                binding.pad_token_id != owner.pad_token_id ||
                binding.require_exact_rows != owner.require_exact_rows ||
                (binding.require_exact_rows && schedule.chunks[index].chunk.real_count != binding.bucket_seq_len))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill bucket family changes request ownership or lacks the exact physical graph identity at chunk " << index);
                return false;
            }
        }

        // Host wall clocks only: no event, synchronization or device read is
        // added, so the unprofiled pipeline keeps its overlap. Per-chunk wall,
        // the gap between consecutive chunks and the whole family attribute
        // bucket-pipeline overhead against the per-replay launch/GPU timers.
        const bool time_family = PerfStatsCollector::isDomainEnabled("forward_graph");
        using Clock = std::chrono::steady_clock;
        const auto elapsed_ns = [](Clock::time_point begin, Clock::time_point end)
        {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
        };
        const auto family_start = time_family ? Clock::now() : Clock::time_point{};
        Clock::time_point previous_end{};
        for (size_t index = 0; index < schedule.chunks.size(); ++index)
        {
            const auto &chunk_plan = schedule.chunks[index];
            const auto &chunk_base = chunk_inputs.empty() ? base_input : chunk_inputs[index];
            const auto chunk_start = time_family ? Clock::now() : Clock::time_point{};
            if (!runPrefillChunk(chunk_base, chunk_plan, output, host))
            {
                host.cancelPrefillChunkSnapshotDiagnostics();
                LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule failed at chunk "
                          << chunk_plan.chunk_index);
                return false;
            }
            if (time_family)
            {
                const auto chunk_end = Clock::now();
                recordPrefillChunkHostTiming(chunk_base, chunk_plan, elapsed_ns(chunk_start, chunk_end),
                    index > 0 ? std::optional<uint64_t>(elapsed_ns(previous_end, chunk_start)) : std::nullopt);
                previous_end = chunk_end;
            }
        }

        if (!host.finalizePrefillChunkSnapshotDiagnostics(schedule.schedule))
        {
            host.cancelPrefillChunkSnapshotDiagnostics();
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule produced "
                      "incomplete snapshot diagnostics");
            return false;
        }

        if (time_family)
        {
            const auto &last = schedule.chunks.back().chunk;
            PerfStatsCollector::recordTimingNsWithSequence(
                "forward_graph",
                "prefill_family_host_wall",
                elapsed_ns(family_start, Clock::now()),
                {static_cast<uint64_t>(schedule.chunks.size()),
                 static_cast<uint64_t>(schedule.chunks.front().chunk.token_offset),
                 static_cast<uint64_t>(last.token_offset + last.real_count)},
                "prefill",
                base_input.device.toString(),
                {{"timing_scope", "host_wall"}, {"sync_scope", "none"}});
        }

        return true;
    }

    void ForwardExecutionEngine::invalidateAll()
    {
        all_position_verifier_recapture_pending_ = false;
        for (auto &[_, cache] : cache_)
        {
            cache.invalidate();
        }
    }

    void ForwardExecutionEngine::discardAllCachedGraphs()
    {
        invalidateAll();
        cache_.clear();
    }

    size_t ForwardExecutionEngine::invalidateMoEPlacementSensitiveGraphsForStablePlacement(
        const std::function<void()> &before_invalidation)
    {
        size_t invalidated = 0;
        for (auto &[signature, cache] : cache_)
        {
            if (!cache.valid)
                continue;

            const bool graph_stable_single_token_decode =
                signature.decode && !signature.all_position_logits &&
                signature.seq_len == 1 && signature.batch_size <= 1;
            const bool graph_stable_fixed_prefill = !signature.decode;
            if (graph_stable_single_token_decode || graph_stable_fixed_prefill)
            {
                cache.markGPUStreamBindingsDirty();
                continue;
            }

            if (invalidated == 0 && before_invalidation) before_invalidation();
            cache.invalidate();
            ++invalidated;
        }
        return invalidated;
    }

    void ForwardExecutionEngine::resetCapturedReplayState()
    {
        all_position_verifier_recapture_pending_ = false;
        for (auto &entry : cache_)
        {
            entry.second.resetReplayState();
        }
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetSessionReplayState(
        bool preserve_replay_safe_graphs)
    {
        ReplayStateResetSummary summary;
        all_position_verifier_recapture_pending_ = false;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                preserve_replay_safe_graphs
                    ? chooseForwardReplayStateAction(
                          ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                          signature)
                    : ForwardReplayStateAction::ResetReplayState;

            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetSessionState();
                ++summary.reset_replay_state;
                if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                    ++summary.ordinary_decode_reset;
                continue;
            }

            cache.resetSessionStatePreservingGraphReplay();
            ++summary.preserved_for_stream_rebind;
            if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
                ++summary.all_position_verifier_preserved;
            else if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                     cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                ++summary.other_preserved;
            else
                ++summary.other_preserved;
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetCapturedReplayStateForCorrectionReplay(
        uint64_t live_state_epoch,
        bool preserve_single_token_decode_replay)
    {
        ReplayStateResetSummary summary;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            ForwardReplayStateAction action =
                chooseForwardReplayStateAction(
                    ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                    signature);
            if (!preserve_single_token_decode_replay &&
                (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                 cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode))
            {
                action = ForwardReplayStateAction::ResetReplayState;
            }
            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                    ++summary.ordinary_decode_reset;
            }
            else
            {
                cache.markGPUStreamBindingsDirty();
                cache.markReplayStateSafeForLiveEpoch(live_state_epoch);
                ++summary.preserved_for_stream_rebind;
                if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
                    ++summary.all_position_verifier_preserved;
                else
                    ++summary.other_preserved;
            }
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::rebindCapturedReplayStateAfterPrefixRestore(
        uint64_t live_state_epoch)
    {
        ReplayStateResetSummary summary;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                chooseForwardReplayStateAction(
                    ForwardReplayStateMutationKind::PrefixCheckpointRestore,
                    signature);
            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                if (cache_class ==
                        ForwardReplayStateCacheClass::OrdinaryDecode ||
                    cache_class ==
                        ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
                {
                    ++summary.ordinary_decode_reset;
                }
                continue;
            }

            /*
             * Prefix restore also retires request-local token, position, and
             * sequence-length metadata from the discarded timeline.  Reset that
             * metadata through the captured-replay-preserving stage API: this
             * keeps the graph executable and stable buffers, but guarantees that
             * the next invocation republishes every dynamic row before replay.
             * The restore producer event still owns all device writes, and the
             * epoch stamp prevents the cache from being mistaken for an
             * unversioned capture.
             */
            cache.resetSessionStatePreservingGraphReplay();
            cache.markReplayStateSafeForLiveEpoch(live_state_epoch);
            ++summary.preserved_for_stream_rebind;
            if (cache_class ==
                ForwardReplayStateCacheClass::AllPositionVerifier)
            {
                ++summary.all_position_verifier_preserved;
            }
            else
            {
                ++summary.other_preserved;
            }
        }
        return summary;
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetAllPositionVerifierReplayState()
    {
        ReplayStateResetSummary summary;
        all_position_verifier_recapture_pending_ = true;
        summary.all_position_verifier_recapture_requested = true;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            if (cache_class == ForwardReplayStateCacheClass::AllPositionVerifier)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                continue;
            }

            cache.markGPUStreamBindingsDirty();
            ++summary.preserved_for_stream_rebind;
            if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode ||
                cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
            {
                ++summary.other_preserved;
            }
        }
        return summary;
    }

    void ForwardExecutionEngine::touchPrefillForwardCache(
        const ForwardGraphSignature &signature,
        ForwardGraphCache &cache)
    {
        if (signature.decode || !cache.valid)
            return;
        cache.prefill_last_access_tick = ++prefill_forward_access_counter_;
    }

    size_t ForwardExecutionEngine::prefillForwardCacheSize() const
    {
        size_t count = 0;
        for (const auto &[signature, cache] : cache_)
        {
            if (!signature.decode && cache.valid)
                ++count;
        }
        return count;
    }

    void ForwardExecutionEngine::enforcePrefillForwardCapacity(
        const IForwardExecutionHost &host, const ForwardGraphSignature *active_signature)
    {
        const auto configured_cap = host.retainedExactPrefillCacheEntries().value_or(
            static_cast<std::size_t>(debugEnv().execution.prefill_graph_max_cached_buckets));
        if (configured_cap == 0)
            return;

        // The option bounds admitted physical bucket owners; the closed QSA
        // programs retain distinct complete executables for each such bucket.
        const size_t cap = static_cast<size_t>(configured_cap) * config_.sparse_query_program_count;
        while (prefillForwardCacheSize() > cap)
        {
            auto victim = cache_.end();
            uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();

            // Exact CPU prefill and bucketed GPU prefill both retain complete
            // graph topology and stage-owned persistent buffers. Bound them as
            // one cache class; decode has its independent stable-shape lifetime.
            for (auto it = cache_.begin(); it != cache_.end(); ++it)
            {
                const auto &signature = it->first;
                const auto &cache = it->second;
                if (signature.decode || !cache.valid)
                    continue;
                if (active_signature && signature == *active_signature)
                    continue;

                const uint64_t tick = cache.prefill_last_access_tick;
                if (victim == cache_.end() ||
                    tick < oldest_tick ||
                    (tick == oldest_tick && prefillSignatureLessForEviction(signature, victim->first)))
                {
                    oldest_tick = tick;
                    victim = it;
                }
            }

            if (victim == cache_.end())
                return;

            const auto evicted_signature = victim->first;
            victim->second.invalidate();
            cache_.erase(victim);
            ++prefill_forward_eviction_count_;

            LOG_INFO("[ForwardExecutionEngine] Evicted prefill forward topology rows="
                     << (evicted_signature.is_bucketed_prefill
                             ? evicted_signature.bucket_seq_len
                             : evicted_signature.seq_len)
                     << " bucketed=" << evicted_signature.is_bucketed_prefill
                     << " device=" << evicted_signature.device.toString()
                     << " due to cache cap=" << configured_cap);
        }
    }

    // =========================================================================
    // execute() — Entry Point
    // =========================================================================

    bool ForwardExecutionEngine::execute(
        const ForwardInput &input,
        ForwardOutput &output,
        IForwardExecutionHost &host,
        std::optional<ForwardGraphSignature> *materialized_signature)
    {
        // Geometry manifests and unknown values cannot resolve a GPU context,
        // consult an executable cache, resize workspace, or submit any graph.
        if(!executableForwardSubmissionIntent(input.graph_submission_intent))
        {
            LOG_ERROR("[ForwardExecutionEngine] Workspace declaration or unknown submission intent is not executable");
            return false;
        }
        if (input.captured_verifier_extent.isExact() &&
            (input.execution_role != ForwardExecutionRole::GroupedMTPVerifier ||
             input.execution_phase != ForwardExecutionPhase::Decode || input.batch_size != 1 ||
             input.seq_len < 2 || input.bucket_seq_len != 0 ||
             (input.real_seq_len != 0 && input.real_seq_len != input.seq_len) ||
             !input.sequence_lengths_device ||
             !input.captured_verifier_extent.matches(input.device,
                 DeviceRowRange::deviceCounted(input.seq_len, input.sequence_lengths_device))))
            throw std::invalid_argument("Exact hosted verifier input changed its original role, device or rows");
        (void)sparseQueryExecutionProgramIndex(input.sparse_query_execution_program);
        if (input.sparse_query_prefill_proof)
        {
            const auto &proof = *input.sparse_query_prefill_proof;
            const bool materializing = input.graph_submission_intent ==
                ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch;
            const int rows = input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
            if (input.execution_phase != ForwardExecutionPhase::Prefill || input.batch_size != 1 ||
                sparseQueryExecutionProgramIndex(input.sparse_query_execution_program) >= config_.sparse_query_program_count ||
                (proof.isMaterialization() ? !materializing : !proof.namesRange(effectiveTokenOffset(input), rows)) ||
                proof.program(qwen4exp::QwenSparseFirstRankingLogicalRow) != input.sparse_query_execution_program)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill query topology lacks its exact original admitted range or cold materialization");
                return false;
            }
        }
        if (input.sparse_query_execution_program == SparseQueryExecutionProgram::ProvenNoRanking &&
            input.execution_phase != ForwardExecutionPhase::Decode &&
            !(input.execution_phase == ForwardExecutionPhase::Prefill && input.sparse_query_prefill_proof))
        {
            LOG_ERROR("[ForwardExecutionEngine] No-ranking topology requires explicit Decode or proved Prefill admission");
            return false;
        }
        if(input.device_request_query)requireDeviceRequestQueryOwnsInput(input);
        if (materialized_signature)
            materialized_signature->reset();
        last_executed_forward_graph_ = {};
        // Validate before a cache lookup: a matching shape must never hide a
        // malformed or foreign position producer behind an older executable.
        if (input.device_decode_position &&
            (!input.device_decode_position->valid() || !input.device.is_gpu() ||
             input.execution_phase != ForwardExecutionPhase::Decode ||
             input.execution_role != ForwardExecutionRole::MainInference ||
             input.batch_size != 1 || input.seq_len != 1 || input.position_ids ||
             input.position_ids_device != input.device_decode_position->position ||
             input.position_policy != ForwardPositionPolicy::ExplicitRows))
        {
            LOG_ERROR("[ForwardExecutionEngine] Invalid graph-owned decode position binding");
            return false;
        }
        if (input.pipeline_activation_ownership != ForwardPipelineActivationOwnership::ForwardGraph &&
            (!config_.pipeline_forward_edges || config_.pipeline_forward_edges->encloses(input)))
        {
            LOG_ERROR("[ForwardExecutionEngine] Parent-owned activation input lacks its exact native pipeline setup owner");
            return false;
        }
        ScopedWorkerGPUContextResolver worker_resolver_scope(
            executor_,
            host,
            input.device);
        const bool setup_materialization =
            input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch;
        if (setup_materialization &&
            (!input.device.is_gpu() ||
             input.moe_overlay_graph_launch_dependency))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup-only forward graph "
                "materialization requires a GPU input with no request-scoped "
                "launch dependency");
            return false;
        }

        if (input.state_transaction == ForwardStateTransaction::ForwardOnlyCondition &&
            (!setup_materialization || !config_.pipeline_forward_edges ||
             input.execution_role != ForwardExecutionRole::MainInference ||
             input.execution_phase != ForwardExecutionPhase::Decode ||
             input.batch_size != 1 || input.seq_len != 1 || !input.device_decode_position))
        {
            LOG_ERROR("ForwardOnly condition requires its original retained pipeline one-row source owner");
            return false;
        }

        /*
         * Workspace-family discovery builds the exact production topology with
         * permanent arena/KV addresses before a workspace generation exists.
         * That typed declaration is never executable: only the runtime binding
         * whose capture identity includes the finalized generation may cross
         * this boundary.
         */
        const bool graph_integrated_shifted_mtp =
            isGraphIntegratedShiftedMTPTransaction(
                input.execution_role,
                input.execution_phase,
                input.state_transaction,
                input.batch_size,
                input.seq_len);
        if (input.shifted_mtp_prefill &&
            (!input.shifted_mtp_prefill->executableForRequestCount(
                 input.batch_size) ||
             !graph_integrated_shifted_mtp))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Refusing a declaration-only or "
                "incomplete shifted-MTP prefill binding");
            return false;
        }
        if (input.mtp_main_terminal_logits &&
            (input.mtp_main_terminal_logits->declaration_only ||
             !input.mtp_main_terminal_logits->validForRequestCount(input.batch_size)))
        {
            LOG_ERROR("[ForwardExecutionEngine] MTP terminal-logit execution requires original finalized bindings");
            return false;
        }
        if (input.mtp_main_terminal_hidden &&
            (!input.mtp_main_terminal_hidden->executableForRequestCount(
                 input.batch_size) ||
             input.shifted_mtp_prefill.has_value() ||
             (input.state_transaction != ForwardStateTransaction::Ordinary &&
              !(input.state_transaction == ForwardStateTransaction::ForwardOnlyCondition &&
                input.execution_role == ForwardExecutionRole::MainInference &&
                input.batch_size == 1 && input.seq_len == 1) &&
              !(input.state_transaction == ForwardStateTransaction::CommittedMTPCondition &&
                input.execution_role == ForwardExecutionRole::MTPCondition &&
                input.batch_size == 1 && input.seq_len == 1)) ||
             input.execution_phase != ForwardExecutionPhase::Decode ||
             (input.execution_role != ForwardExecutionRole::MainInference &&
              input.execution_role != ForwardExecutionRole::MTPCondition)))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Refusing an incomplete or "
                "mis-scoped MTP main terminal-hidden binding");
            return false;
        }

        auto start = std::chrono::high_resolution_clock::now();

        const int first_position = input.position_ids ? input.position_ids[0] : input.position_offset;
        /*
         * The public orchestrator resolves legacy shape/history heuristics once
         * and publishes the result through ForwardInput::execution_phase. This
         * lower execution boundary must consume that typed decision verbatim.
         * Reclassifying M=1 here used to turn a one-token request prefill into a
         * decode-shaped continuation graph after the heterogeneous overlay had
         * already admitted a padded prefill transaction, leaving the two sides
         * with incompatible sparse-collective timelines.
         */
        const bool all_position_logits = host.computeAllPositionLogitsEnabled();
        const bool live_mtp_request_batch_condition =
            host.liveMTPRequestBatchConditionEnabled();
        const bool grouped_mtp_verifier =
            input.execution_role ==
            ForwardExecutionRole::GroupedMTPVerifier;
        if (grouped_mtp_verifier)
        {
            /*
             * A new verifier attempt owns the publication handle. Clear any
             * earlier row-snapshot producer before execution so a failed
             * verifier cannot leave stale snapshots publishable. All-position
             * logits alone are insufficient: condition and diagnostic graphs
             * may also expose multiple rows but do not own verifier state.
             */
            clearLastAllPositionVerifierForwardGraph();
        }
        const bool mtp_condition =
            input.execution_role == ForwardExecutionRole::MTPCondition;
        const bool role_requires_decode =
            grouped_mtp_verifier || mtp_condition;
        const bool is_decode =
            input.execution_phase == ForwardExecutionPhase::Decode;
        if (live_mtp_request_batch_condition && !mtp_condition)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Live MTP condition policy was "
                "submitted without the typed MTPCondition execution role");
            return false;
        }
        if (role_requires_decode && !is_decode)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Decode-owned execution role was "
                "submitted with prefill phase (role="
                << static_cast<int>(input.execution_role)
                << ", live_request_batch_condition="
                << live_mtp_request_batch_condition << ")");
            return false;
        }
        const bool decode_has_history = is_decode && first_position > 0;
        const bool is_standard_path = !config_.pp_stage_config.has_value();
        const bool is_partial_pp_path = config_.pp_stage_config.has_value();

        // =====================================================================
        // Decode Graph Cache: Reuse cached graph for decode mode (seq_len=1)
        // =====================================================================
        // During decode, the graph structure is identical between steps —
        // only token_ids, position_ids, and position_offset change.
        // Instead of rebuilding hundreds of stage objects every forward() call,
        // we cache the graph after the first decode step and reuse it.
        //
        // Benefits:
        // - Eliminates stage object construction/destruction (~100s of allocs)
        // - Preserves kernel caches in stages (JIT attention, RoPE inv_freq)
        // - Avoids workspace re-binding (bindWorkspace → inv_freq reset)
        // - Skips graph traversal in ensureDeviceWorkspaceAllocated()
        // =====================================================================

        // PP non-embedding stages receive hidden state instead of tokens,
        // so token_ids is legitimately nullptr. They still have stable inputs
        // (position_ids for RoPE, hidden state via setHiddenState).
        const bool is_pp_non_embedding_stage =
            config_.pp_stage_config.has_value() && !config_.pp_stage_config->has_embedding;
        const bool has_position_input =
            input.position_ids != nullptr ||
            (input.position_ids_device != nullptr && input.device.is_gpu()) ||
            (input.position_policy == ForwardPositionPolicy::ContiguousOffset &&
             input.batch_size == 1 &&
             input.position_ids == nullptr &&
             input.position_ids_device == nullptr);
        const bool has_stable_forward_inputs =
            (((input.token_ids != nullptr) || (input.token_ids_device != nullptr)) &&
             has_position_input) ||
            (is_pp_non_embedding_stage && has_position_input);

        const auto &env = debugEnv();
        const int resident_graph_rows = host.residentGraphRows();
        const auto resident_prefill_buckets =
            prefillGraphBucketsAtOrBelowCapacity(
                env.execution.prefill_graph_bucket_sizes,
                resident_graph_rows);
        const int raw_prefill_bucket_floor =
            effectivePrefillGraphMinimumPaddedBucketSeqLen(
                env.execution.prefill_graph_min_seq,
                resident_graph_rows);
        const std::vector<int> raw_prefill_buckets =
            rawPrefillGraphBucketsForResidentCapacity(
                env.execution.prefill_graph_bucket_sizes,
                resident_graph_rows,
                env.execution.prefill_graph_min_seq);
        /*
         * Forward topology reuse and native GPU graph capture are separate
         * policies. Exact-shape CPU prefill keeps its ComputeGraph, prepared
         * kernels, and stage-owned persistent workspace warm, then executes it
         * through the ordinary CPU executor. Homogeneous GPU prefill additionally
         * enters the bucketed capture/replay state machine below.
         */
        const bool prefill_topology_cache_eligible =
            config_.cache_config.enabled &&
            !is_decode &&
            has_stable_forward_inputs &&
            (is_standard_path || is_partial_pp_path);
        const bool cpu_exact_prefill_cache_eligible =
            prefill_topology_cache_eligible && input.device.is_cpu();
        const bool prefill_cache_eligible =
            prefill_topology_cache_eligible &&
            input.device.is_gpu() &&
            env.execution.gpu_graphs;

        ForwardInput effective_input = input;
        std::optional<PrefillChunkRuntimePlan> raw_bucket_plan;
        bool bucketed_prefill = false;
        int bucketed_prefill_seq_len = input.seq_len;

        if (!is_decode && input.bucket_seq_len > 0 && isPaddedBucketExecution(input) &&
            (!prefill_cache_eligible || !env.execution.prefill_graph_buckets))
        {
            LOG_ERROR("[ForwardExecutionEngine] Padded bucketed prefill input requires GPU graph bucket preflight: seq_len="
                      << input.seq_len << " real_seq_len=" << effectiveRealSeqLen(input)
                      << " bucket_seq_len=" << effectiveBucketSeqLen(input)
                      << " device=" << input.device.toString());
            return false;
        }

        const bool bucketed_prefill_eligible =
            prefill_cache_eligible &&
            env.execution.prefill_graph_buckets &&
            input.batch_size == 1;

        if (prefill_cache_eligible && env.execution.prefill_graph_buckets && input.batch_size > 1)
        {
            LOG_DEBUG("[ForwardExecutionEngine] Skipping bucketed prefill for batched request prefill: batch_size="
                      << input.batch_size << " seq_len=" << input.seq_len
                      << " device=" << input.device.toString());
        }

        if (bucketed_prefill_eligible)
        {
            const bool already_bucketed_input = input.bucket_seq_len > 0;
            if (already_bucketed_input)
            {
                const int real_seq_len = effectiveRealSeqLen(input);
                if (input.seq_len != input.bucket_seq_len || real_seq_len > input.bucket_seq_len)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prepared bucketed prefill input has inconsistent shape: seq_len="
                              << input.seq_len << " real_seq_len=" << real_seq_len
                              << " bucket_seq_len=" << input.bucket_seq_len);
                    return false;
                }
                if (resident_graph_rows > 0 &&
                    input.bucket_seq_len > resident_graph_rows)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prepared prefill bucket "
                              << input.bucket_seq_len
                              << " exceeds memory-planned resident graph rows "
                              << resident_graph_rows);
                    return false;
                }
                bucketed_prefill = true;
                bucketed_prefill_seq_len = input.bucket_seq_len;
            }
            else
            {
                if (raw_prefill_buckets.empty())
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] No resident prefill graph bucket meets the configured raw-prompt floor: floor="
                        << raw_prefill_bucket_floor
                        << " resident_graph_rows=" << resident_graph_rows
                        << "; correct the bucket/capacity contract instead of executing eagerly");
                    return false;
                }
                ForwardInput planning_input = input;
                planning_input.token_offset = effectiveTokenOffset(input);
                raw_bucket_plan = prepareSinglePrefillChunkRuntimePlan(
                    planning_input,
                    raw_prefill_buckets,
                    env.execution.prefill_graph_pad_token_id,
                    /*allow_padded_execution=*/true);
                if (!raw_bucket_plan || !raw_bucket_plan->chunk)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph request rejected: "
                              << (raw_bucket_plan ? raw_bucket_plan->error : std::string("failed to prepare bucketed input"))
                              << " (seq_len=" << input.seq_len << ")");
                    return false;
                }
                effective_input = makeBucketedPrefillInput(input, *raw_bucket_plan);
                bucketed_prefill = true;
                bucketed_prefill_seq_len = raw_bucket_plan->chunk.bucket_seq_len;
            }
        }

        const bool forward_cache_eligible =
            (config_.cache_config.enabled &&
             is_decode &&
             has_stable_forward_inputs &&
             (is_standard_path || is_partial_pp_path)) ||
            cpu_exact_prefill_cache_eligible ||
            prefill_cache_eligible;

        ForwardGraphSignature forward_signature;
        ForwardGraphCache *active_forward_cache = nullptr;

        if (forward_cache_eligible)
        {
            int pp_first_layer = -1;
            int pp_last_layer = -1;
            bool pp_has_embedding = false;
            bool pp_has_lm_head = false;
            if (config_.pp_stage_config.has_value())
            {
                const auto &pp = config_.pp_stage_config.value();
                pp_first_layer = pp.first_layer;
                pp_last_layer = pp.last_layer;
                pp_has_embedding = pp.has_embedding;
                pp_has_lm_head = pp.has_lm_head;
            }

            forward_signature = ForwardGraphSignature{
                .seq_len = bucketed_prefill
                               ? bucketed_prefill_seq_len
                               : effective_input.seq_len,
                .batch_size = effective_input.batch_size,
                .device = effective_input.device,
                .execution_role = effective_input.execution_role,
                .state_transaction = effective_input.state_transaction,
                .captured_verifier_extent = effective_input.captured_verifier_extent,
                .decode = is_decode,
                .sparse_query_execution_program = effective_input.sparse_query_execution_program,
                .decode_has_history = decode_has_history,
                .all_position_logits = all_position_logits,
                .live_mtp_request_batch_condition =
                    live_mtp_request_batch_condition,
                .all_position_logit_rows =
                    all_position_logits
                        ? std::max(0, host.allPositionLogitRows())
                        : 0,
                .mtp_verifier_outcome_graph_mode =
                    host.mtpVerifierOutcomeGraphMode(),
                .device_token_ids = effective_input.token_ids_device,
                .device_position_ids = effective_input.position_ids_device,
                .position_policy = effective_input.position_policy,
                .device_sequence_lengths = effective_input.sequence_lengths_device,
                .device_decode_position = effective_input.device_decode_position,
                .device_request_query_capture_identity = effective_input.device_request_query
                    ? effective_input.device_request_query->captureIdentity() : uint64_t{0},
                .device_prefill_chunk_capture_identity =
                    effective_input.device_prefill_chunk
                        ? effective_input.device_prefill_chunk->capture_identity
                        : uint64_t{0},
                .pipeline_forward_edges = config_.pipeline_forward_edges &&
                    (config_.pipeline_forward_edges->encloses(effective_input) ||
                     effective_input.pipeline_activation_ownership == ForwardPipelineActivationOwnership::NativeGenerationParent)
                    ? config_.pipeline_forward_edges : nullptr,
                .pipeline_activation_ownership = effective_input.pipeline_activation_ownership,
                .shifted_mtp_prefill_capture_identity =
                    effective_input.shifted_mtp_prefill
                        ? effective_input.shifted_mtp_prefill->capture_identity
                        : uint64_t{0},
                .mtp_main_terminal_logits_capture_identity = effective_input.mtp_main_terminal_logits
                    ? effective_input.mtp_main_terminal_logits->capture_identity : uint64_t{0},
                .mtp_main_terminal_hidden_capture_identity =
                    effective_input.mtp_main_terminal_hidden
                        ? effective_input.mtp_main_terminal_hidden
                              ->capture_identity
                        : uint64_t{0},
                .standard_path = is_standard_path,
                .pp_stage_enabled = config_.pp_stage_config.has_value(),
                .pp_first_layer = pp_first_layer,
                .pp_last_layer = pp_last_layer,
                .pp_has_embedding = pp_has_embedding,
                .pp_has_lm_head = pp_has_lm_head,
                .is_bucketed_prefill = bucketed_prefill,
                .bucket_seq_len =
                    bucketed_prefill ? bucketed_prefill_seq_len : 0,
                .rehydrate_prefix_runtime_on_device =
                    effective_input.rehydrate_prefix_runtime_on_device,
                .moe_placement_epoch = host.moePlacementEpoch(),
                .snapshot_configuration_identity =
                    executor_.snapshotConfigurationEpoch()};

            auto cache_it = cache_.find(forward_signature);
            if (cache_it != cache_.end())
            {
                active_forward_cache = &cache_it->second;
                bindGraphSnapshotArenaReusePolicy(
                    *active_forward_cache,
                    forward_signature);
                active_forward_cache->segment_cache.replay_workload =
                    replayWorkloadGeometryForSignature(forward_signature);
            }
        }

        const bool use_cached_forward = forward_cache_eligible &&
                                        active_forward_cache &&
                                        active_forward_cache->valid;

        if (forward_cache_eligible)
        {
            LOG_TRACE(
                "[ForwardGraphCacheIdentity] result="
                << (use_cached_forward ? "hit" : "miss")
                << " submission="
                << (setup_materialization ? "setup_materialize" : "runtime")
                << " seq_len=" << forward_signature.seq_len
                << " batch=" << forward_signature.batch_size
                << " device=" << forward_signature.device.toString()
                << " role="
                << static_cast<int>(forward_signature.execution_role)
                << " decode=" << boolTag(forward_signature.decode)
                << " history="
                << boolTag(forward_signature.decode_has_history)
                << " all_logits="
                << boolTag(forward_signature.all_position_logits)
                << " device_tokens="
                << boolTag(forward_signature.usesDeviceTokenIds())
                << " device_positions="
                << boolTag(forward_signature.usesDevicePositionIds())
                << " position_policy="
                << static_cast<int>(forward_signature.position_policy)
                << " device_lengths="
                << boolTag(forward_signature.usesDeviceSequenceLengths())
                << " chunk_identity="
                << forward_signature.device_prefill_chunk_capture_identity
                << " shifted_mtp_identity="
                << forward_signature.shifted_mtp_prefill_capture_identity
                << " mtp_terminal_hidden_identity="
                << forward_signature
                       .mtp_main_terminal_hidden_capture_identity
                << " bucketed="
                << boolTag(forward_signature.is_bucketed_prefill)
                << " bucket_rows=" << forward_signature.bucket_seq_len
                << " prefix_rehydrate="
                << boolTag(
                       forward_signature.rehydrate_prefix_runtime_on_device)
                      << " placement_epoch="
                      << forward_signature.moe_placement_epoch
                      << " snapshot_identity="
                      << forward_signature.snapshot_configuration_identity);

            /*
             * A production GPU runner seals its admitted serving family before
             * request admission. A runtime miss is therefore rare diagnostic
             * evidence that setup and live graph identity disagree. Keep hits
             * and setup materialization at TRACE, but surface this one bounded
             * event so cold-capture latency cannot hide inside inference.
             */
            if (!use_cached_forward && !setup_materialization &&
                forward_signature.device.is_gpu())
            {
                LOG_INFO(
                    "[ForwardGraphCacheIdentity] runtime_materialization"
                    << " signature_hash="
                    << ForwardGraphSignatureHash{}(forward_signature)
                    << " context="
                    << forwardGraphPerfContext(forward_signature)
                    << " seq_len=" << forward_signature.seq_len
                    << " batch=" << forward_signature.batch_size
                    << " device=" << forward_signature.device.toString()
                    << " role="
                    << static_cast<int>(forward_signature.execution_role)
                    << " state_transaction="
                    << static_cast<int>(forward_signature.state_transaction)
                    << " history="
                    << boolTag(forward_signature.decode_has_history)
                    << " all_logits="
                    << boolTag(forward_signature.all_position_logits)
                    << " all_logit_rows="
                    << forward_signature.all_position_logit_rows
                    << " live_mtp_condition="
                    << boolTag(
                           forward_signature
                               .live_mtp_request_batch_condition)
                    << " device_tokens="
                    << boolTag(forward_signature.usesDeviceTokenIds())
                    << " device_positions="
                    << boolTag(forward_signature.usesDevicePositionIds())
                    << " position_policy="
                    << static_cast<int>(forward_signature.position_policy)
                    << " device_lengths="
                    << boolTag(
                           forward_signature
                               .usesDeviceSequenceLengths())
                    << " chunk_identity="
                    << forward_signature
                           .device_prefill_chunk_capture_identity
                    << " shifted_mtp_identity="
                    << forward_signature
                           .shifted_mtp_prefill_capture_identity
                    << " terminal_hidden_identity="
                    << forward_signature
                           .mtp_main_terminal_hidden_capture_identity
                    << " bucket_rows="
                    << forward_signature.bucket_seq_len
                    << " prefix_rehydrate="
                    << boolTag(
                           forward_signature
                               .rehydrate_prefix_runtime_on_device)
                    << " placement_epoch="
                    << forward_signature.moe_placement_epoch);
            }
        }

        if (live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=cache_decision"
                << " device=" << effective_input.device.toString()
                << " result=" << (use_cached_forward ? "hit" : "miss")
                << " seq_len=" << forward_signature.seq_len
                << " batch_size=" << forward_signature.batch_size
                << " device_tokens="
                << boolTag(forward_signature.usesDeviceTokenIds())
                << " device_positions="
                << boolTag(forward_signature.usesDevicePositionIds())
                << " device_lengths="
                << boolTag(
                       forward_signature.usesDeviceSequenceLengths())
                << " placement_epoch="
                << forward_signature.moe_placement_epoch);
        }

        if (use_cached_forward)
        {
            auto lookup_tags =
                forwardCacheLookupTags(forward_signature, "hit");
            lookup_tags.emplace(
                "submission",
                setup_materialization ? "setup_materialize" : "runtime");
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                std::move(lookup_tags));
            touchPrefillForwardCache(forward_signature, *active_forward_cache);
            const bool success = executeCacheHit(effective_input, output, *active_forward_cache, host,
                                                 forward_signature, start);
            if (success && materialized_signature)
                *materialized_signature = forward_signature;
            if (setup_materialization)
            {
                if (success && !forward_signature.decode)
                    enforcePrefillForwardCapacity(host, &forward_signature);
                return success;
            }
            if (live_mtp_request_batch_condition &&
                debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
            {
                LOG_INFO(
                    "[MTPConditionGraphContract] event=execution_complete"
                    << " device=" << effective_input.device.toString()
                    << " cache=hit"
                    << " success=" << boolTag(success));
            }
            if (success)
            {
                recordLastExecutedForwardGraph(
                    forward_signature,
                    /*cache_hit=*/true,
                    output.execution.stream);
            }
            if (success && !forward_signature.decode)
                enforcePrefillForwardCapacity(host, &forward_signature);
            if (success)
                host.commitSuccessfulForwardOutput(output);
            return success;
        }

        // Cache MISS path
        ForwardGraphCache *build_cache = nullptr;
        bool should_cache_after_build = false;
        if (forward_cache_eligible)
        {
            auto lookup_tags =
                forwardCacheLookupTags(forward_signature, "miss");
            lookup_tags.emplace(
                "submission",
                setup_materialization ? "setup_materialize" : "runtime");
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                std::move(lookup_tags));
            auto [it, _inserted] = cache_.try_emplace(forward_signature);
            build_cache = &it->second;
            bindGraphSnapshotArenaReusePolicy(
                *build_cache,
                forward_signature);
            build_cache->segment_cache.replay_workload =
                replayWorkloadGeometryForSignature(forward_signature);
            should_cache_after_build = !build_cache->valid;
        }

        ForwardInput build_input = effective_input;
        if (forward_cache_eligible && forward_signature.is_bucketed_prefill && build_input.bucket_seq_len <= 0)
        {
            // Mark exact raw bucket requests as bucket-shaped so LM-head
            // row-select is present if the cached bucket is later replayed with
            // a shorter real length.
            build_input.real_seq_len = effectiveRealSeqLen(effective_input);
            build_input.bucket_seq_len = forward_signature.bucket_seq_len;
        }

        const bool success = executeCacheMiss(build_input, output, forward_signature, build_cache,
                                              should_cache_after_build, host, is_decode,
                                              start);
        if (!success && forward_cache_eligible)
        {
            /*
             * First-use materialization is transactional. Leaving an invalid
             * entry behind would preserve graph-owned streams, events, and
             * pointers from a failed capture and make the next invocation look
             * like a reusable cache object. Erase the entire half-built entry;
             * a later request must construct a fresh ownership lifetime.
             */
            cache_.erase(forward_signature);
        }
        if (live_mtp_request_batch_condition &&
            debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
        {
            LOG_INFO(
                "[MTPConditionGraphContract] event=execution_complete"
                << " device=" << effective_input.device.toString()
                << " cache=miss"
                << " success=" << boolTag(success));
        }
        if (success && build_cache && build_cache->valid)
        {
            if (materialized_signature)
                *materialized_signature = forward_signature;
        }
        if (success && build_cache && build_cache->valid && !setup_materialization)
            recordLastExecutedForwardGraph(
                forward_signature,
                /*cache_hit=*/false,
                output.execution.stream);
        if (setup_materialization)
        {
            if (success && !forward_signature.decode)
                enforcePrefillForwardCapacity(host, &forward_signature);
            return success;
        }
        if (success && !output.execution.valid)
        {
            LOG_ERROR("[ForwardExecutionEngine] Successful forward did not publish execution provenance for "
                      << effective_input.device.toString());
            return false;
        }
        if (success)
            host.commitSuccessfulForwardOutput(output);
        return success;
    }

    void ForwardExecutionEngine::recordLastExecutedForwardGraph(
        const ForwardGraphSignature &signature,
        bool cache_hit,
        void *producer_stream)
    {
        last_executed_forward_graph_.valid = true;
        last_executed_forward_graph_.signature = signature;
        last_executed_forward_graph_.cache_hit = cache_hit;
        last_executed_forward_graph_.producer_stream = producer_stream;
        if (signature.execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier &&
            signature.decode &&
            signature.all_position_logits)
        {
            last_all_position_verifier_graph_.valid = true;
            last_all_position_verifier_graph_.signature = signature;
            last_all_position_verifier_graph_.cache_hit = cache_hit;
            last_all_position_verifier_graph_.producer_stream =
                producer_stream;
        }
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::lastExecutedForwardGraph()
    {
        return viewForLastExecutedForwardGraphState(last_executed_forward_graph_);
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::lastAllPositionVerifierForwardGraph()
    {
        return viewForLastExecutedForwardGraphState(last_all_position_verifier_graph_);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::lastExecutedDeviceLoopGraphTemplate(
        std::string *error) const
    {
        return deviceLoopGraphTemplateForState(
            last_executed_forward_graph_, error);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::lastAllPositionVerifierDeviceLoopGraphTemplate(
        std::string *error) const
    {
        return deviceLoopGraphTemplateForState(
            last_all_position_verifier_graph_, error);
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::deviceLoopGraphTemplate(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
        {
            return reject(
                "no cached forward graph matches the exact signature");
        }
        if (!cache_it->second.valid || !cache_it->second.graph)
        {
            return reject(
                "the exact forward graph cache entry is not valid");
        }

        const ForwardGraphCache &cache = cache_it->second;
        if (signature.captured_verifier_extent != cache.segment_cache.captured_verifier_extent)
            return reject("exact forward signature and native verifier extent disagree");
        if (!signature.device.is_gpu() || !signature.decode)
            return reject("device-loop composition requires a GPU decode graph");
        if (!signature.usesDeviceTokenIds() ||
            !signature.usesDevicePositionIds())
        {
            return reject(
                "device-loop composition requires device-owned token and position rows");
        }
        // Composition consumes a sealed executable, not evidence that it has
        // already executed. `phase3_active` only skips semantic-node resets
        // after ordinary replay; using it as readiness forced synthetic first
        // inference before an ordinary/PP generation parent could be built.
        if (cache.segment_cache.executable_submission_state ==
            DeviceGraphExecutor::GraphSegmentCache::ExecutableSubmissionState::Empty)
            return reject("forward graph has no materialized executable");

        std::string template_error;
        const auto template_view = cache.segment_cache.deviceLoopGraphTemplate(
            *cache.graph,
            &template_error);
        if (!template_view)
            return reject(template_error);

        DeviceLoopGraphTemplateView view;
        view.capture = template_view->capture;
        view.signature = signature;
        view.device = signature.device;
        view.stream = template_view->stream;
        view.stage_count = template_view->stage_count;
        view.captured_node_count = template_view->captured_node_count;
        view.storage_boundary=template_view->storage_boundary;
        view.read_ahead_resources=template_view->read_ahead_resources;
        return view;
    }

    bool ForwardExecutionEngine::observeComposedForwardCompletion(
        const ForwardGraphSignature &signature,
        const IGPUGraphCapture *source_capture,
        int64_t completed_invocations,
        std::string *error) const
    {
        const auto reject = [&](const char *reason) {
            if (error) *error = reason;
            return false;
        };
        if (error) error->clear();
        if (!source_capture || completed_invocations < 0 ||
            !signature.device.is_gpu() || !signature.decode)
            return reject("Composed forward completion requires a GPU decode source and nonnegative count");
        const auto found = cache_.find(signature);
        if (found == cache_.end() || !found->second.valid || !found->second.graph)
            return reject("Composed forward completion has no exact cached source");
        const auto &segment = found->second.segment_cache;
        // The compiler already proved complete stage coverage at export. Check
        // the retained owner in constant time here, not by walking the model or
        // querying native graph nodes again for every completed request.
        if (!segment.initialized || segment.needs_capture ||
            segment.executable_submission_state ==
                DeviceGraphExecutor::GraphSegmentCache::ExecutableSubmissionState::Empty ||
            segment.segments.size() != 1 ||
            segment.segments.front().capture.get() != source_capture ||
            !source_capture->hasExecutable())
            return reject("Composed forward completion lost its original materialized child");
        if (completed_invocations != 0)
            PerfStatsCollector::addCounter("forward_graph", "decode_graph_phase",
                static_cast<double>(completed_invocations), "decode", signature.device.toString(),
                {{"context", forwardGraphPerfContext(signature)}, {"phase", "replay"},
                 {"source", "captured_generation_parent"}, {"observation", "authenticated_terminal"}});
        return true;
    }

    bool ForwardExecutionEngine::publishComposedForwardSnapshots(
        const ForwardGraphSignature &signature,
        const IGPUGraphCapture *source_capture,
        void *producer_stream,
        std::string *error)
    {
        const auto reject = [&](const char *reason) {
            if (error) *error = reason;
            return false;
        };
        if (error) error->clear();
        if (reinterpret_cast<std::uintptr_t>(producer_stream) <= 2)
            return reject("Composed forward diagnostics require the exact non-default parent stream");
        if (!observeComposedForwardCompletion(signature, source_capture, 0, error))
            return false;
        auto &original = cache_.at(signature);
        if (original.snapshot_configuration_epoch != executor_.snapshotConfigurationEpoch())
            return reject("Composed forward diagnostics have a stale original snapshot configuration");
        // The captured child already wrote this manifest's D2D slots. Use the
        // ordinary bulk diagnostic acquisition and callbacks on the actual
        // parent stream, never the stages' retained capture-time stream guesses.
        if (!executor_.publishSnapshotsAfterGraphExecution(*original.graph, producer_stream,
                "completed_composed_forward", &original.segment_cache.snapshot_manifest))
            return reject("Composed forward could not publish its original captured snapshot manifest");
        recordLastExecutedForwardGraph(signature, true, producer_stream);
        return true;
    }

    std::vector<RetainedForwardSourceDefinition>
    ForwardExecutionEngine::retainedSourceGraphDefinitions(uint64_t snapshot_identity) const
    {
        if (!snapshot_identity)
            throw std::invalid_argument("Finite source family requires its original snapshot topology");
        std::vector<RetainedForwardSourceDefinition> result;
        for (const auto& [signature, cache] : cache_)
        {
            if (signature.snapshot_configuration_identity != snapshot_identity ||
                !cache.valid || !cache.graph || !cache.segment_cache.initialized)
                continue;
            const auto& native = cache.segment_cache;
            if (native.needs_capture || !native.successful_capture_count ||
                !native.capture_variant_signature)
                throw std::logic_error("Finite source family contains an incomplete captured graph");
            const auto& declaration = cache.graph->retainedParticipantSourcePlan();
            const auto& preparation = native.participant_preparation_stream;
            if (static_cast<bool>(declaration) != static_cast<bool>(preparation))
                throw std::logic_error("Finite source family lost its exact optional source preparation");
            if (preparation)
            {
                preparation->requireCurrentDefinition();
                const auto& definition = preparation->definition();
                if (definition.graph != cache.graph.get() || definition.plan != declaration ||
                    definition.topology_generation != cache.graph->topologyGeneration() ||
                    definition.variant != native.capture_variant_signature)
                    throw std::logic_error("Finite source family substituted its original graph definition");
            }
            const auto role = signature.execution_role == ForwardExecutionRole::GroupedMTPVerifier
                ? MoEOverlayInferenceGraphRole::MTPGroupedVerifier
                : signature.decode ? MoEOverlayInferenceGraphRole::MainDecode
                                   : MoEOverlayInferenceGraphRole::MainPrefill;
            const auto rows = static_cast<int64_t>(signature.batch_size) * signature.seq_len;
            if (rows <= 0 || rows > std::numeric_limits<int>::max())
                throw std::logic_error("Finite source family has no original physical row geometry");
            auto owned=RetainedForwardSourceDefinition::projectOwned(signature,role,static_cast<int>(rows),
                *cache.graph,native.capture_variant_signature,preparation,native.recorded_source_definition,
                native.participant_auxiliary_preparation,native.original_storage_attachment.get());
            if(owned)result.push_back(std::move(*owned));
        }
        return result;
    }

    std::optional<ForwardExecutionEngine::RetainedDecodeGraphView>
    ForwardExecutionEngine::retainedDecodeGraph(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<RetainedDecodeGraphView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
            return reject("no cached forward graph matches the exact signature");

        const ForwardGraphCache &cache = cache_it->second;
        if (!cache.valid || !cache.graph)
            return reject("the exact forward graph cache entry is not valid");
        if (signature.captured_verifier_extent != cache.segment_cache.captured_verifier_extent)
            return reject("exact forward signature and native verifier extent disagree");
        if (!signature.device.is_gpu() || !signature.decode)
            return reject("retained replay requires a GPU decode graph");
        if (!signature.usesDeviceTokenIds() ||
            !signature.usesDevicePositionIds())
        {
            return reject(
                "retained replay requires device-owned token and position rows");
        }
        if (cache.segment_cache.executable_submission_state ==
                DeviceGraphExecutor::GraphSegmentCache::ExecutableSubmissionState::Empty ||
            !cache.segment_cache.initialized ||
            cache.segment_cache.needs_capture)
        {
            return reject(
                "forward graph has not materialized its complete retained executable");
        }
        if (!cache.segment_cache.capture_stream || !cache.gpu_ctx ||
            cache.segment_cache.segments.empty())
        {
            return reject(
                "retained forward replay has incomplete stream, context, or segment ownership");
        }

        RetainedDecodeGraphView view;
        view.signature = signature;
        view.device = signature.device;
        view.stream = cache.segment_cache.capture_stream;
        view.graph_owner = cache.graph.get();
        view.topology_generation = cache.graph->topologyGeneration();
        view.workspace_generation = cache.workspace_generation;
        view.capture_variant_signature = cache.segment_cache.capture_variant_signature;
        view.capture_generation = cache.segment_cache.successful_capture_count;
        view.read_ahead_resources=cache.graph->readAheadBranch()?cache.graph->readAheadBranch()->resources:nullptr;
        const auto &parent_plan =
            cache.segment_cache.retained_composed_parent_replay;
        if (parent_plan.valid())
        {
            if (!DeviceGraphExecutor::isRetainedParentPlanPolicy(
                    cache.segment_cache.graph_replay_plan_policy) ||
                !cache.segment_cache.retained_parent_capture ||
                !cache.segment_cache.retained_parent_capture->hasExecutable())
            {
                return reject(
                    "retained parent identity is incomplete or disagrees with replay policy");
            }
            view.replay_unit_count = 1u;
            view.source_capture_unit_count = parent_plan.child_unit_count;
            view.composed_parent = true;
            view.segmented = false;
        }
        else
        {
            view.replay_unit_count = cache.segment_cache.segments.size();
            view.source_capture_unit_count = view.replay_unit_count;
            view.segmented = view.replay_unit_count > 1;
        }
        return view;
    }

    std::optional<std::vector<
        DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView>>
    ForwardExecutionEngine::retainedDecodeCaptureUnitTemplates(
        const ForwardGraphSignature &signature,
        std::string *error) const
    {
        using UnitView = DeviceGraphExecutor::GraphSegmentCache::
            RetainedCaptureUnitTemplateView;
        auto reject = [&](const std::string &reason)
            -> std::optional<std::vector<UnitView>>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();

        std::string retained_error;
        if (!retainedDecodeGraph(signature, &retained_error))
            return reject(retained_error);

        const auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end() || !cache_it->second.graph)
        {
            return reject(
                "retained decode capture-unit cache identity disappeared");
        }
        return cache_it->second.segment_cache.retainedCaptureUnitTemplates(
            *cache_it->second.graph, error);
    }

    bool ForwardExecutionEngine::replayRetainedDecodeGraph(
        const ForwardGraphSignature &signature,
        IDeviceContext *ctx,
        const IComputeStage::MoEOverlayCollectiveRuntimeParams &sparse_params,
        const DeviceGraphExecutor::GraphLaunchDependencyHook
            &launch_dependency,
        IForwardExecutionHost &host,
        void *transaction_stream,
        void **out_producer_stream,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            return false;
        };
        if (error)
            error->clear();
        if (out_producer_stream)
            *out_producer_stream = nullptr;

        std::string retained_error;
        const auto retained = retainedDecodeGraph(signature, &retained_error);
        if (!retained)
            return fail(retained_error);
        if (!ctx || ctx->deviceId() != signature.device)
            return fail("retained replay received the wrong device context");
        if (!transaction_stream)
        {
            return fail(
                "retained replay requires a non-null external transaction stream");
        }

        auto cache_it = cache_.find(signature);
        if (cache_it == cache_.end())
            return fail("retained replay cache identity disappeared");
        ForwardGraphCache &cache = cache_it->second;
        if (!cache.gpu_ctx ||
            !cache.segment_cache.orderCaptureStreamAfter(
                cache.gpu_ctx, transaction_stream))
        {
            return fail(
                "retained replay could not consume its external transaction stream");
        }

        if (!cache.moe_overlay_collective_runtime_stages.empty() &&
            !sparse_params.valid())
        {
            return fail(
                "retained sparse replay requires a root-authoritative wire identity");
        }
        for (IComputeStage *stage :
             cache.moe_overlay_collective_runtime_stages)
        {
            if (!stage)
                return fail("retained sparse replay contains a null runtime stage");
            stage->updateMoEOverlayCollectiveRuntimeParams(sparse_params);
        }

        const auto plan_policy =
            cache.segment_cache.graph_replay_plan_policy;
        DeviceGraphExecutor::RetainedParentCompositionHook parent_composer;
        if (retained->composed_parent)
        {
            if (!DeviceGraphExecutor::isRetainedParentPlanPolicy(plan_policy))
            {
                return fail(
                    "retained parent view disagrees with cache replay policy");
            }
            try
            {
                const uint64_t current_topo_gen = cache.graph->topologyGeneration();
                if (!cache.cached_retained_parent_valid ||
                    cache.cached_retained_parent_graph != cache.graph.get() ||
                    cache.cached_retained_parent_topology_generation != current_topo_gen)
                {
                    auto parent_plan =
                        makeMoEOverlayRetainedParentPlan(*cache.graph);
                    cache.cached_retained_parent_graph = cache.graph.get();
                    cache.cached_retained_parent_topology_generation = current_topo_gen;
                    cache.cached_retained_parent_valid = true;
                    cache.cached_retained_parent_has_plan = parent_plan.has_value();
                    if (parent_plan)
                    {
                        cache.cached_retained_parent_policy = parent_plan->replay_policy;
                        cache.cached_retained_parent_composer = std::move(parent_plan->composer);
                    }
                    else
                    {
                        cache.cached_retained_parent_policy =
                            DeviceGraphExecutor::GraphReplayPlanPolicy{};
                        cache.cached_retained_parent_composer = {};
                    }
                }
                const bool pipeline_budget = config_.pipeline_forward_edges &&
                    config_.pipeline_forward_edges->requiresMaintenanceAdmission(signature.execution_role);
                if ((!cache.cached_retained_parent_has_plan && !pipeline_budget) ||
                    (cache.cached_retained_parent_has_plan &&
                     cache.cached_retained_parent_policy != plan_policy &&
                     !pipeline_budget))
                    return fail("retained parent graph no longer declares its sealed packet/budget lifecycle");
                if (cache.cached_retained_parent_has_plan)
                    parent_composer = cache.cached_retained_parent_composer;
                if (pipeline_budget)
                {
                    DeviceGraphExecutor::DecodeCapturePolicy composition;
                    composition.allow_cached_graph_replay = true;
                    composition.retained_parent_composer = std::move(parent_composer);
                    (void)composePipelineMaintenanceAdmission(composition, config_.pipeline_forward_edges,
                        signature.execution_role, cache.gpu_ctx, cache.segment_cache.capture_stream);
                    parent_composer = std::move(composition.retained_parent_composer);
                }
            }
            catch (const std::exception &ex)
            {
                return fail(
                    std::string(
                        "retained parent topology validation failed: ") +
                    ex.what());
            }
        }

        /*
         * Phase-3 graph units do not call markCompleted(), while manual sparse
         * boundaries retain ordinary ComputeGraph stage flags. When the sealed
         * retained-parent replay executable is already active for this exact
         * graph topology, skip the full-graph node reset walk; the executor
         * resets the graph before any fallback/recapture path.
         */
        // Retained MTP submission uses the same runner-owned authority as
        // setup and ordinary forward. Reading the cached factory instead would
        // hide a changed owner; accepting a caller-supplied empty descriptor
        // previously dropped the maintenance branch only on hosted verification.
        const auto auxiliary_branch_factory =
            host.forwardGraphAuxiliaryBranchFactory(ctx->deviceId());
        const bool sealed_retained_parent_fast_path =
            cache.segment_cache.retained_composed_parent_replay.valid() &&
            cache.segment_cache.retained_composed_parent_replay.graph == cache.graph.get() &&
            cache.segment_cache.retained_composed_parent_replay.topology_generation ==
                cache.graph->topologyGeneration() &&
            cache.segment_cache.initialized &&
            !cache.segment_cache.needs_capture &&
            cache.segment_cache.retained_parent_capture &&
            cache.segment_cache.retained_parent_capture->hasExecutable();
        if (!sealed_retained_parent_fast_path)
            cache.graph->reset();
        if (!executor_.executeWithCachedGraphReplay(
                *cache.graph,
                ctx,
                cache.segment_cache,
                retained->stream,
                cache.gpu_ctx,
                &cache.collective_nodes,
                /*collectives_graph_capturable=*/
                    !cache.collective_nodes.empty(),
                /*force_recapture=*/false,
                /*defer_final_sync=*/true,
                {},
                plan_policy,
                launch_dependency,
                std::span<const BufferId>{},
                std::move(parent_composer),
                DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                    CaptureInstantiateAndLaunch,
                auxiliary_branch_factory,
                signature.captured_verifier_extent))
        {
            return fail("retained decode graph replay submission failed");
        }
        if (!cache.segment_cache.orderStreamAfterCapture(
                cache.gpu_ctx, transaction_stream))
        {
            return fail(
                "retained replay could not publish its exact terminal to the external transaction stream");
        }

        if (out_producer_stream)
            *out_producer_stream = retained->stream;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "hosted_retained_decode_replays",
            1.0,
            "decode",
            signature.device.toString(),
            {{"replay_units", std::to_string(retained->replay_unit_count)},
             {"source_capture_units",
              std::to_string(retained->source_capture_unit_count)},
             {"segmented", retained->segmented ? "true" : "false"},
             {"composed_parent",
              retained->composed_parent ? "true" : "false"},
             {"role", std::to_string(static_cast<int>(
                          signature.execution_role))},
             {"physical_rows", std::to_string(signature.seq_len)},
             {"sparse_query_program", std::to_string(sparseQueryExecutionProgramIndex(signature.sparse_query_execution_program))},
             {"capture_family", std::to_string(cache.segment_cache.evidence_family)}});
        return true;
    }

    void ForwardExecutionEngine::clearLastAllPositionVerifierForwardGraph()
    {
        last_all_position_verifier_graph_ = {};
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::viewForLastExecutedForwardGraphState(
        const LastExecutedForwardGraphState &state)
    {
        if (!state.valid)
            return std::nullopt;

        auto it = cache_.find(state.signature);
        if (it == cache_.end() || !it->second.valid || !it->second.graph)
            return std::nullopt;

        LastExecutedForwardGraphView view;
        view.graph = it->second.graph.get();
        view.signature = state.signature;
        view.device = view.signature.device;
        view.stream = state.producer_stream;
        view.cache_hit = state.cache_hit;
        view.is_decode = view.signature.decode;
        view.all_position_logits = view.signature.all_position_logits;
        return view;
    }

    std::optional<ForwardExecutionEngine::DeviceLoopGraphTemplateView>
    ForwardExecutionEngine::deviceLoopGraphTemplateForState(
        const LastExecutedForwardGraphState &state,
        std::string *error) const
    {
        auto reject = [&](const std::string &reason)
            -> std::optional<DeviceLoopGraphTemplateView>
        {
            if (error)
                *error = reason;
            return std::nullopt;
        };
        if (error)
            error->clear();
        if (!state.valid)
            return reject("no successful forward graph is retained");

        auto view = deviceLoopGraphTemplate(state.signature, error);
        if (!view && error &&
            *error == "no cached forward graph matches the exact signature")
        {
            *error = "retained forward graph no longer owns a valid cache entry";
        }
        return view;
    }

    std::vector<ForwardExecutionEngine::ReplayCacheObservation>
    ForwardExecutionEngine::replayCacheObservations(uint64_t live_state_epoch) const
    {
        std::vector<ReplayCacheObservation> observations;
        observations.reserve(cache_.size());
        for (const auto &[signature, cache] : cache_)
        {
            ReplayCacheObservation observation;
            observation.signature = signature;
            observation.valid = cache.valid;
            observation.segment_initialized = cache.segment_cache.initialized;
            observation.segment_needs_capture = cache.segment_cache.needs_capture;
            observation.phase3_active = cache.phase3_active;
            observation.gpu_stream_bindings_applied = cache.gpu_stream_applied;
            observation.has_capture_stream =
                cache.segment_cache.capture_stream != nullptr;
            observation.segment_decode_step = cache.segment_cache.decode_step;
            observation.graph_replay_live_state_epoch =
                cache.graph_replay_live_state_epoch;
            observation.requires_live_state_epoch_recapture =
                cache.requiresLiveStateEpochRecapture(
                    isLiveStateVersionedReplayCache(signature),
                    /*graph_replay_allowed=*/true,
                    live_state_epoch);
            observation.all_position_verifier_recapture_pending =
                all_position_verifier_recapture_pending_;
            observations.push_back(observation);
        }
        return observations;
    }

    std::vector<ForwardExecutionEngine::PrefillGraphCacheSnapshot>
    ForwardExecutionEngine::prefillGraphCacheSnapshots() const
    {
        std::vector<PrefillGraphCacheSnapshot> snapshots;
        snapshots.reserve(cache_.size());
        for (const auto &[signature, cache] : cache_)
        {
            if (signature.decode)
                continue;

            PrefillGraphCacheSnapshot snapshot;
            snapshot.forward_cache_valid = cache.valid;
            snapshot.eviction_count = prefill_forward_eviction_count_;
            snapshot.bucket_seq_len = signature.is_bucketed_prefill
                                          ? signature.bucket_seq_len
                                          : signature.seq_len;
            snapshot.placement_epoch = signature.moe_placement_epoch;
            snapshot.capture_phase = "unknown";
            snapshot.recapture_reason = "none";

            PrefillGraphCacheKey key;
            key.seq_len = snapshot.bucket_seq_len;
            key.device_id = signature.device;
            key.placement_epoch = signature.moe_placement_epoch;

            if (cache.last_prefill_graph_observation.valid)
            {
                const auto &observation = cache.last_prefill_graph_observation;
                snapshot.observation_valid = true;
                snapshot.chunk_index = observation.chunk_index;
                snapshot.bucket_seq_len = observation.bucket_seq_len;
                snapshot.real_token_start = observation.real_token_start;
                snapshot.real_token_count = observation.real_token_count;
                snapshot.real_token_end = observation.real_token_end;
                snapshot.domain_id = observation.domain_id;
                snapshot.participant_id = observation.participant_id;
                snapshot.placement_epoch = observation.placement_epoch;
                snapshot.topology_signature = observation.topology_signature;
                snapshot.capture_phase = observation.capture_phase;
                snapshot.recapture_reason = observation.recapture_reason;
                snapshot.reject_stage_name = observation.reject_stage_name;
                snapshot.reject_stage_type = observation.reject_stage_type;

                key.seq_len = observation.bucket_seq_len;
                key.domain_id = observation.domain_id;
                key.participant_id = observation.participant_id;
                key.placement_epoch = observation.placement_epoch;
                key.topology_signature = observation.topology_signature;
            }

            if (cache.segment_cache.initialized ||
                cache.segment_cache.successful_capture_count != 0u)
            {
                applySegmentedPrefillGraphSnapshot(
                    cache.segment_cache, &snapshot);
            }
            else if (cache.prefill_graph_cache)
            {
                const PrefillGraphCache &prefill_cache = *cache.prefill_graph_cache;
                snapshot.prefill_cache_initialized = true;
                snapshot.phase = prefill_cache.phase(key);
                snapshot.cache_size = prefill_cache.size();
                snapshot.node_count = prefill_cache.nodeCount(key);
                snapshot.replay_count = prefill_cache.replayCount(key);
                snapshot.warmup_count = prefill_cache.warmupCount(key);
                snapshot.initialized_count = prefill_cache.initializedCount(key);
                snapshot.capture_count = prefill_cache.captureCount(key);
                snapshot.eviction_count += prefill_cache.evictionCount();
            }

            snapshots.push_back(std::move(snapshot));
        }
        return snapshots;
    }

    bool ForwardExecutionEngine::materializeCachedExecutableWithoutLaunch(
        const ForwardInput &input,
        ForwardGraphCache &forward_cache,
        IForwardExecutionHost &host,
        const ForwardGraphSignature &signature)
    {
        const bool is_decode = signature.decode;
        if (input.graph_submission_intent !=
                ForwardGraphSubmissionIntent::
                    MaterializeExecutableWithoutLaunch ||
            !input.device.is_gpu() || !forward_cache.graph ||
            input.moe_overlay_graph_launch_dependency)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization requires "
                "one cached GPU graph and forbids request-scoped launch "
                "dependencies");
            return false;
        }
        if (is_decode && forward_cache.segment_cache.initialized &&
            !forward_cache.segment_cache.needs_capture &&
            forward_cache.segment_cache.executable_submission_state !=
                DeviceGraphExecutor::GraphSegmentCache::
                    ExecutableSubmissionState::Empty)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "setup_materialized_graph_reuses",
                1.0,
                "setup",
                input.device.toString(),
                {{"context", forwardGraphPerfContext(signature)}});
            return true;
        }

        IDeviceContext *const ctx = host.getDeviceContext(input.device);
        IWorkerGPUContext *const gpu_ctx =
            ctx ? host.getWorkerGPUContext(ctx->deviceId()) : nullptr;
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization could "
                "not establish its exact device context on "
                << input.device.toString());
            return false;
        }
        forward_cache.gpu_ctx = gpu_ctx;
        forward_cache.gpu_stream = gpu_ctx->defaultStream();
        if (!forward_cache.gpu_stream)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup graph materialization resolved "
                "a null worker stream on "
                << input.device.toString());
            return false;
        }

        constexpr auto kMaterialize =
            DeviceGraphExecutor::GraphInitialSubmissionPolicy::
                MaterializeWithoutLaunch;
        if (!is_decode)
        {
            bool used_graph_replay = false;
            return executePrefillWithGraphCache(
                input,
                forward_cache,
                ctx,
                host,
                &used_graph_replay,
                nullptr,
                kMaterialize);
        }

        if (!forward_cache.segment_cache.ensureCaptureStream(
                gpu_ctx, ctx->deviceId()) ||
            !forward_cache.segment_cache.ensureCaptureInputEvent(gpu_ctx) ||
            !forward_cache.segment_cache.ensureCaptureOutputEvent(gpu_ctx) ||
            !forward_cache.segment_cache.capture_stream)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode graph materialization "
                "could not establish its exact capture stream on "
                << input.device.toString());
            return false;
        }

        /*
         * Setup capture has no admitted request and launches no model work.
         * Graph construction may nevertheless publish immutable device
         * descriptors. Join that producer on the exact decode capture stream so
         * the same event-owned state is visible to every graph in the family.
         */
        if (!host.prepareGraphBuildStateForMaterialization(
                input,
                forward_cache.segment_cache.capture_stream,
                ctx->deviceId()))
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode graph materialization "
                "could not consume graph-build device-state readiness on "
                << input.device.toString());
            return false;
        }

        DeviceGraphExecutor::DecodeCapturePolicy capture_policy =
            host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(), ctx);
        capture_policy.captured_verifier_extent = input.captured_verifier_extent;
        if (!capture_policy.allow_cached_graph_replay)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode materialization requires "
                "the production cached-graph policy on "
                << input.device.toString());
            return false;
        }
        try
        {
            capture_policy.auxiliary_branch_factory =
                sourceDefinitionBoundaryFactory(host,input,ctx->deviceId(),true,forward_cache);
            const GraphNativeCaptureEnvelope native_capture_envelope =
                forward_cache.graph->nativeCaptureEnvelope();
            std::string envelope_error;
            if (!DeviceGraphCaptureController::
                    constrainReplayPolicyToNativeEnvelope(
                        native_capture_envelope,
                        capture_policy,
                        &envelope_error))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Setup graph materialization "
                    "rejected its native capture envelope: "
                    << envelope_error);
                return false;
            }
            if (auto sparse_parent_plan =
                    makeMoEOverlayRetainedParentPlan(
                        *forward_cache.graph);
                sparse_parent_plan)
            {
                capture_policy.graph_replay_plan_policy =
                    sparse_parent_plan->replay_policy;
                capture_policy.retained_parent_composer =
                    std::move(sparse_parent_plan->composer);
                capture_policy.defer_final_sync = true;
            }
            (void)composePipelineMaintenanceAdmission(capture_policy, config_.pipeline_forward_edges,
                input.execution_role, gpu_ctx, forward_cache.segment_cache.capture_stream);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[ForwardExecutionEngine] Setup decode materialization could "
                "not derive its retained ExpertOverlay topology: "
                << error.what());
            return false;
        }

        const DeviceId capture_device = ctx->deviceId();
        capture_policy.capture_boundary =
            [&host, &input, capture_device](
                const std::string &boundary_name,
                void *capture_stream)
        {
            return host.waitAtDecodeGraphCaptureBoundary(
                input,
                capture_device,
                boundary_name,
                capture_stream);
        };
        capture_policy.launch_dependency = {};
        capture_policy.force_recapture = false;
        // Setup and request replay must name the same graph owner. Treating
        // every decode-shaped graph as main_decode hides verifier captures.
        forward_cache.segment_cache.perf_context = forwardGraphPerfContext(signature);

        bool used_graph_replay = false;
        return executor_.executeDecodeWithCapturePolicy(
            *forward_cache.graph,
            ctx,
            &forward_cache.segment_cache,
            forward_cache.gpu_stream,
            forward_cache.gpu_ctx,
            &forward_cache.collective_nodes,
            capture_policy,
            &used_graph_replay,
            kMaterialize);
    }

    // =========================================================================
    // executeCacheHit() — Reuse Cached Forward Graph
    // =========================================================================

    bool ForwardExecutionEngine::executeCacheHit(
        const ForwardInput &input,
        ForwardOutput &output,
        ForwardGraphCache &forward_cache,
        IForwardExecutionHost &host,
        const ForwardGraphSignature &signature,
        std::chrono::high_resolution_clock::time_point start,
        HostPipelineBoundarySubmission pp_submission)
    {
        const bool is_decode = signature.decode;
        // ===== CACHE HIT: Reuse cached decode graph =====

        // Setup sub-phase timing (only when profiling is active)
        /*
         * The coarse forward-pass decomposition is intentionally available
         * without per-kernel or per-stage GPU events.  It adds only host clock
         * reads and thread-local accumulation, which lets a focused PerfStats
         * campaign attribute orchestration latency without changing the GPU
         * execution DAG it is trying to measure.
         */
        const bool profiling_setup =
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass");
        using Clock = std::chrono::high_resolution_clock;
        Clock::time_point setup_workspace_t0, setup_workspace_t1;
        Clock::time_point setup_token_copy_t0, setup_token_copy_t1;
        Clock::time_point setup_stream_t0, setup_stream_t1;
        Clock::time_point setup_dynamic_params_t0, setup_dynamic_params_t1;
        Clock::time_point setup_graph_reset_t0, setup_graph_reset_t1;

        if (profiling_setup)
            setup_workspace_t0 = Clock::now();

        if (input.device.is_gpu() && forward_cache.graph)
        {
            // Fast-path: if workspace generation hasn't changed since our last
            // validation, skip the expensive O(N_stages) graph traversal inside
            // ensureDeviceWorkspaceAllocated. During steady-state decode (seq_len=1),
            // nothing triggers workspace reallocation, so this saves ~3.5ms/iter
            // by avoiding 1595 dynamic_casts + string comparisons every step.
            const uint64_t current_workspace_generation = host.workspaceGeneration(input.device);
            const bool workspace_validated = (forward_cache.workspace_generation != 0 &&
                                              current_workspace_generation == forward_cache.workspace_generation);

            if (!workspace_validated)
            {
                const WorkspaceGraphParticipantRole workspace_role =
                    workspaceParticipantRole(
                        input.execution_role,
                        is_decode);
                if (!host.ensureDeviceWorkspaceAllocated(
                        *forward_cache.graph,
                        input.seq_len,
                        workspaceFamilyPolicy(workspace_role),
                        workspace_role))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to refresh cached graph workspace for "
                              << input.device.toString() << " seq_len=" << input.seq_len);
                    return false;
                }

                const uint64_t new_generation = host.workspaceGeneration(input.device);
                if (new_generation != forward_cache.workspace_generation)
                {
                    if (forward_cache.workspace_generation != 0)
                    {
                        LOG_DEBUG("[ForwardExecutionEngine] Workspace generation changed on "
                                  << input.device.toString()
                                  << " from " << forward_cache.workspace_generation
                                  << " to " << new_generation
                                  << "; dropping captured replay state for cached graph");
                        forward_cache.resetReplayStateAfterWorkspaceRebind();
                    }
                    forward_cache.workspace_generation = new_generation;
                }
            }
        }

        if (profiling_setup)
        {
            setup_workspace_t1 = Clock::now();
            setup_token_copy_t0 = setup_workspace_t1;
        }

        // Update stable buffers — stages hold pointers to these, so the
        // pointed-to values change but the pointers remain valid
        const int total_tokens = input.batch_size * input.seq_len;
        if (input.token_ids)
        {
            if (static_cast<int>(forward_cache.token_ids.size()) == total_tokens)
            {
                std::memcpy(forward_cache.token_ids.data(), input.token_ids,
                            static_cast<size_t>(total_tokens) * sizeof(int));
            }
            else
            {
                forward_cache.token_ids.assign(input.token_ids,
                                               input.token_ids + total_tokens);
            }
        }
        if (input.position_ids)
        {
            if (static_cast<int>(forward_cache.position_ids.size()) == total_tokens)
            {
                std::memcpy(forward_cache.position_ids.data(), input.position_ids,
                            static_cast<size_t>(total_tokens) * sizeof(int));
            }
            else
            {
                forward_cache.position_ids.assign(input.position_ids,
                                                  input.position_ids + total_tokens);
            }
        }

        // The explicit host boundary owns this invocation's live rows. Resolve
        // again on every replay; graph-cache capacity is never a transfer extent.
        const auto current_pp_copy = host.resolvePPCopyInfo(input);
        consumeHostPipelineActivationBoundary(
            input.graph_submission_intent == ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch
                ? HostPipelineBoundaryExecution::MaterializeOnly : HostPipelineBoundaryExecution::Execute,
            pp_submission,
            [&] { return current_pp_copy; },
            [&](const auto &pp_copy) {
                auto result = TransferEngine::instance().copyActivation(
                    pp_copy.external_hidden, pp_copy.working_buffer,
                    pp_copy.device, pp_copy.copy_bytes);
                if (!result.success)
                    throw std::runtime_error("PP hidden-state copy failed: " + result.error);
            });

        if (profiling_setup)
        {
            setup_token_copy_t1 = Clock::now();
            setup_stream_t0 = setup_token_copy_t1;
        }

        // For GPU graph replay: set an explicit capture stream on all stages
        // before dynamic params are updated. The stream is owned by the segment
        // cache and survives ordinary replay-state resets, so stages never need
        // to fall back to the device context's default stream between a failed
        // replay/capture and the next warmup attempt.
        auto apply_cached_graph_stream = [&](void *stream)
        {
            if (!stream)
                return;
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(stream);
            }
            forward_cache.gpu_stream_applied = true;
            forward_cache.applied_stream = stream;
        };

        void *replay_stream = forward_cache.segment_cache.capture_stream;
        void *dynamic_param_stream = nullptr;
        IDeviceContext *stream_ctx = nullptr;
        IWorkerGPUContext *stream_gpu_ctx = nullptr;
        DeviceId stream_device = input.device;
        bool graph_gpu_stage_found = false;
        const auto &stream_order = forward_cache.graph->getExecutionOrder();
        for (const auto &node_name : stream_order)
        {
            ComputeNode *node = forward_cache.graph->getNode(node_name);
            if (!node || !node->stage)
                continue;
            const DeviceId stage_device = node->stage->device();
            if (stage_device.is_gpu())
            {
                stream_device = stage_device;
                graph_gpu_stage_found = true;
                break;
            }
            if (node->device.is_gpu())
            {
                stream_device = node->device;
                graph_gpu_stage_found = true;
                break;
            }
        }

        /*
         * GraphCacheConfig::decode_seq_len is a heuristic boundary for ordinary
         * continuation traffic. It is not an MTP capacity declaration. Typed MTP
         * roles have already passed their configured verifier-row validation and
         * are decode-equivalent for every supported M, including M values above
         * the legacy four-row threshold. Applying that heuristic here silently
         * sent grouped verifier M=5..15 through eager execution.
         */
        const bool typed_mtp_decode_role =
            input.execution_role ==
                ForwardExecutionRole::GroupedMTPVerifier ||
            input.execution_role ==
                ForwardExecutionRole::MTPCondition;
        const bool decode_capture_allowed =
            is_decode &&
            input.batch_size <= 1 &&
            (typed_mtp_decode_role ||
             input.seq_len <=
                 std::max(1, config_.cache_config.decode_seq_len));

        if (stream_device.is_gpu())
        {
            stream_ctx = host.getDeviceContext(stream_device);
            if (stream_ctx && stream_ctx->deviceId().is_gpu())
            {
                try
                {
                    stream_gpu_ctx = host.getWorkerGPUContext(stream_ctx->deviceId());
                    if (!stream_gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for "
                                  << stream_ctx->deviceId().toString()
                                  << " before cached dynamic params");
                        return false;
                    }
                    forward_cache.gpu_ctx = stream_gpu_ctx;
                    if (!forward_cache.gpu_stream)
                        forward_cache.gpu_stream = stream_gpu_ctx->defaultStream();
                    dynamic_param_stream = forward_cache.gpu_stream;
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to resolve the mandatory worker GPU context for "
                              << stream_ctx->deviceId().toString()
                              << " before cached dynamic params: " << e.what());
                    return false;
                }
            }
        }

        bool prefill_uses_common_cache_stream = false;
        if ((decode_capture_allowed || !is_decode) &&
            stream_ctx && stream_ctx->deviceId().is_gpu() && stream_gpu_ctx)
        {
            auto early_capture_policy = host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                stream_ctx);
            const auto stream_capture_envelope = forward_cache.graph->nativeCaptureEnvelope();
            std::string envelope_error;
            if (!DeviceGraphCaptureController::constrainReplayPolicyToNativeEnvelope(
                    stream_capture_envelope, early_capture_policy, &envelope_error))
            {
                LOG_ERROR("[ForwardExecutionEngine] Cached graph stream prelude rejected its native capture envelope: "
                    << envelope_error);
                return false;
            }
            const bool native_prefill_uses_common_cache = !is_decode &&
                prefillUsesCommonGraphCache(stream_capture_envelope, early_capture_policy);
            const bool use_segment_cache_stream = decode_capture_allowed || native_prefill_uses_common_cache;
            if (early_capture_policy.allow_cached_graph_replay &&
                use_segment_cache_stream)
            {
                if (forward_cache.segment_cache.ensureCaptureStream(
                        stream_gpu_ctx,
                        stream_ctx->deviceId()) &&
                    forward_cache.segment_cache.ensureCaptureInputEvent(
                        stream_gpu_ctx) &&
                    forward_cache.segment_cache.ensureCaptureOutputEvent(
                        stream_gpu_ctx))
                {
                    replay_stream = forward_cache.segment_cache.capture_stream;
                    executor_.bindRetainedParticipantPreparationStream(*forward_cache.graph,forward_cache.segment_cache,stream_gpu_ctx);
                    dynamic_param_stream = replay_stream;
                    prefill_uses_common_cache_stream = native_prefill_uses_common_cache;
                }
                else
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to create the mandatory explicit decode graph stream for "
                              << stream_ctx->deviceId().toString());
                    return false;
                }
            }
        }

        if (!is_decode && stream_gpu_ctx && !prefill_uses_common_cache_stream)
        {
            if (forward_cache.prefill_capture_stream.ensure(stream_gpu_ctx))
            {
                dynamic_param_stream = forward_cache.prefill_capture_stream.stream;
            }
            else
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to create explicit prefill graph stream for "
                          << stream_ctx->deviceId().toString());
            }
        }

        // Dynamic-param stages can perform device-side uploads before replay
        // (for example ROCm attention's device-param row). Bind an explicit
        // stream before updateDynamicParams(); DeviceGraphExecutor's normal
        // per-stage stream binding happens later and is too late for this setup
        // phase. `defaultStream()` here is Llaminar's owned non-null stream, not
        // the HIP/CUDA legacy null stream.
        if (dynamic_param_stream && forward_cache.applied_stream != dynamic_param_stream)
        {
            apply_cached_graph_stream(dynamic_param_stream);
        }
        else if (stream_device.is_gpu())
        {
            LOG_DEBUG("[ForwardExecutionEngine] Cached graph dynamic-param stream prelude: input_device="
                      << input.device.toString()
                      << " stream_device=" << stream_device.toString()
                      << " graph_gpu_stage_found=" << graph_gpu_stage_found
                      << " stream_ctx=" << (stream_ctx != nullptr)
                      << " stream_gpu_ctx=" << (stream_gpu_ctx != nullptr)
                      << " dynamic_param_stream=" << dynamic_param_stream
                      << " applied_stream=" << forward_cache.applied_stream);
        }

        if (current_pp_copy.needs_copy && current_pp_copy.copy_bytes > 0 &&
            input.graph_submission_intent == ForwardGraphSubmissionIntent::Execute &&
            stream_device.is_gpu())
        {
            // Retained replay deliberately skips the full stage-input walk.
            // The PP ingress is nevertheless a newly published external value
            // on every call. Acquire this one frontier on the exact replay
            // stream; residency and cached pointer identity do not imply that
            // an asynchronous transfer has finished writing the bytes.
            TransferEngine::requireDeviceInput(
                current_pp_copy.working_buffer, stream_device,
                dynamic_param_stream);
        }

        // Discovery is topology-owned and shared by setup capture and every
        // request execution. Only the small resulting list is walked per launch.
        if (!forward_cache.dynamic_param_stages_cached)
        {
            forward_cache.dynamic_param_stages =
                collectDynamicParamStages(*forward_cache.graph);
            forward_cache.dynamic_param_stages_cached = true;
        }

        if (input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::
                MaterializeExecutableWithoutLaunch)
        {
            /*
             * Setup capture intentionally skips mutable request admission, but
             * native capture still requires every stable input slot to be ready.
             * In particular, host-token decode must enqueue its sentinel H2D copy
             * before beginCapture(); the embedding kernel correctly rejects an
             * H2D operation discovered from inside capture.
             */
            const int *setup_position_ids =
                selectForwardReplayHostPositionIds(forward_cache, input);
            if (!updateDynamicParamStages(
                    input,
                    forward_cache.dynamic_param_stages,
                    setup_position_ids,
                    "setup graph materialization"))
            {
                return false;
            }
            return materializeCachedExecutableWithoutLaunch(
                input,
                forward_cache,
                host,
                signature);
        }

        if (profiling_setup)
        {
            setup_stream_t1 = Clock::now();
            setup_dynamic_params_t0 = setup_stream_t1;
        }

        if (!host.prepareLiveStateForForwardGraphExecution(
                input,
                dynamic_param_stream,
                stream_device))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for cached forward graph execution");
            return false;
        }
        ScopedForwardLiveStateRead live_state_read(
            host,
            input,
            dynamic_param_stream,
            stream_device);

        if (!host.prepareDeviceTokenInputsForForwardGraphExecution(
                input,
                dynamic_param_stream,
                stream_device))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to prepare cached forward graph device-token inputs");
            return false;
        }

        if (host.computeAllPositionLogitsEnabled() &&
            host.allPositionLogitRows() > 0)
        {
            if (!host.prepareAllPositionVerifierGraphMetadata(
                    input,
                    dynamic_param_stream,
                    stream_device))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare cached all-position verifier graph metadata");
                return false;
            }
        }

        if (!forward_cache.prefill_replay_param_stages_cached)
        {
            forward_cache.prefill_replay_param_stages.clear();
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->hasPrefillReplayParams())
                    forward_cache.prefill_replay_param_stages.push_back(node->stage.get());
            }
            forward_cache.prefill_replay_param_stages_cached = true;
        }
        if (!is_decode)
        {
            updatePrefillReplayParamStages(input, forward_cache.prefill_replay_param_stages);
        }
        if (!forward_cache.moe_overlay_collective_runtime_stages_cached)
        {
            forward_cache.moe_overlay_collective_runtime_stages =
                collectMoEOverlayCollectiveRuntimeStages(*forward_cache.graph);
            forward_cache.moe_overlay_collective_runtime_stages_cached = true;
        }
        updateMoEOverlayCollectiveRuntimeStages(
            input,
            forward_cache.moe_overlay_collective_runtime_stages);
        const int *replay_position_ids =
            selectForwardReplayHostPositionIds(forward_cache, input);
        if (!updateDynamicParamStages(
                input,
                forward_cache.dynamic_param_stages,
                replay_position_ids,
                "cached graph replay"))
        {
            return false;
        }

        if (profiling_setup)
        {
            setup_dynamic_params_t1 = Clock::now();
            setup_graph_reset_t0 = setup_dynamic_params_t1;
        }

        // Skip graph reset when Phase 3 replay is active — Phase 3 doesn't
        // call markCompleted(), so all flags are already false from last reset.
        if (!forward_cache.phase3_active)
        {
            forward_cache.graph->reset();
        }

        if (profiling_setup)
        {
            setup_graph_reset_t1 = Clock::now();
        }

        if (!host.waitBeforeForwardGraphExecution(
                input,
                input.device,
                /*cache_miss=*/false))
        {
            LOG_ERROR("[ForwardExecutionEngine] Forward graph pre-execution rendezvous failed for "
                      << input.device.toString());
            return false;
        }

        output = forward_cache.output;

        // Execute with single device context (standard path, no PP)
        IDeviceContext *ctx = host.getDeviceContext(input.device);
        if (!ctx)
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to get device context");
            return false;
        }

        bool success;
        const bool has_collective_nodes = !forward_cache.collective_nodes.empty();

        // Stable decode and verifier shapes use one complete captured graph on
        // homogeneous GPUs. A mixed-device collective domain may explicitly
        // admit collective-only segmentation through the topology policy.
        // Prefill uses its separate bucketed graph-cache state machine below.
        bool used_graph_replay = false;
        void *prefill_producer_stream = nullptr;
        bool requested_deferred_main_decode_sync = false;
        bool executed_deferred_main_decode_sync = false;

        auto exec_t0 = std::chrono::high_resolution_clock::now();
        /* The execution window between here and the cached-graph completion
         * is the fork/join domain the wake-up work item optimizes. The scope
         * is the window's owner cell on this executor thread: concurrent
         * ranks/devices keep their own cells, so the delta attributes exactly
         * without serializing production. GPU devices fork no OpenMP regions
         * and the summary publisher refuses them. */
        const llaminar::v2::OpenMPTeamForkScope openmp_forks_at_exec_entry;
        if (profiling_setup)
        {
            ForwardPassProfiler::resetReplayTimings();
        }

        if (!is_decode)
        {
            if (input.device.is_gpu())
            {
                // GPU prefill uses the native capture/replay state machine.
                // Snapshot capture is a post-graph drain so parity diagnostics
                // never force eager stage execution.
                success = executePrefillWithGraphCache(
                    input,
                    forward_cache,
                    ctx,
                    host,
                    &used_graph_replay,
                    &prefill_producer_stream);
            }
            else
            {
                /*
                 * CPU exact prefill reuses the complete graph topology without
                 * pretending that CPU execution is a GPU graph replay. The
                 * graph owns its prepared kernels and persistent stage state;
                 * reset/update above refreshes request data before each eager
                 * traversal. Keep snapshots bound to this exact graph just as
                 * on first materialization.
                 */
                success = executor_.executeWithSnapshotManifest(
                    *forward_cache.graph,
                    ctx,
                    forward_cache.snapshot_manifest);
            }
        }
        else
        {
            DeviceGraphExecutor::DecodeCapturePolicy capture_policy;
            if (decode_capture_allowed)
            {
                capture_policy = host.buildDecodeCapturePolicy(
                    has_collective_nodes,
                    ctx);
            }
            capture_policy.captured_verifier_extent = input.captured_verifier_extent;
            if (input.moe_overlay_graph_launch_dependency)
            {
                if (capture_policy.launch_dependency)
                {
                    const auto existing =
                        std::move(capture_policy.launch_dependency);
                    const auto overlay =
                        input.moe_overlay_graph_launch_dependency;
                    capture_policy.launch_dependency =
                        [existing, overlay](
                            DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
                            void *stream)
                    {
                        return existing(phase, stream) &&
                               overlay(phase, stream);
                    };
                }
                else
                {
                    capture_policy.launch_dependency =
                        input.moe_overlay_graph_launch_dependency;
                }
                if (!capture_policy.allow_cached_graph_replay)
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] GPU ExpertOverlay decode "
                        "requires retained graph execution so its remote "
                        "ticket can be armed at the exact executable launch "
                        "edge");
                    return false;
                }
            }
            const GraphNativeCaptureEnvelope native_capture_envelope =
                forward_cache.graph->nativeCaptureEnvelope();
            const bool uses_device_timeline_transaction =
                requiresSingleNativeExecutable(native_capture_envelope);
            const bool uses_heterogeneous_ticket_transaction =
                requiresHeterogeneousTicketSegmentation(
                    native_capture_envelope);
            bool uses_retained_sparse_parent = false;
            try
            {
                capture_policy.auxiliary_branch_factory =
                    sourceDefinitionBoundaryFactory(host,input,ctx->deviceId(),true,forward_cache);
                std::string envelope_error;
                if (!DeviceGraphCaptureController::
                        constrainReplayPolicyToNativeEnvelope(
                            native_capture_envelope,
                            capture_policy,
                            &envelope_error))
                {
                    LOG_ERROR(
                        "[ForwardExecutionEngine] Decode rejected its graph-native capture envelope: "
                        << envelope_error);
                    return false;
                }
                if (auto sparse_parent_plan =
                        makeMoEOverlayRetainedParentPlan(
                            *forward_cache.graph);
                    sparse_parent_plan)
                {
                    if (!decode_capture_allowed ||
                        !capture_policy.allow_cached_graph_replay)
                    {
                        LOG_ERROR(
                            "[ForwardExecutionEngine] Typed ExpertOverlay packet graph requires mandatory retained-parent capture/replay");
                        return false;
                    }
                    capture_policy.graph_replay_plan_policy =
                        sparse_parent_plan->replay_policy;
                    capture_policy.retained_parent_composer =
                        std::move(sparse_parent_plan->composer);
                    uses_retained_sparse_parent = true;
                }
                (void)composePipelineMaintenanceAdmission(capture_policy, config_.pipeline_forward_edges,
                    input.execution_role, forward_cache.gpu_ctx, forward_cache.segment_cache.capture_stream);
            }
            catch (const std::exception &ex)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Could not derive retained ExpertOverlay endpoint authority: "
                    << ex.what());
                return false;
            }
            if (typed_mtp_decode_role &&
                input.device.is_gpu() &&
                !capture_policy.allow_cached_graph_replay)
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] GPU MTP role requires cached "
                    "full-graph replay for every validated row count"
                    << " role="
                    << static_cast<int>(input.execution_role)
                    << " seq_len=" << input.seq_len
                    << " device=" << input.device.toString());
                return false;
            }
            if (capture_policy.heterogeneous_segmented_enabled)
            {
                LOG_DEBUG(
                    "[ForwardExecutionEngine] Heterogeneous collective-only "
                    "segmented GPU-graph replay admitted by topology policy");
            }

            const bool all_position_verifier =
                host.computeAllPositionLogitsEnabled();
            const bool pending_all_position_verifier_recapture =
                all_position_verifier &&
                input.seq_len > 1 &&
                all_position_verifier_recapture_pending_;
            if (all_position_verifier &&
                DebugEnv::isTruthyEnv("LLAMINAR_MTP_FORCE_VERIFIER_GRAPH_RECAPTURE"))
            {
                /*
                 * Diagnostic-only recapture splitter: Phase 10 tuning needs to
                 * distinguish stale all-position verifier graph inputs from
                 * stale MTP sidecar graph inputs.  This switch deliberately
                 * affects only verifier graphs so the benchmark matrix can
                 * identify the stale replay owner without disabling GPU graphs
                 * globally.
                 */
                capture_policy.force_recapture = true;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "main_verifier_force_recapture_diagnostic",
                    1.0,
                    "decode",
                    input.device.toString());
            }
            if (pending_all_position_verifier_recapture)
            {
                /*
                 * Shifted-MTP-KV catch-up changes verifier input state without
                 * broadly invalidating ordinary decode or sidecar replay.  The
                 * next all-position verifier must therefore refresh its GPU
                 * executable before publication consumes verifier rows.  Keep
                 * this as a one-shot structural recapture instead of relying on
                 * the diagnostic LLAMINAR_MTP_FORCE_VERIFIER_GRAPH_RECAPTURE
                 * environment switch.
                 */
                capture_policy.force_recapture = true;
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "main_verifier_pending_recapture",
                    1.0,
                    "decode",
                    input.device.toString());
            }
            if (capture_policy.allow_cached_graph_replay)
            {
                const DeviceId capture_device = ctx->deviceId();
                capture_policy.capture_boundary =
                    [&host, &input, capture_device](
                        const std::string &boundary_name,
                        void *capture_stream) -> bool
                {
                    return host.waitAtDecodeGraphCaptureBoundary(
                        input,
                        capture_device,
                        boundary_name,
                        capture_stream);
                };
            }
            const bool wants_all_position_sync_defer =
                all_position_verifier &&
                host.shouldDeferAllPositionVerifierFinalSync();
            const bool wants_main_decode_sync_defer =
                !all_position_verifier &&
                (host.shouldDeferMainDecodeFinalSync() ||
                 uses_retained_sparse_parent ||
                 uses_device_timeline_transaction ||
                 uses_heterogeneous_ticket_transaction);
            const bool wants_captured_collective_sync_defer =
                !all_position_verifier &&
                has_collective_nodes &&
                capture_policy.collectives_graph_capturable &&
                !debugEnv().gpu_stage_timing &&
                !debugEnv().gpu_stage_timing_detail;

            if (capture_policy.allow_cached_graph_replay &&
                wants_all_position_sync_defer)
            {
                capture_policy.defer_final_sync = true;
            }
            else if (capture_policy.allow_cached_graph_replay &&
                     (wants_main_decode_sync_defer ||
                      wants_captured_collective_sync_defer))
            {
                capture_policy.defer_final_sync = true;
            }
            if (uses_retained_sparse_parent ||
                uses_device_timeline_transaction ||
                uses_heterogeneous_ticket_transaction)
            {
                // The endpoint executable publishes an asynchronous device
                // timeline transaction; its consumer owns completion.
                capture_policy.defer_final_sync = true;
            }
            requested_deferred_main_decode_sync =
                wants_main_decode_sync_defer ||
                wants_captured_collective_sync_defer;
            executed_deferred_main_decode_sync =
                requested_deferred_main_decode_sync && capture_policy.defer_final_sync;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_capture_policy",
                1.0,
                "decode",
                input.device.toString(),
                {{"context", forwardGraphPerfContext(signature)},
                 {"allow_graph_replay", boolTag(capture_policy.allow_cached_graph_replay)},
                 {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                 {"has_collectives", boolTag(has_collective_nodes)},
                 {"heterogeneous_segmented", boolTag(capture_policy.heterogeneous_segmented_enabled)},
                 {"retained_sparse_parent", boolTag(uses_retained_sparse_parent)},
                 {"device_timeline_transaction", boolTag(uses_device_timeline_transaction)},
                 {"heterogeneous_ticket_transaction",
                  boolTag(uses_heterogeneous_ticket_transaction)},
                 {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)},
                 {"replay_plan_policy",
                  capture_policy.graph_replay_plan_policy ==
                          DeviceGraphExecutor::GraphReplayPlanPolicy::
                              RequireFullGraph
                      ? "require_full_graph"
                      : (capture_policy.graph_replay_plan_policy ==
                                 DeviceGraphExecutor::GraphReplayPlanPolicy::
                                     RequireRetainedParentComposition
                             ? "require_retained_parent_composition"
                             : (capture_policy.graph_replay_plan_policy ==
                                        DeviceGraphExecutor::GraphReplayPlanPolicy::
                                            RequireRetainedParentWithConcurrentTicketService
                                    ? "require_retained_parent_with_concurrent_ticket_service"
                                    : "allow_heterogeneous_boundary_segmentation"))}});

            if (capture_policy.allow_cached_graph_replay && !forward_cache.gpu_stream)
            {
                DeviceId dev_id = ctx->deviceId();
                if (dev_id.is_gpu())
                {
                    IWorkerGPUContext *gpu_ctx = host.getWorkerGPUContext(dev_id);
                    if (!gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for captured decode on "
                                  << dev_id.toString());
                        return false;
                    }
                    forward_cache.gpu_stream = gpu_ctx->defaultStream();
                    forward_cache.gpu_ctx = gpu_ctx;
                }
            }

            const bool main_decode_replay = !all_position_verifier;
            /*
             * Multi-token ordinary decode captures are live-state-versioned:
             * their graph mutates live KV/recurrent state inline.  All-position
             * verifier captures are deliberately not epoch-versioned here.
             * Verifier row state is published through stage-owned capture slots
             * and row metadata is refreshed before each launch, so preserving the
             * verifier executable is the canonical vLLM-style fast path.
             */
            const bool live_state_versioned_replay =
                main_decode_replay && input.seq_len > 1;
            const uint64_t live_state_epoch = host.liveReplayStateEpoch();
            if (forward_cache.requiresLiveStateEpochRecapture(
                    live_state_versioned_replay,
                    capture_policy.allow_cached_graph_replay,
                    live_state_epoch))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "decode_graph_state_epoch_recapture",
                    1.0,
                    "decode",
                    input.device.toString(),
                    {{"context", forwardGraphPerfContext(signature)},
                     {"old_epoch", std::to_string(forward_cache.graph_replay_live_state_epoch)},
                     {"new_epoch", std::to_string(live_state_epoch)}});
                forward_cache.resetReplayState();
            }

            // Replay retains the exact role that setup captured. A request-
            // batch condition is decode-shaped but is not an ordinary decode;
            // collapsing that distinction splits its capture/replay evidence.
            forward_cache.segment_cache.perf_context =
                forwardGraphPerfContext(signature);
            success = executor_.executeDecodeWithCapturePolicy(
                *forward_cache.graph,
                ctx,
                &forward_cache.segment_cache,
                forward_cache.gpu_stream,
                forward_cache.gpu_ctx,
                &forward_cache.collective_nodes,
                capture_policy,
                &used_graph_replay);
        }

        /*
         * Seal any live-state read admitted by the prelude before another
         * producer can reuse those persistent rows. This call is deliberately
         * outside the success branch: an executor failure can still follow
         * asynchronously enqueued GPU work, and that work retains ownership
         * until its exact stream reaches this event publication.
         */
        void *const exact_output_producer_stream = success
            ? (is_decode
                   ? forward_cache.decodeOutputProducerStream(
                         used_graph_replay
                             ? ForwardGraphCache::DecodeLaunchPath::CapturedReplay
                             : ForwardGraphCache::DecodeLaunchPath::Direct)
                   : prefill_producer_stream)
            : dynamic_param_stream;
        void *const live_state_completion_stream =
            exact_output_producer_stream;
        live_state_read.complete(live_state_completion_stream);

        if (success && !is_decode && !input.device.is_gpu() &&
            effectiveRealSeqLen(input) > 0)
        {
            const int terminal_row = effectiveRealSeqLen(input) - 1;
            if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                    *forward_cache.graph,
                    terminal_row,
                    nullptr,
                    "cpu_exact_prefill_cache_hit",
                    input.sequence_lengths_device,
                    input.batch_size,
                    input.seq_len,
                    input.sequence_lengths))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to publish terminal CPU prefill state after exact topology replay");
                return false;
            }
        }

        auto exec_t1 = std::chrono::high_resolution_clock::now();

        if (success && used_graph_replay &&
            forward_cache.segment_cache.initialized &&
            !forward_cache.segment_cache.needs_capture)
        {
            // Phase 3 replay doesn't call markCompleted(), so we can
            // skip graph.reset() on subsequent steps.
            forward_cache.phase3_active = true;
            if (is_decode)
            {
                const bool all_position_verifier =
                    host.computeAllPositionLogitsEnabled();
                const bool main_decode_replay = !all_position_verifier;
                const bool live_state_versioned_replay =
                    main_decode_replay && input.seq_len > 1;
                if (live_state_versioned_replay)
                {
                    forward_cache.graph_replay_live_state_epoch =
                        host.liveReplayStateEpoch();
                }
                if (all_position_verifier &&
                    all_position_verifier_recapture_pending_)
                {
                    /*
                     * Consume the request only once the cache is replay-ready.
                     * Warmup and capture phases are allowed to run after the
                     * request, but they are not sufficient evidence that later
                     * replay will use a fresh executable.
                     */
                    all_position_verifier_recapture_pending_ = false;
                    PerfStatsCollector::addCounter(
                        "forward_graph",
                        "main_verifier_pending_recapture_consumed",
                        1.0,
                        "decode",
                        input.device.toString());
                }
            }
        }
        else
        {
            forward_cache.phase3_active = false;
        }

        const bool main_decode_sync_deferred =
            success &&
            used_graph_replay &&
            executed_deferred_main_decode_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        if (success)
        {
            /*
             * Publish invocation provenance while `used_graph_replay` is still
             * in scope. A decode replay must name the segment cache's typed
             * capture stream, never the generic stream most recently applied
             * to stage bindings. DeviceGraphOrchestrator immediately records
             * its durable output event on this exact producer stream.
             */
            publishForwardOutputProvenance(
                output,
                input.device,
                exact_output_producer_stream,
                input.execution_role,
                is_decode,
                host.computeAllPositionLogitsEnabled(),
                forwardCompletionScopeForInput(input),
                input.seq_len,
                input.batch_size);
            if (input.device.is_gpu() && !output.execution.valid)
            {
                LOG_ERROR("[ForwardExecutionEngine] Successful cached GPU forward has no exact output producer stream"
                          << " device=" << input.device.toString()
                          << " decode=" << is_decode
                          << " replay=" << used_graph_replay);
                return false;
            }

            /*
             * Grouped verifier state is always consumed asynchronously by a
             * sampler, accepted-state publisher, or diagnostic observer. Publish
             * its completion edge for every successful GPU invocation, including
             * warmup and first capture. Whether the caller also requests a
             * host-facing logits boundary is independent of this device-state
             * ownership contract.
             */
            if (is_decode &&
                input.device.is_gpu() &&
                host.computeAllPositionLogitsEnabled())
            {
                host.setPendingAllPositionVerifierStream(
                    output.execution.stream);
            }
            if (main_decode_sync_deferred)
            {
                host.setPendingMainDecodeStream(
                    output.execution.stream);
            }
            else if (requested_deferred_main_decode_sync)
            {
                host.setPendingMainDecodeStream(nullptr);
            }
        }

        // Every successful graph publishes its tensor, even when completion is
        // deferred to a device consumer. A sampler's private stream handoff does
        // not publish a PP hidden tensor to TransferEngine on another stream.
        if (success)
        {
            if (!host.publishForwardResultAtBoundary(output, ctx))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to publish cached forward result at the device boundary");
                return false;
            }

            const std::string stage_context =
                is_decode
                    ? (forward_cache.segment_cache.perf_context.empty()
                           ? "main_decode"
                           : forward_cache.segment_cache.perf_context)
                    : "prefill";
            collectTimeline(host, ctx, is_decode, input, start, stage_context);
        }

        if (success)
        {
            recordMoEOverlayCollectiveTransaction(
                input,
                forward_cache.moe_overlay_collective_runtime_stages,
                "cache_hit");
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        /*
         * Keep one low-overhead end-to-end cache-hit timing in the same CSV as
         * the heterogeneous transport evidence.  The timer already brackets
         * every cache hit for the DEBUG summary below; exporting it therefore
         * adds no clock reads and lets Release campaigns distinguish capture
         * setup from steady graph replay without enabling log-heavy DEBUG.
         */
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                "forward_cache_hit_total",
                static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - start)
                        .count()),
                is_decode ? "decode" : "prefill",
                input.device.toString(),
                {{"phase3_active", forward_cache.phase3_active ? "true" : "false"},
                 {"sequence_rows", std::to_string(input.seq_len)},
                 {"used_graph_replay", used_graph_replay ? "true" : "false"}});
        }

        /* The always-on summary counter for the OpenMP wake-up work item: the
         * exact team fork/join count of one cached-forward CPU decode step.
         * The fork sites count whether or not anyone collects; the publisher
         * owns the policy gate and the record shape, so default serving
         * materializes nothing and every evidence run retains the structure it
         * must be able to assert. */
        if (success && is_decode && !input.device.is_gpu())
        {
            llaminar::v2::publishOpenMPTeamForkSummary(
                openmp_forks_at_exec_entry,
                input.device,
                [this, &signature] { return forwardGraphPerfContext(signature); },
                used_graph_replay);
        }

        // Decode step timing breakdown (enabled via TP_TIMING)
        if (is_decode && debugEnv().tp_timing)
        {
            double setup_us = std::chrono::duration<double, std::micro>(exec_t0 - start).count();
            double exec_us = std::chrono::duration<double, std::micro>(exec_t1 - exec_t0).count();
            double sync_us = std::chrono::duration<double, std::micro>(end - exec_t1).count();
            std::ostringstream subphase;
            if (profiling_setup)
            {
                subphase << " [ws=" << std::chrono::duration<double, std::micro>(setup_workspace_t1 - setup_workspace_t0).count()
                         << "us tok=" << std::chrono::duration<double, std::micro>(setup_token_copy_t1 - setup_token_copy_t0).count()
                         << "us strm=" << std::chrono::duration<double, std::micro>(setup_stream_t1 - setup_stream_t0).count()
                         << "us dyn=" << std::chrono::duration<double, std::micro>(setup_dynamic_params_t1 - setup_dynamic_params_t0).count()
                         << "us rst=" << std::chrono::duration<double, std::micro>(setup_graph_reset_t1 - setup_graph_reset_t0).count() << "us]";
            }
            LOG_DEBUG("[DEVICE_DECODE] dev=" << input.device
                                             << " setup=" << std::fixed << std::setprecision(1) << setup_us << "us"
                                             << " exec=" << exec_us << "us"
                                             << " sync=" << sync_us << "us"
                                             << " total=" << (ms * 1000.0) << "us"
                                             << " phase3=" << forward_cache.phase3_active
                                             << subphase.str());
        }

        // Forward-pass host phases are exported through the explicit PerfStats timing gate.
        if (profiling_setup)
        {
            ForwardPassProfiler::PhaseTimings timings;
            timings.setup_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(exec_t0 - start).count());
            timings.execute_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(exec_t1 - exec_t0).count());
            timings.sync_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - exec_t1).count());

            // Setup sub-phase timings
            timings.setup_workspace_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_workspace_t1 - setup_workspace_t0).count());
            timings.setup_token_copy_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_token_copy_t1 - setup_token_copy_t0).count());
            timings.setup_stream_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_stream_t1 - setup_stream_t0).count());
            timings.setup_dynamic_params_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_dynamic_params_t1 - setup_dynamic_params_t0).count());
            timings.setup_graph_reset_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(setup_graph_reset_t1 - setup_graph_reset_t0).count());

            // Graph replay sub-phase timings are populated by decode segmented
            // replay and prefill graph-cache launch paths via thread-local
            // ReplayPhaseTimings. Eager paths simply consume zeros.
            const auto &replay_timings = ForwardPassProfiler::consumeReplayTimings();
            timings.graph_launch_ns = replay_timings.graph_launch_ns;
            timings.post_launch_ns = replay_timings.post_launch_ns;
            timings.stream_sync_ns = replay_timings.stream_sync_ns;
            timings.manual_dispatch_ns =
                replay_timings.manual_dispatch_ns;
            timings.manual_sparse_protocol_ns =
                replay_timings.manual_sparse_protocol_ns;
            timings.manual_other_ns = replay_timings.manual_other_ns;

            if (is_decode)
                forward_pass_profiler_.recordDecodeIteration(
                    timings, input.device.toString());
            else
                forward_pass_profiler_.recordPrefillIteration(
                    timings, input.device.toString());
        }

        LOG_DEBUG("[ForwardExecutionEngine] Forward (cached) completed in "
                  << ms << "ms, success=" << success);

        return success;
    }

} // namespace llaminar2
