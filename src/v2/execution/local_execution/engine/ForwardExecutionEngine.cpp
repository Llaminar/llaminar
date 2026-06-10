/**
 * @file ForwardExecutionEngine.cpp
 * @brief Implementation of ForwardExecutionEngine
 *
 * Contains the forward graph execution logic extracted from
 * DeviceGraphOrchestrator::executeForward().
 *
 * Split into:
 * - execute():          Entry point — signature computation, cache dispatch
 * - executeCacheHit():  Reuse cached graph (buffer update, PP copy, execution)
 * - executeCacheMiss(): Build new graph, execute, populate cache
 * - collectTimeline():  GPU stage timing collection and printing
 */

#include "ForwardExecutionEngine.h"
#include "PrefillBucketUtils.h"
#include "../graph/GraphCaptureGuard.h"
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <unordered_set>

namespace llaminar2
{
    namespace
    {
        /// @brief Build a prefill graph-cache config from the cached debug environment.
        PrefillGraphConfig makePrefillGraphConfigFromEnv()
        {
            const auto &env = debugEnv();
            PrefillGraphConfig config;
            config.enabled = env.execution.gpu_graphs;
            config.min_seq_len = env.execution.prefill_graph_min_seq;
            config.trace = env.execution.prefill_graph_trace;
            config.buckets_enabled = env.execution.prefill_graph_buckets;
            config.bucket_sizes = env.execution.prefill_graph_bucket_sizes;
            config.max_cached_entries = static_cast<size_t>(env.execution.prefill_graph_max_cached_buckets);
            return config;
        }

        /// @brief Return the real-token count for an input, falling back to its execution length.
        int effectiveRealSeqLen(const ForwardInput &input)
        {
            return input.real_seq_len > 0 ? input.real_seq_len : input.seq_len;
        }

        /// @brief Return the fixed bucket length for an input, falling back to its execution length.
        int effectiveBucketSeqLen(const ForwardInput &input)
        {
            return input.bucket_seq_len > 0 ? input.bucket_seq_len : input.seq_len;
        }

        /// @brief Return true when this execution uses a padded fixed bucket.
        bool isPaddedBucketExecution(const ForwardInput &input)
        {
            const int real_seq_len = effectiveRealSeqLen(input);
            const int bucket_seq_len = effectiveBucketSeqLen(input);
            return real_seq_len > 0 && bucket_seq_len > 0 && real_seq_len < bucket_seq_len;
        }

        void *executionStreamForCache(const ForwardGraphCache &cache)
        {
            if (cache.applied_stream)
                return cache.applied_stream;
            if (cache.segment_cache.capture_stream)
                return cache.segment_cache.capture_stream;
            if (cache.gpu_stream)
                return cache.gpu_stream;
            if (cache.prefill_capture_stream.stream)
                return cache.prefill_capture_stream.stream;
            return nullptr;
        }

        /// @brief Return the absolute chunk offset for raw server-style inputs.
        int effectiveTokenOffset(const ForwardInput &input)
        {
            return input.token_offset != 0 ? input.token_offset : input.position_offset;
        }

        /// @brief Build the fixed-bucket ForwardInput view backed by a runtime plan.
        ForwardInput makeBucketedPrefillInput(
            const ForwardInput &base_input,
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan)
        {
            ForwardInput chunk_input = base_input;
            chunk_input.token_ids = plan.chunk.token_ids.data();
            chunk_input.position_ids = plan.chunk.position_ids.data();
            chunk_input.seq_len = plan.chunk.bucket_seq_len;
            chunk_input.real_seq_len = plan.chunk.real_count;
            chunk_input.bucket_seq_len = plan.chunk.bucket_seq_len;
            chunk_input.token_offset = plan.chunk.token_offset;
            chunk_input.position_offset = plan.chunk.token_offset;
            chunk_input.prefill_chunk_index = plan.chunk_index;
            return chunk_input;
        }

        PrefillChunkPlan makeMaintenanceChunkPlan(
            const ForwardExecutionEngine::PrefillChunkRuntimePlan &plan)
        {
            PrefillChunkPlan chunk;
            chunk.token_offset = plan.chunk.token_offset;
            chunk.real_count = plan.chunk.real_count;
            chunk.bucket_seq_len = plan.chunk.bucket_seq_len;
            chunk.chunk_index = plan.chunk_index;
            chunk.rebalance_allowed_after = plan.rebalance_allowed_after;
            chunk.rebalance_required_after = plan.rebalance_required_after;
            return chunk;
        }

        /// @brief Build the fixed-bucket replay metadata consumed by row-select,
        ///        LM-head, and KV-cache append stages.
        IComputeStage::PrefillReplayParams makePrefillReplayParams(const ForwardInput &input)
        {
            return IComputeStage::PrefillReplayParams{
                effectiveRealSeqLen(input),
                effectiveBucketSeqLen(input),
                input.token_offset};
        }

        /// @brief Collect stages that need real-token metadata for padded prefill execution.
        std::vector<IComputeStage *> collectPrefillReplayParamStages(ComputeGraph &graph)
        {
            std::vector<IComputeStage *> stages;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                ComputeNode *node = graph.getNode(node_name);
                if (node && node->stage && node->stage->hasPrefillReplayParams())
                    stages.push_back(node->stage.get());
            }
            return stages;
        }

        /// @brief Push real-token metadata to fixed-bucket stages before any prefill execution path.
        void updatePrefillReplayParamStages(
            const ForwardInput &input,
            const std::vector<IComputeStage *> &stages)
        {
            const auto replay_params = makePrefillReplayParams(input);
            for (auto *stage : stages)
            {
                if (stage)
                    stage->updatePrefillReplayParams(replay_params);
            }
        }

        /// @brief Collect graph nodes whose stage type blocks monolithic prefill graph capture.
        std::unordered_set<std::string> collectCollectiveNodeNames(const ComputeGraph &graph)
        {
            std::unordered_set<std::string> collective_nodes;
            for (const auto &node_name : graph.getExecutionOrder())
            {
                const auto *node = graph.getNode(node_name);
                if (!node || !node->stage)
                    continue;

                const auto type = node->stage->type();
                if (type == ComputeStageType::ALLREDUCE ||
                    type == ComputeStageType::ALLGATHER ||
                    type == ComputeStageType::ALLGATHER_V)
                {
                    collective_nodes.insert(node_name);
                }
            }
            return collective_nodes;
        }

        std::string boolTag(bool value)
        {
            return value ? "true" : "false";
        }

        const char *prefillGraphPhaseName(PrefillGraphPhase phase)
        {
            switch (phase)
            {
            case PrefillGraphPhase::Disabled:
                return "disabled";
            case PrefillGraphPhase::Cold:
                return "cold";
            case PrefillGraphPhase::Warmup:
                return "warmup";
            case PrefillGraphPhase::Capturing:
                return "capturing";
            case PrefillGraphPhase::Ready:
                return "ready";
            }
            return "unknown";
        }

        PrefillGraphCacheKey makePrefillGraphKey(
            const ForwardInput &input,
            const IForwardExecutionHost &host)
        {
            PrefillGraphCacheKey key;
            key.seq_len = input.seq_len;
            key.device_id = input.device;
            key.domain_id = host.prefillGraphDomainId();
            key.participant_id = host.prefillGraphParticipantId();
            key.placement_epoch = host.moePlacementEpoch();
            key.topology_signature = host.prefillGraphTopologySignature();
            return key;
        }

        PrefillGraphExecutionObservation makePrefillGraphObservation(
            const ForwardInput &input,
            const PrefillGraphCacheKey &key,
            const char *capture_phase,
            const std::string &recapture_reason)
        {
            const int real_count = effectiveRealSeqLen(input);
            const int bucket_len = effectiveBucketSeqLen(input);
            PrefillGraphExecutionObservation observation;
            observation.valid = true;
            observation.chunk_index = input.prefill_chunk_index;
            observation.bucket_seq_len = bucket_len;
            observation.real_token_start = effectiveTokenOffset(input);
            observation.real_token_count = real_count;
            observation.real_token_end = observation.real_token_start + real_count;
            observation.domain_id = key.domain_id;
            observation.participant_id = key.participant_id;
            observation.placement_epoch = key.placement_epoch;
            observation.topology_signature = key.topology_signature;
            observation.capture_phase = capture_phase ? capture_phase : "unknown";
            observation.recapture_reason = recapture_reason.empty() ? "none" : recapture_reason;
            return observation;
        }

        PerfStatsCollector::Tags prefillGraphObservationTags(
            const PrefillGraphExecutionObservation &observation,
            const char *cache_phase)
        {
            return {
                {"capture_phase", observation.capture_phase},
                {"cache_phase", cache_phase ? cache_phase : "unknown"},
                {"chunk_index", std::to_string(observation.chunk_index)},
                {"bucket_seq_len", std::to_string(observation.bucket_seq_len)},
                {"real_token_start", std::to_string(observation.real_token_start)},
                {"real_token_count", std::to_string(observation.real_token_count)},
                {"real_token_end", std::to_string(observation.real_token_end)},
                {"domain_id", observation.domain_id},
                {"participant_id", std::to_string(observation.participant_id)},
                {"placement_epoch", std::to_string(observation.placement_epoch)},
                {"topology_signature", std::to_string(observation.topology_signature)},
                {"recapture_reason", observation.recapture_reason}};
        }

        void publishPrefillGraphObservation(
            ForwardGraphCache &forward_cache,
            const ForwardInput &input,
            const PrefillGraphCacheKey &key,
            PrefillGraphPhase cache_phase,
            const char *capture_phase,
            const std::string &recapture_reason)
        {
            auto observation = makePrefillGraphObservation(
                input,
                key,
                capture_phase,
                recapture_reason);
            forward_cache.last_prefill_graph_observation = observation;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_graph_lifecycle",
                1.0,
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
        }

        std::string forwardGraphPerfContext(const ForwardGraphSignature &signature)
        {
            if (!signature.decode)
                return signature.is_bucketed_prefill ? "prefill_bucket" : "prefill";
            return signature.all_position_logits ? "main_verifier" : "main_decode";
        }

        PerfStatsCollector::Tags forwardCacheLookupTags(
            const ForwardGraphSignature &signature,
            const char *result)
        {
            return {
                {"context", forwardGraphPerfContext(signature)},
                {"result", result},
                {"seq_len", std::to_string(signature.seq_len)},
                {"decode_has_history", boolTag(signature.decode_has_history)},
                {"all_position_logits", boolTag(signature.all_position_logits)},
                {"all_position_logit_rows", std::to_string(signature.all_position_logit_rows)},
                {"uses_device_token_ids", boolTag(signature.uses_device_token_ids)},
                {"moe_placement_epoch", std::to_string(signature.moe_placement_epoch)}};
        }

        /// @brief Run the full prefill graph preflight for a fixed-bucket input.
        PrefillGraphRejectReason preflightPrefillGraph(
            const PrefillGraphCache &cache,
            const ComputeGraph &graph,
            const PrefillGraphCacheKey &key,
            const std::unordered_set<std::string> &collective_nodes,
            const ForwardInput &input,
            bool snapshots_active,
            bool moe_rebalancing_active)
        {
            return cache.preflight(
                graph,
                key,
                &collective_nodes,
                snapshots_active,
                moe_rebalancing_active,
                effectiveRealSeqLen(input),
                effectiveBucketSeqLen(input));
        }

        /// @brief Deterministic tie-breaker for bucketed forward-cache LRU victims.
        bool bucketedSignatureLessForEviction(
            const ForwardGraphSignature &lhs,
            const ForwardGraphSignature &rhs)
        {
            if (lhs.bucket_seq_len != rhs.bucket_seq_len)
                return lhs.bucket_seq_len < rhs.bucket_seq_len;
            if (lhs.device != rhs.device)
                return lhs.device < rhs.device;
            if (lhs.seq_len != rhs.seq_len)
                return lhs.seq_len < rhs.seq_len;
            if (lhs.all_position_logits != rhs.all_position_logits)
                return lhs.all_position_logits < rhs.all_position_logits;
            if (lhs.all_position_logit_rows != rhs.all_position_logit_rows)
                return lhs.all_position_logit_rows < rhs.all_position_logit_rows;
            return lhs.batch_size < rhs.batch_size;
        }
    }

    // =========================================================================
    // Construction
    // =========================================================================

    ForwardExecutionEngine::ForwardExecutionEngine(Config config, DeviceGraphExecutor &executor)
        : config_(std::move(config)), executor_(executor)
    {
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

        if (!input.token_ids)
        {
            plan.error = "bucketed prefill requires token_ids";
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

        plan.padding_required = !plan.selection.exact;
        plan.chunk.token_offset = input.token_offset;
        plan.chunk.real_count = real_seq_len;
        plan.chunk.bucket_seq_len = plan.selection.bucket_seq_len;
        plan.chunk.token_ids = padPrefillTokensToBucket(
            input.token_ids,
            real_seq_len,
            plan.selection.bucket_seq_len,
            pad_token_id);
        plan.chunk.position_ids = buildPrefillChunkPositionIds(
            real_seq_len,
            plan.selection.bucket_seq_len,
            input.token_offset,
            input.batch_size);

        if (plan.chunk.token_ids.empty() || plan.chunk.position_ids.empty())
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

        if (!input.token_ids)
        {
            runtime_schedule.error = "bucketed prefill schedule requires token_ids";
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
            chunk_input.token_ids = input.token_ids + relative_offset;
            chunk_input.seq_len = chunk.real_count;
            chunk_input.real_seq_len = chunk.real_count;
            chunk_input.bucket_seq_len = 0;
            chunk_input.token_offset = chunk.token_offset;
            chunk_input.position_offset = chunk.token_offset;

            auto runtime_plan = prepareSinglePrefillChunkRuntimePlan(
                chunk_input,
                policy.bucket_sizes,
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

        // Preserve the caller's execution context and swap in the prepared
        // chunk buffers plus fixed-bucket metadata for the delegated launch.
        ForwardInput chunk_input = makeBucketedPrefillInput(base_input, plan);

        if (!execute(chunk_input, output, host))
            return false;

        const PrefillChunkPlan maintenance_chunk = makeMaintenanceChunkPlan(plan);
        const PrefillChunkMaintenanceState maintenance_state =
            host.prefillChunkMaintenanceState(maintenance_chunk);
        const PrefillChunkMaintenanceDecision maintenance_decision =
            evaluatePrefillChunkMaintenance(maintenance_chunk, maintenance_state);

        const bool maintenance_was_requested =
            maintenance_state.rebalance_requested ||
            maintenance_chunk.rebalance_required_after;
        if (maintenance_was_requested && !maintenance_decision)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance blocked after chunk "
                      << maintenance_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }
        if (maintenance_decision.can_run &&
            !host.onPrefillChunkMaintenance(maintenance_chunk, maintenance_decision))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill chunk maintenance failed after chunk "
                      << maintenance_chunk.chunk_index << ": "
                      << maintenance_decision.reason);
            return false;
        }

        return true;
    }

    bool ForwardExecutionEngine::runPrefillChunkSchedule(
        const ForwardInput &base_input,
        const PrefillChunkRuntimeSchedule &schedule,
        ForwardOutput &output,
        IForwardExecutionHost &host)
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

        for (const auto &chunk_plan : schedule.chunks)
        {
            if (!runPrefillChunk(base_input, chunk_plan, output, host))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill chunk schedule failed at chunk "
                          << chunk_plan.chunk_index);
                return false;
            }
        }

        return true;
    }

    void ForwardExecutionEngine::invalidateAll()
    {
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

    void ForwardExecutionEngine::resetCapturedReplayState()
    {
        for (auto &entry : cache_)
        {
            entry.second.resetReplayState();
        }
    }

    ForwardExecutionEngine::ReplayStateResetSummary
    ForwardExecutionEngine::resetCapturedReplayStateForCorrectionReplay(
        uint64_t live_state_epoch)
    {
        ReplayStateResetSummary summary;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                chooseForwardReplayStateAction(
                    ForwardReplayStateMutationKind::MTPCorrectionReplayBoundary,
                    cache_class);
            if (action == ForwardReplayStateAction::ResetReplayState)
            {
                cache.resetReplayState();
                ++summary.reset_replay_state;
                if (cache_class == ForwardReplayStateCacheClass::OrdinaryDecode)
                    ++summary.ordinary_decode_reset;
            }
            else
            {
                cache.markGPUStreamBindingsDirty();
                if (cache_class == ForwardReplayStateCacheClass::SingleTokenOrdinaryDecode)
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

    void ForwardExecutionEngine::touchBucketedPrefillForwardCache(
        const ForwardGraphSignature &signature,
        ForwardGraphCache &cache)
    {
        if (!signature.is_bucketed_prefill || !cache.valid)
            return;
        cache.bucketed_prefill_last_access_tick = ++bucketed_prefill_forward_access_counter_;
    }

    size_t ForwardExecutionEngine::bucketedPrefillForwardCacheSize() const
    {
        size_t count = 0;
        for (const auto &[signature, cache] : cache_)
        {
            if (signature.is_bucketed_prefill && cache.valid)
                ++count;
        }
        return count;
    }

    void ForwardExecutionEngine::enforceBucketedPrefillForwardCapacity(
        const ForwardGraphSignature *active_signature)
    {
        const int configured_cap = debugEnv().execution.prefill_graph_max_cached_buckets;
        if (configured_cap <= 0)
            return;

        const size_t cap = static_cast<size_t>(configured_cap);
        while (bucketedPrefillForwardCacheSize() > cap)
        {
            auto victim = cache_.end();
            uint64_t oldest_tick = std::numeric_limits<uint64_t>::max();

            // The cap only applies to reusable bucketed prefill graphs. Decode
            // and non-bucketed prefill retain their existing cache lifetime.
            for (auto it = cache_.begin(); it != cache_.end(); ++it)
            {
                const auto &signature = it->first;
                const auto &cache = it->second;
                if (!signature.is_bucketed_prefill || !cache.valid)
                    continue;
                if (active_signature && signature == *active_signature)
                    continue;

                const uint64_t tick = cache.bucketed_prefill_last_access_tick;
                if (victim == cache_.end() ||
                    tick < oldest_tick ||
                    (tick == oldest_tick && bucketedSignatureLessForEviction(signature, victim->first)))
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
            ++bucketed_prefill_forward_eviction_count_;

            LOG_INFO("[ForwardExecutionEngine] Evicted bucketed prefill forward graph bucket_seq_len="
                     << evicted_signature.bucket_seq_len
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
        IForwardExecutionHost &host)
    {
        last_executed_forward_graph_ = {};

        auto start = std::chrono::high_resolution_clock::now();

        const int first_position = input.position_ids ? input.position_ids[0] : input.position_offset;
        // Classify the execution path. Greedy MTP verifies short multi-token
        // continuations against an existing KV/GDN history; those must use the
        // decode graph path rather than the prompt-prefill path.
        const int decode_max_seq_len = std::max(1, config_.cache_config.decode_seq_len);
        const bool is_single_token_decode = (input.seq_len == 1 && input.batch_size <= 1);
        const bool is_short_continuation_decode =
            input.batch_size <= 1 &&
            input.seq_len > 1 &&
            input.seq_len <= decode_max_seq_len &&
            first_position > 0;
        const bool is_decode = is_single_token_decode || is_short_continuation_decode;
        const bool decode_has_history = is_decode && first_position > 0;
        const bool all_position_logits = host.computeAllPositionLogitsEnabled();
        const bool has_unified_pp = config_.has_unified_pp;
        const bool is_standard_path = !has_unified_pp && !config_.pp_stage_config.has_value();
        const bool is_partial_pp_path = !has_unified_pp && config_.pp_stage_config.has_value();

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
        const bool has_stable_forward_inputs =
            (((input.token_ids != nullptr) || (input.token_ids_device != nullptr)) &&
             (input.position_ids != nullptr)) ||
            (is_pp_non_embedding_stage && (input.position_ids != nullptr));

        const auto &env = debugEnv();
        const int input_real_seq_len = effectiveRealSeqLen(input);
        const bool prefill_graph_min_seq_met =
            input_real_seq_len >= env.execution.prefill_graph_min_seq;

        // Prefill graph caching: eligible for GPU devices with gpu_graphs enabled.
        // The minimum-length gate is evaluated against the caller's real input
        // before any bucket padding so ordinary short prompts take the normal
        // prefill path instead of becoming unsupported padded graph attempts.
        const bool prefill_cache_eligible =
            config_.cache_config.enabled &&
            !is_decode &&
            !has_unified_pp &&
            has_stable_forward_inputs &&
            (is_standard_path || is_partial_pp_path) &&
            input.device.is_gpu() &&
            env.execution.gpu_graphs &&
            prefill_graph_min_seq_met;

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

        if (!is_decode && input.bucket_seq_len <= 0 && env.execution.prefill_graph_buckets &&
            prefill_graph_min_seq_met && input.token_ids && input.batch_size == 1)
        {
            const int real_seq_len = input_real_seq_len;
            const auto selection = selectPrefillGraphBucket(
                real_seq_len,
                env.execution.prefill_graph_bucket_sizes);
            if (!selection)
            {
                LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph request rejected: "
                          << selection.error << " (seq_len=" << input.seq_len << ")");
                return false;
            }
            if (!selection.exact && !prefill_cache_eligible)
            {
                // Non-GPU devices cannot use padded graph buckets — fall through
                // to unbucketed prefill execution (no error, just skip bucketing).
                LOG_DEBUG("[ForwardExecutionEngine] Skipping padded bucket for non-GPU device: real_seq_len="
                          << real_seq_len << " bucket_seq_len=" << selection.bucket_seq_len
                          << " device=" << input.device.toString());
            }
        }

        if (prefill_cache_eligible && env.execution.prefill_graph_buckets)
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
                bucketed_prefill = true;
                bucketed_prefill_seq_len = input.bucket_seq_len;
            }
            else
            {
                ForwardInput planning_input = input;
                planning_input.token_offset = effectiveTokenOffset(input);
                raw_bucket_plan = prepareSinglePrefillChunkRuntimePlan(
                    planning_input,
                    env.execution.prefill_graph_bucket_sizes,
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
             !has_unified_pp &&
             has_stable_forward_inputs &&
             (is_standard_path || is_partial_pp_path)) ||
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
                bucketed_prefill ? bucketed_prefill_seq_len : effective_input.seq_len,
                effective_input.batch_size,
                effective_input.device,
                is_decode,
                decode_has_history,
                all_position_logits,
                all_position_logits ? std::max(0, host.allPositionLogitRows()) : 0,
                input.token_ids_device != nullptr,
                is_standard_path,
                config_.pp_stage_config.has_value(),
                pp_first_layer,
                pp_last_layer,
                pp_has_embedding,
                pp_has_lm_head,
                bucketed_prefill,
                bucketed_prefill ? bucketed_prefill_seq_len : 0,
                host.moePlacementEpoch()};

            auto cache_it = cache_.find(forward_signature);
            if (cache_it != cache_.end())
            {
                active_forward_cache = &cache_it->second;
            }
        }

        const bool use_cached_forward = forward_cache_eligible &&
                                        active_forward_cache &&
                                        active_forward_cache->valid;

        if (use_cached_forward)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                forwardCacheLookupTags(forward_signature, "hit"));
            touchBucketedPrefillForwardCache(forward_signature, *active_forward_cache);
            const bool success = executeCacheHit(effective_input, output, *active_forward_cache, host,
                                                 is_decode, start);
            if (success)
                recordLastExecutedForwardGraph(forward_signature, /*cache_hit=*/true);
            if (success && forward_signature.is_bucketed_prefill)
                enforceBucketedPrefillForwardCapacity(&forward_signature);
            return success;
        }

        // Cache MISS path
        ForwardGraphCache *build_cache = nullptr;
        bool should_cache_after_build = false;
        if (forward_cache_eligible)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "forward_cache_lookup",
                1.0,
                forward_signature.decode ? "decode" : "prefill",
                forward_signature.device.toString(),
                forwardCacheLookupTags(forward_signature, "miss"));
            auto [it, _inserted] = cache_.try_emplace(forward_signature);
            build_cache = &it->second;
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
                                              has_unified_pp, start);
        if (success && build_cache && build_cache->valid)
            recordLastExecutedForwardGraph(forward_signature, /*cache_hit=*/false);
        return success;
    }

    void ForwardExecutionEngine::recordLastExecutedForwardGraph(
        const ForwardGraphSignature &signature,
        bool cache_hit)
    {
        last_executed_forward_graph_.valid = true;
        last_executed_forward_graph_.signature = signature;
        last_executed_forward_graph_.cache_hit = cache_hit;
    }

    std::optional<ForwardExecutionEngine::LastExecutedForwardGraphView>
    ForwardExecutionEngine::lastExecutedForwardGraph()
    {
        if (!last_executed_forward_graph_.valid)
            return std::nullopt;

        auto it = cache_.find(last_executed_forward_graph_.signature);
        if (it == cache_.end() || !it->second.valid || !it->second.graph)
            return std::nullopt;

        LastExecutedForwardGraphView view;
        view.graph = it->second.graph.get();
        view.signature = last_executed_forward_graph_.signature;
        view.device = view.signature.device;
        view.stream = executionStreamForCache(it->second);
        view.cache_hit = last_executed_forward_graph_.cache_hit;
        view.is_decode = view.signature.decode;
        view.all_position_logits = view.signature.all_position_logits;
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
            observation.segmented_capture_live_state_epoch =
                cache.segmented_capture_live_state_epoch;
            observation.requires_live_state_epoch_recapture =
                cache.requiresLiveStateEpochRecapture(
                    classifyForwardReplayStateCache(signature) ==
                        ForwardReplayStateCacheClass::OrdinaryDecode,
                    /*segmented_capture_allowed=*/true,
                    live_state_epoch);
            observations.push_back(observation);
        }
        return observations;
    }

    // =========================================================================
    // executeCacheHit() — Reuse Cached Forward Graph
    // =========================================================================

    bool ForwardExecutionEngine::executeCacheHit(
        const ForwardInput &input,
        ForwardOutput &output,
        ForwardGraphCache &forward_cache,
        IForwardExecutionHost &host,
        bool is_decode,
        std::chrono::high_resolution_clock::time_point start)
    {
        // ===== CACHE HIT: Reuse cached decode graph =====

        // Setup sub-phase timing (only when profiling is active)
        const bool profiling_setup = KernelProfiler::isEnabled();
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
                if (!host.ensureDeviceWorkspaceAllocated(*forward_cache.graph, input.seq_len))
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

        // PP hidden state copy: for non-embedding PP stages, copy the
        // external hidden state (from previous PP stage) to the working
        // buffer before executing the cached graph.
        if (forward_cache.pp_needs_copy &&
            forward_cache.pp_external_hidden_state &&
            forward_cache.pp_working_buffer)
        {
            // PP hidden-state handoff. copyActivation picks the cheapest transport
            // for the (source authoritative device -> pp_device) pair:
            //   - same physical GPU   -> intra-VRAM device-to-device memcpy
            //   - same-vendor diff GPU -> peer copy (NCCL/RCCL or peer DMA)
            //   - cross-vendor / host  -> single host-staged bounce
            // and leaves the destination DEVICE_AUTHORITATIVE on pp_device.
            //
            // In the steady-state heterogeneous CUDA->ROCm case the producing
            // stage's transferActivation() has already moved the external hidden
            // state onto pp_device (rocm:0), so this resolves to an intra-GPU D2D
            // copy. This replaces the old data()/memcpy/ensureOnDevice host
            // round-trip that cost ~30ms/token.
            const bool pp_time = is_decode && debugEnv().tp_timing;
            auto pp_t0 = pp_time ? Clock::now() : Clock::time_point{};

            auto pp_result = TransferEngine::instance().copyActivation(
                forward_cache.pp_external_hidden_state, forward_cache.pp_working_buffer,
                forward_cache.pp_device, forward_cache.pp_copy_bytes);

            // Fail loud: a failed hidden-state handoff corrupts the entire
            // downstream pipeline stage, so there is no safe fallback.
            if (!pp_result.success)
            {
                LOG_ERROR("[PP_COPY] copyActivation failed on "
                          << forward_cache.pp_device << ": " << pp_result.error);
                throw std::runtime_error("PP hidden-state copy failed: " + pp_result.error);
            }

            if (pp_time)
            {
                auto pp_t1 = Clock::now();
                double total_us =
                    std::chrono::duration<double, std::micro>(pp_t1 - pp_t0).count();
                LOG_DEBUG("[PP_COPY] dev=" << forward_cache.pp_device << " method="
                          << to_string(pp_result.method_used) << " total=" << total_us
                          << "us bytes=" << forward_cache.pp_copy_bytes);
            }
        }

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

        const bool decode_capture_allowed =
            is_decode &&
            input.batch_size <= 1 &&
            input.seq_len <= std::max(1, config_.cache_config.decode_seq_len);

        if (stream_device.is_gpu())
        {
            stream_ctx = host.getDeviceContext(stream_device);
            if (stream_ctx && stream_ctx->deviceId().is_gpu())
            {
                try
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    stream_gpu_ctx = &pool.getContext(stream_ctx->deviceId());
                    forward_cache.gpu_ctx = stream_gpu_ctx;
                    if (!forward_cache.gpu_stream)
                        forward_cache.gpu_stream = stream_gpu_ctx->defaultStream();
                    dynamic_param_stream = forward_cache.gpu_stream;
                }
                catch (const std::exception &e)
                {
                    LOG_DEBUG("[ForwardExecutionEngine] Could not resolve worker GPU context for "
                              << stream_ctx->deviceId().toString()
                              << " before cached dynamic params: " << e.what());
                }
            }
        }

        if (decode_capture_allowed && stream_ctx && stream_ctx->deviceId().is_gpu() && stream_gpu_ctx)
        {
            const auto early_capture_policy = host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                stream_ctx,
                forward_cache.segment_cache.consecutive_failures);

            if (early_capture_policy.allow_segmented_capture &&
                forward_cache.segment_cache.consecutive_failures < early_capture_policy.max_segment_failures)
            {
                if (forward_cache.segment_cache.ensureCaptureStream(
                        stream_gpu_ctx,
                        stream_ctx->deviceId()))
                {
                    replay_stream = forward_cache.segment_cache.capture_stream;
                    dynamic_param_stream = replay_stream;
                }
                else
                {
                    LOG_WARN("[ForwardExecutionEngine] Failed to create explicit decode graph stream for "
                             << stream_ctx->deviceId().toString()
                             << "; cached GPU dynamic params will use the worker stream");
                }
            }
        }

        if (!is_decode && stream_gpu_ctx)
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

        if (profiling_setup)
        {
            setup_stream_t1 = Clock::now();
            setup_dynamic_params_t0 = setup_stream_t1;
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

        // Update position-dependent params using cached stage pointers.
        // Only ~4 stages override updateDynamicParams() — avoids iterating
        // all ~339 stages with hash lookups on every decode step.
        if (!forward_cache.dynamic_param_stages_cached)
        {
            forward_cache.dynamic_param_stages.clear();
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->hasDynamicParams())
                    forward_cache.dynamic_param_stages.push_back(node->stage.get());
            }
            forward_cache.dynamic_param_stages_cached = true;
        }

        // Cache replay callback stages (KVCacheAppend, MoERouting, etc.)
        // Called after monolithic graph replay to advance host-side metadata.
        if (!forward_cache.replay_callback_stages_cached)
        {
            forward_cache.replay_callback_stages.clear();
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->needsOnGraphReplayed())
                    forward_cache.replay_callback_stages.push_back(node->stage.get());
            }
            forward_cache.replay_callback_stages_cached = true;
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
        for (auto *stage : forward_cache.dynamic_param_stages)
        {
            stage->updateDynamicParams(input.position_offset, input.seq_len);
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

        // Graph capture (segmented/monolithic) is beneficial for decode graphs
        // whose shape is stable across steps. This includes one-token decode and
        // the fixed short-continuation verifier used by greedy MTP. Prompt prefill
        // remains outside this path because prompt shapes vary and capture setup
        // is not amortized.
        bool used_segmented_capture = false;
        bool requested_deferred_all_position_sync = false;
        bool requested_deferred_main_decode_sync = false;
        bool executed_deferred_all_position_sync = false;
        bool executed_deferred_main_decode_sync = false;

        auto exec_t0 = std::chrono::high_resolution_clock::now();
        if (profiling_setup)
        {
            ForwardPassProfiler::resetReplayTimings();
        }

        if (!is_decode)
        {
            if (executor_.config().snapshot_callback)
            {
                // Snapshot tests need callbacks on cached prefill replays too.
                // executeFastDecode() intentionally disables callbacks, so use
                // the full policy whenever capture is enabled.
                success = executor_.execute(*forward_cache.graph, ctx);
            }
            else
            {
                // Prefill with graph capture/replay state machine
                success = executePrefillWithGraphCache(input, forward_cache, ctx, host);
            }
        }
        else
        {
            DeviceGraphExecutor::DecodeCapturePolicy capture_policy;
            if (decode_capture_allowed)
            {
                capture_policy = host.buildDecodeCapturePolicy(
                    has_collective_nodes,
                    ctx,
                    forward_cache.segment_cache.consecutive_failures);
            }
            if (capture_policy.collective_segmented_enabled)
            {
                LOG_DEBUG("[ForwardExecutionEngine] Experimental collective segmented GPU-graph replay enabled");
            }

            const bool all_position_verifier =
                host.computeAllPositionLogitsEnabled();
            const bool wants_all_position_sync_defer =
                all_position_verifier &&
                host.shouldDeferAllPositionVerifierFinalSync();
            const bool wants_main_decode_sync_defer =
                !all_position_verifier &&
                host.shouldDeferMainDecodeFinalSync();

            if (capture_policy.allow_segmented_capture &&
                wants_all_position_sync_defer)
            {
                capture_policy.defer_final_sync = true;
            }
            else if (capture_policy.allow_segmented_capture &&
                     wants_main_decode_sync_defer)
            {
                capture_policy.defer_final_sync = true;
            }
            requested_deferred_all_position_sync =
                wants_all_position_sync_defer;
            requested_deferred_main_decode_sync =
                wants_main_decode_sync_defer;
            executed_deferred_all_position_sync =
                wants_all_position_sync_defer && capture_policy.defer_final_sync;
            executed_deferred_main_decode_sync =
                wants_main_decode_sync_defer && capture_policy.defer_final_sync;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_capture_policy",
                1.0,
                "decode",
                input.device.toString(),
                {{"context", all_position_verifier ? "main_verifier" : "main_decode"},
                 {"allow_segmented", boolTag(capture_policy.allow_segmented_capture)},
                 {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                 {"has_collectives", boolTag(has_collective_nodes)},
                 {"collective_segmented", boolTag(capture_policy.collective_segmented_enabled)},
                 {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)}});

            if (capture_policy.allow_segmented_capture && !forward_cache.gpu_stream)
            {
                DeviceId dev_id = ctx->deviceId();
                if (dev_id.is_gpu())
                {
                    auto &pool = GPUDeviceContextPool::instance();
                    IWorkerGPUContext &gpu_ctx = pool.getContext(dev_id);
                    forward_cache.gpu_stream = gpu_ctx.defaultStream();
                    forward_cache.gpu_ctx = &gpu_ctx;
                }
            }

            const bool main_decode_replay =
                !host.computeAllPositionLogitsEnabled();
            const uint64_t live_state_epoch = host.liveReplayStateEpoch();
            if (forward_cache.requiresLiveStateEpochRecapture(
                    main_decode_replay,
                    capture_policy.allow_segmented_capture,
                    live_state_epoch))
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "decode_segmented_state_epoch_recapture",
                    1.0,
                    "decode",
                    input.device.toString(),
                    {{"context", "main_decode"},
                     {"old_epoch", std::to_string(forward_cache.segmented_capture_live_state_epoch)},
                     {"new_epoch", std::to_string(live_state_epoch)}});
                forward_cache.resetReplayState();
            }

            forward_cache.segment_cache.perf_context =
                all_position_verifier ? "main_verifier" : "main_decode";
            success = executor_.executeDecodeWithCapturePolicy(
                *forward_cache.graph,
                ctx,
                &forward_cache.segment_cache,
                forward_cache.gpu_stream,
                forward_cache.gpu_ctx,
                &forward_cache.collective_nodes,
                capture_policy,
                &used_segmented_capture);
        }

        auto exec_t1 = std::chrono::high_resolution_clock::now();

        if (success && used_segmented_capture &&
            forward_cache.segment_cache.initialized &&
            !forward_cache.segment_cache.needs_capture)
        {
            // Phase 3 replay doesn't call markCompleted(), so we can
            // skip graph.reset() on subsequent steps.
            forward_cache.phase3_active = true;
            if (is_decode && !host.computeAllPositionLogitsEnabled())
                forward_cache.segmented_capture_live_state_epoch =
                    host.liveReplayStateEpoch();
        }
        else
        {
            forward_cache.phase3_active = false;
        }

        const bool all_position_verifier_sync_deferred =
            success &&
            forward_cache.phase3_active &&
            executed_deferred_all_position_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        const bool main_decode_sync_deferred =
            success &&
            forward_cache.phase3_active &&
            executed_deferred_main_decode_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        if (all_position_verifier_sync_deferred)
        {
            host.setPendingAllPositionVerifierStream(
                forward_cache.segment_cache.capture_stream);
        }
        else if (requested_deferred_all_position_sync)
        {
            host.setPendingAllPositionVerifierStream(nullptr);
        }
        if (main_decode_sync_deferred)
        {
            host.setPendingMainDecodeStream(
                forward_cache.segment_cache.capture_stream);
        }
        else if (requested_deferred_main_decode_sync)
        {
            host.setPendingMainDecodeStream(nullptr);
        }

        // Sync the stream at the forward pass boundary so logits are
        // immediately available to the caller without per-access event waits.
        if (success)
        {
            if (forward_cache.phase3_active)
            {
                // Phase 3 replay normally synchronizes both capture_stream
                // and defaultStream at the end of executeReplayPhase(). When a
                // GPU logits consumer explicitly inherits the replay stream,
                // that consumer owns ordering and host-visible mapped logits
                // must not be marked fresh at this boundary.
                if (!all_position_verifier_sync_deferred &&
                    !main_decode_sync_deferred)
                {
                    TensorBase *logits = host.logitsTensor();
                    if (logits && logits->isMapped())
                    {
                        logits->markMappedSynced();
                    }
                }
            }
            else
            {
                if (used_segmented_capture &&
                    forward_cache.segment_cache.capture_stream &&
                    forward_cache.gpu_ctx)
                {
                    // Warmup/capture phases run stage kernels on the capture
                    // stream, but the ordinary boundary sync only covers the
                    // device context/default stream. Synchronize the capture
                    // stream before callers sample logits from verifier graphs.
                    forward_cache.gpu_ctx->synchronizeStream(
                        forward_cache.segment_cache.capture_stream);
                }
                host.syncLogitsAtBoundary(ctx);
            }

            const std::string stage_context =
                is_decode
                    ? (forward_cache.segment_cache.perf_context.empty()
                           ? "main_decode"
                           : forward_cache.segment_cache.perf_context)
                    : "prefill";
            collectTimeline(ctx, is_decode, input, start, stage_context);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

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

        // Forward pass wall-clock profiler (enabled via LLAMINAR_PROFILING=1)
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

            if (is_decode)
                forward_pass_profiler_.recordDecodeIteration(timings);
            else
                forward_pass_profiler_.recordPrefillIteration(timings);
        }

        LOG_DEBUG("[ForwardExecutionEngine] Forward (cached) completed in "
                  << ms << "ms, success=" << success);

        return success;
    }

    // =========================================================================
    // executePrefillWithGraphCache() — Prefill Graph Capture/Replay State Machine
    // =========================================================================

    bool ForwardExecutionEngine::executePrefillWithGraphCache(
        const ForwardInput &input,
        ForwardGraphCache &forward_cache,
        IDeviceContext *ctx,
        IForwardExecutionHost &host)
    {
        // Initialize prefill graph cache on first use
        if (!forward_cache.prefill_graph_cache)
        {
            forward_cache.prefill_graph_cache = std::make_unique<PrefillGraphCache>(makePrefillGraphConfigFromEnv());
        }

        auto &cache = *forward_cache.prefill_graph_cache;

        PrefillGraphCacheKey key = makePrefillGraphKey(input, host);

        auto phase = cache.phase(key);
        const int real_seq_len = effectiveRealSeqLen(input);
        const int bucket_seq_len = effectiveBucketSeqLen(input);
        const bool padded_bucket = isPaddedBucketExecution(input);
        const bool snapshots_active = (executor_.config().snapshot_callback != nullptr);
        const bool moe_rebalancing_active = host.isMoeRebalancingActive();
        bool padded_preflight_checked = false;
        PrefillGraphRejectReason padded_preflight_reason = PrefillGraphRejectReason::None;

        auto launchPrefillGraph = [&](IWorkerGPUContext *gpu_ctx,
                                      void *stream,
                                      const char *capture_phase,
                                      PrefillGraphPhase cache_phase,
                                      const char *launch_kind,
                                      const std::string &recapture_reason) -> bool
        {
            void *gpu_start_event = nullptr;
            void *gpu_stop_event = nullptr;
            const bool should_time_gpu =
                PerfStatsCollector::gpuStageEventTimingEnabled() &&
                gpu_ctx != nullptr &&
                stream != nullptr;
            bool gpu_event_timing_active = false;
            if (should_time_gpu)
            {
                gpu_start_event = gpu_ctx->createEvent();
                gpu_stop_event = gpu_ctx->createEvent();
                if (gpu_start_event && gpu_stop_event)
                {
                    gpu_ctx->recordEvent(gpu_start_event, stream);
                    gpu_event_timing_active = true;
                }
            }

            const auto launch_t0 = std::chrono::high_resolution_clock::now();
            const bool ok = cache.launch(key);
            const auto launch_t1 = std::chrono::high_resolution_clock::now();
            const auto ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count());

            auto destroy_gpu_events = [&]()
            {
                if (gpu_ctx)
                {
                    if (gpu_start_event)
                        gpu_ctx->destroyEvent(gpu_start_event);
                    if (gpu_stop_event)
                        gpu_ctx->destroyEvent(gpu_stop_event);
                }
                gpu_start_event = nullptr;
                gpu_stop_event = nullptr;
            };

            if (KernelProfiler::isEnabled())
            {
                ForwardPassProfiler::addReplayLaunchNs(ns);
            }

            if (PerfStatsCollector::isEnabled())
            {
                auto observation = makePrefillGraphObservation(
                    input,
                    key,
                    capture_phase,
                    recapture_reason);
                auto tags = prefillGraphObservationTags(
                    observation,
                    prefillGraphPhaseName(cache_phase));
                tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                PerfStatsCollector::recordTimingNs(
                    "forward_graph",
                    "prefill_graph_launch",
                    ns,
                    "prefill",
                    input.device.toString(),
                    std::move(tags));
            }

            if (gpu_event_timing_active && ok)
            {
                gpu_ctx->recordEvent(gpu_stop_event, stream);
                gpu_ctx->synchronizeEvent(gpu_stop_event);
                const float elapsed_ms = gpu_ctx->eventElapsedTime(gpu_start_event, gpu_stop_event);
                if (elapsed_ms >= 0.0f)
                {
                    auto observation = makePrefillGraphObservation(
                        input,
                        key,
                        capture_phase,
                        recapture_reason);
                    auto tags = prefillGraphObservationTags(
                        observation,
                        prefillGraphPhaseName(cache_phase));
                    tags.emplace("attribution", "gpu_event");
                    tags.emplace("source", "prefill_graph_cache");
                    tags.emplace("graph_capture_scope", "prefill_graph_replay");
                    tags.emplace("timing_scope", "total_replay_gpu_event");
                    tags.emplace("sync_scope", "profiling_event_synchronized");
                    tags.emplace("launch_kind", launch_kind ? launch_kind : "unknown");
                    PerfStatsCollector::recordTimingNs(
                        "stage_gpu",
                        "prefill_graph.replay",
                        static_cast<uint64_t>(static_cast<double>(elapsed_ms) * 1.0e6),
                        "prefill",
                        input.device.toString(),
                        std::move(tags));
                }
            }
            destroy_gpu_events();

            return ok;
        };

        if (padded_bucket)
        {
            padded_preflight_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active);
            padded_preflight_checked = true;

            if (padded_preflight_reason != PrefillGraphRejectReason::None)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight: "
                          << toString(padded_preflight_reason)
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
        }

        auto gpuContextForPrefill = [&]() -> IWorkerGPUContext *
        {
            if (!ctx || !ctx->deviceId().is_gpu())
                return nullptr;
            auto &pool = GPUDeviceContextPool::instance();
            return &pool.getContext(ctx->deviceId());
        };

        auto bindPrefillStreamToStages = [&](void *stream)
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(stream);
            }
        };

        auto ensurePrefillCaptureStream = [&]() -> std::pair<IWorkerGPUContext *, void *>
        {
            IWorkerGPUContext *gpu_ctx = gpuContextForPrefill();
            if (!gpu_ctx)
                return {nullptr, nullptr};
            if (!forward_cache.prefill_capture_stream.ensure(gpu_ctx))
                return {gpu_ctx, nullptr};
            return {gpu_ctx, forward_cache.prefill_capture_stream.stream};
        };

        if (phase == PrefillGraphPhase::Ready)
        {
            // === REPLAY PATH ===
            // Dynamic params already updated by executeCacheHit caller.
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay missing explicit capture stream");
                return false;
            }
            bindPrefillStreamToStages(stream);

            if (!launchPrefillGraph(gpu_ctx, stream, "replay", PrefillGraphPhase::Ready, "replay", "none"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay FAILED for seq_len=" << input.seq_len);
                return false;
            }

            // Post-replay callbacks (KV cache head advance, histogram boundaries)
            for (auto *stage : forward_cache.replay_callback_stages)
                stage->onGraphReplayed();

            if (cache.config().trace)
                LOG_INFO("[ForwardExecutionEngine] Prefill graph REPLAY seq_len=" << input.seq_len
                                                                                  << " replay_count=" << cache.replayCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "replay",
                "none");
            return true;
        }

        if (phase == PrefillGraphPhase::Warmup)
        {
            // === CAPTURE PATH ===
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture missing explicit capture stream");
                return false;
            }

            // Apply the dedicated prefill stream to all stages and drain any
            // previous warmup work before beginning stream capture.
            bindPrefillStreamToStages(stream);
            gpu_ctx->synchronizeStream(stream);
            gpu_ctx->clearLastError();

            if (!cache.beginCapture(key, gpu_ctx, stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture BEGIN failed for seq_len=" << input.seq_len);
                return false;
            }

            // Execute stages into the capture stream with GraphCaptureGuard active.
            // This prevents timeline events and coherence events from being
            // recorded on the capture stream (they become graph nodes, not
            // real synchronizable events).
            bool exec_success;
            {
                GraphCaptureGuard capture_guard(/*host_bookkeeping=*/true);
                exec_success = executor_.executeFastDecode(
                    *forward_cache.graph, ctx, &forward_cache.collective_nodes);
            }

            if (!exec_success)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture EXECUTION failed for seq_len=" << input.seq_len);
                return false;
            }

            if (!cache.endCaptureAndInstantiate(key))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture END/INSTANTIATE failed for seq_len=" << input.seq_len);
                return false;
            }

            // Kernels recorded during HIP/CUDA stream capture are not executed
            // until the executable graph is launched. Launch once immediately so
            // the capture request produces logits and advances device state.
            if (!launchPrefillGraph(gpu_ctx, stream, "capture", PrefillGraphPhase::Ready, "launch_after_capture",
                                    phase == PrefillGraphPhase::Warmup ? "armed_warmup" : "recapture"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph launch-after-capture failed for seq_len=" << input.seq_len);
                return false;
            }

            // KV append stages advanced host metadata while recording so later
            // captured stages could read the just-appended cache view. The
            // immediate launch-after-capture must not advance those entries
            // again; normal Ready-phase replay still runs every callback.
            for (auto *stage : forward_cache.replay_callback_stages)
            {
                if (stage && stage->type() != ComputeStageType::KV_CACHE_APPEND)
                    stage->onGraphReplayed();
            }

            LOG_INFO("[ForwardExecutionEngine] Prefill graph CAPTURED seq_len=" << input.seq_len
                                                                                << " nodes=" << cache.nodeCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "capture",
                phase == PrefillGraphPhase::Warmup ? "armed_warmup" : "recapture");
            return true;
        }

        // === WARMUP/COLD PATH ===
        bool cold_capture_candidate = false;
        PrefillGraphRejectReason cold_reject_reason = PrefillGraphRejectReason::None;
        if (phase == PrefillGraphPhase::Cold)
        {
            cold_reject_reason = padded_preflight_checked
                                     ? padded_preflight_reason
                                     : preflightPrefillGraph(
                                           cache,
                                           *forward_cache.graph,
                                           key,
                                           forward_cache.collective_nodes,
                                           input,
                                           snapshots_active,
                                           moe_rebalancing_active);
            cold_capture_candidate = (cold_reject_reason == PrefillGraphRejectReason::None);
        }

        bool cold_stream_ready = false;
        if (cold_capture_candidate)
        {
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (gpu_ctx && stream)
            {
                // Warm up lazy allocations on the same explicit stream that the
                // next request will capture on. This mirrors decode segmented
                // capture and avoids capture-unsafe first-use work.
                bindPrefillStreamToStages(stream);
                cold_stream_ready = true;
            }
            else
            {
                cold_reject_reason = PrefillGraphRejectReason::NoGPUContext;
                cold_capture_candidate = false;
                if (padded_bucket)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight: "
                              << toString(cold_reject_reason)
                              << " real_seq_len=" << real_seq_len
                              << " bucket_seq_len=" << bucket_seq_len);
                    return false;
                }
            }
        }

        // Execute normally to warm up lazy allocations.
        bool exec_success = executor_.executeFastDecode(
            *forward_cache.graph, ctx, &forward_cache.collective_nodes);

        if (!exec_success)
            return false;

        // After successful warmup, check if graph capture is eligible
        if (phase == PrefillGraphPhase::Cold)
        {
            if (cold_capture_candidate && cold_stream_ready)
            {
                cache.markWarmedUp(key);
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Warmup,
                    "warmup",
                    "none");
                if (cache.config().trace)
                    LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED for capture: seq_len=" << input.seq_len);
            }
            else
            {
                if (cache.config().trace)
                {
                    LOG_INFO("[ForwardExecutionEngine] Prefill graph capture rejected: "
                             << toString(cold_reject_reason) << " seq_len=" << input.seq_len);
                }
                publishPrefillGraphObservation(
                    forward_cache,
                    input,
                    key,
                    PrefillGraphPhase::Cold,
                    "rejected",
                    toString(cold_reject_reason));
            }
            // Rejection is NOT fatal — we just won't use graph capture for this seq_len.
        }

        return true;
    }

    // =========================================================================
    // executeCacheMiss() — Build and Execute New Forward Graph
    // =========================================================================

    bool ForwardExecutionEngine::executeCacheMiss(
        const ForwardInput &input_in,
        ForwardOutput &output,
        const ForwardGraphSignature &signature,
        ForwardGraphCache *build_cache,
        bool should_cache,
        IForwardExecutionHost &host,
        bool is_decode,
        bool has_unified_pp,
        std::chrono::high_resolution_clock::time_point start)
    {
        // ===== CACHE MISS: Build new graph =====

        // Unified PP path currently executes multi-device graphs and does not use
        // this forward cache; clear entries to avoid stale memory growth.
        if (has_unified_pp && !cache_.empty())
        {
            invalidateAll();
            cache_.clear();
            LOG_DEBUG("[ForwardExecutionEngine] Cleared forward graph cache for unified PP execution path");
        }

        // For cache misses on standard path: redirect token_ids and
        // position_ids to stable buffers so that cached stages' pointers survive.
        ForwardInput effective_input = input_in;

        if (should_cache && build_cache)
        {
            const int total_tokens = effective_input.batch_size * effective_input.seq_len;
            if (effective_input.token_ids)
            {
                build_cache->token_ids.assign(
                    effective_input.token_ids,
                    effective_input.token_ids + total_tokens);
                effective_input.token_ids = build_cache->token_ids.data();
            }
            if (effective_input.position_ids)
            {
                build_cache->position_ids.assign(
                    effective_input.position_ids,
                    effective_input.position_ids + total_tokens);
                effective_input.position_ids = build_cache->position_ids.data();
            }
        }

        // Build forward graph via host callback
        GraphBuildResult build_result = host.buildForwardGraph(effective_input);

        if (!build_result)
        {
            LOG_ERROR("[ForwardExecutionEngine] Graph build failed: " << build_result.error());
            return false;
        }

        output = build_result.output();
        ComputeGraph graph = build_result.takeGraph();
        auto collective_nodes = collectCollectiveNodeNames(graph);

        LOG_DEBUG("[ForwardExecutionEngine] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Empty forward graph");
            return false;
        }

        if (should_cache && build_cache && signature.is_bucketed_prefill && isPaddedBucketExecution(effective_input))
        {
            PrefillGraphCache preflight_cache(makePrefillGraphConfigFromEnv());
            PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);

            const PrefillGraphRejectReason reject_reason = preflightPrefillGraph(
                preflight_cache,
                graph,
                key,
                collective_nodes,
                effective_input,
                executor_.config().snapshot_callback != nullptr,
                host.isMoeRebalancingActive());

            if (reject_reason != PrefillGraphRejectReason::None)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight before execution: "
                          << toString(reject_reason)
                          << " real_seq_len=" << effectiveRealSeqLen(effective_input)
                          << " bucket_seq_len=" << effectiveBucketSeqLen(effective_input));
                return false;
            }
        }

        // Cache hits already refresh this small stage list before replay.
        // Cache misses must do the same before the first graph execution; otherwise
        // padded bucket stages can append/select the bucket tail during warmup.
        std::vector<IComputeStage *> prefill_replay_param_stages;
        if (!is_decode)
        {
            prefill_replay_param_stages = collectPrefillReplayParamStages(graph);
            updatePrefillReplayParamStages(effective_input, prefill_replay_param_stages);
        }

        // Ensure declared CPU/GPU workspace is allocated for this graph. Cache
        // hits repeat this step because another bucket can grow the shared
        // per-device workspace and leave cached stages bound to old pointers.
        if (!host.ensureDeviceWorkspaceAllocated(graph, effective_input.seq_len))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to allocate workspace for forward graph on "
                      << effective_input.device.toString() << " seq_len=" << effective_input.seq_len);
            return false;
        }
        const uint64_t workspace_generation = host.workspaceGeneration(effective_input.device);

        // Notify host that graph is ready — allows releasing transient resources
        // (e.g., mmap pages) before execution allocates large activation buffers.
        if (!first_graph_ready_fired_)
        {
            first_graph_ready_fired_ = true;
            host.onFirstGraphReady();
        }

        bool success = false;

        // Execution path depends on configuration:
        // - Unified PP: multi-device execution with all PP stage devices
        // - Single-device: standard single-context execution
        if (has_unified_pp)
        {
            auto contexts = host.getPipelineDeviceContexts();
            if (contexts.empty())
            {
                LOG_ERROR("[ForwardExecutionEngine] No pipeline device contexts available");
                return false;
            }

            LOG_DEBUG("[ForwardExecutionEngine] Executing unified PP graph with "
                      << contexts.size() << " device contexts...");

            success = executor_.executeMultiDevice(graph, contexts);
        }
        else
        {
            LOG_DEBUG("[ForwardExecutionEngine] Getting device context for " << effective_input.device << "...");
            IDeviceContext *ctx = host.getDeviceContext(effective_input.device);
            if (!ctx)
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to get device context");
                return false;
            }
            LOG_DEBUG("[ForwardExecutionEngine] Got device context, starting execution...");

            if (host.computeAllPositionLogitsEnabled() &&
                host.allPositionLogitRows() > 0)
            {
                void *metadata_stream = nullptr;
                if (ctx->deviceId().is_gpu())
                {
                    try
                    {
                        metadata_stream = GPUDeviceContextPool::instance()
                                              .getContext(ctx->deviceId())
                                              .defaultStream();
                    }
                    catch (const std::exception &e)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Could not resolve verifier metadata stream for "
                                  << ctx->deviceId().toString() << ": " << e.what());
                        return false;
                    }
                }

                if (!host.prepareAllPositionVerifierGraphMetadata(
                        effective_input,
                        metadata_stream,
                        ctx->deviceId()))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to prepare all-position verifier graph metadata");
                    return false;
                }
            }

            success = executor_.execute(graph, ctx);
        }

        // Sync the stream at the forward pass boundary (same as cached path above)
        if (success)
        {
            IDeviceContext *sync_ctx = host.getDeviceContext(effective_input.device);
            if (sync_ctx)
            {
                host.syncLogitsAtBoundary(sync_ctx);
            }

            const std::string stage_context =
                should_cache
                    ? forwardGraphPerfContext(signature)
                    : (is_decode ? "main_decode" : "prefill");
            collectTimeline(
                host.getDeviceContext(effective_input.device),
                is_decode, effective_input, start, stage_context);
        }

        // Cache the graph for future matching forward signatures
        if (should_cache && build_cache && success)
        {
            build_cache->graph = std::make_unique<ComputeGraph>(std::move(graph));
            build_cache->output = output;
            build_cache->workspace_generation = workspace_generation;

            // Pre-compute collective node set for fast decode intercept and
            // padded-prefill safety checks on later same-bucket cache hits.
            build_cache->collective_nodes = std::move(collective_nodes);

            build_cache->prefill_replay_param_stages = std::move(prefill_replay_param_stages);
            build_cache->prefill_replay_param_stages_cached = !is_decode;

            build_cache->valid = true;
            touchBucketedPrefillForwardCache(signature, *build_cache);

            // Store PP hidden state copy info for cache HIT replay
            auto pp_copy = host.resolvePPCopyInfo(effective_input);
            if (pp_copy.needs_copy)
            {
                build_cache->pp_external_hidden_state = pp_copy.external_hidden;
                build_cache->pp_working_buffer = pp_copy.working_buffer;
                build_cache->pp_copy_bytes = pp_copy.copy_bytes;
                build_cache->pp_device = pp_copy.device;
                build_cache->pp_needs_copy = true;

                LOG_DEBUG("[ForwardExecutionEngine] Stored PP copy info: "
                          << pp_copy.copy_bytes << " bytes on " << pp_copy.device.toString());
            }

            LOG_DEBUG("[ForwardExecutionEngine] Cached forward graph for signature "
                      << "[seq_len=" << signature.seq_len
                      << ", batch_size=" << signature.batch_size
                      << ", device=" << signature.device.to_string()
                      << ", decode=" << signature.decode
                      << ", decode_has_history=" << signature.decode_has_history
                      << "] (" << build_cache->graph->size() << " stages)");

            if (signature.is_bucketed_prefill)
                enforceBucketedPrefillForwardCapacity(&signature);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        LOG_DEBUG("[ForwardExecutionEngine] Forward completed in " << ms << "ms, success=" << success);

        return success;
    }

    // =========================================================================
    // collectTimeline() — GPU Stage Timing
    // =========================================================================

    void ForwardExecutionEngine::collectTimeline(
        IDeviceContext *ctx,
        bool is_decode,
        const ForwardInput &input,
        std::chrono::high_resolution_clock::time_point start,
        std::string stage_context)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
            !ctx || !ctx->deviceId().is_gpu())
        {
            return;
        }

        auto &timeline = executor_.stageTimeline();
        if (!timeline.isInitialized())
        {
            return;
        }

        // When suppressed (warmup), discard any stale events but don't
        // collect or accumulate. This prevents stale events from the Cold
        // phase persisting after graph capture invalidates the stream state.
        if (suppress_timeline_)
        {
            timeline.resetTimings();
            return;
        }

        auto &pool = GPUDeviceContextPool::instance();
        IWorkerGPUContext &gpu_ctx = pool.getContext(ctx->deviceId());
        timeline.collect(&gpu_ctx);

        double wall_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::high_resolution_clock::now() - start)
                             .count();
        std::string dev_str = ctx->deviceId().toString();
        const char *dev_name = dev_str.c_str();
        const char *phase_name = is_decode ? "decode" : "prefill";
        PerfStatsCollector::Tags stage_tags;
        if (!stage_context.empty())
            stage_tags.emplace("context", std::move(stage_context));
        timeline.recordPerfStats(phase_name, dev_name, "stage_gpu", std::move(stage_tags));

        if (is_decode)
        {
            // Accumulate decode iterations — print once via flushStageTimeline()
            timeline.accumulateIteration(wall_ms);
        }
        else if (accumulate_prefill_)
        {
            // Accumulate prefill iterations — print once via flushStageTimeline() (benchmark mode)
            int tokens = input.batch_size * input.seq_len;
            timeline.accumulatePrefillIteration(wall_ms, tokens);
        }
        else
        {
            // Flush any pending decode data before printing prefill
            timeline.printAccumulatedSummary("DECODE", dev_name);

            int tokens = input.batch_size * input.seq_len;
            timeline.printSummary("PREFILL", tokens, wall_ms, dev_name);
            if (debugEnv().gpu_stage_timing_detail)
                timeline.printDetailedTimeline("PREFILL", dev_name);
        }
        timeline.resetTimings();
    }

    void ForwardExecutionEngine::forEachCachedStage(
        ComputeStageType type,
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage && node->stage->type() == type)
                    visitor(node->stage.get());
            }
        }
    }

    std::optional<ForwardExecutionEngine::PrefillGraphCacheSnapshot> ForwardExecutionEngine::prefillGraphCacheSnapshot(
        const ForwardGraphSignature &signature,
        const PrefillGraphCacheKey &key) const
    {
        auto it = cache_.find(signature);
        if (it == cache_.end())
            return std::nullopt;

        PrefillGraphCacheSnapshot snapshot;
        const ForwardGraphCache &forward_cache = it->second;
        snapshot.forward_cache_valid = forward_cache.valid;
        snapshot.eviction_count = bucketed_prefill_forward_eviction_count_;
        snapshot.bucket_seq_len = key.seq_len;
        snapshot.domain_id = key.domain_id;
        snapshot.participant_id = key.participant_id;
        snapshot.placement_epoch = key.placement_epoch;
        snapshot.topology_signature = key.topology_signature;
        snapshot.capture_phase = "unknown";
        snapshot.recapture_reason = "none";

        if (forward_cache.last_prefill_graph_observation.valid)
        {
            const auto &observation = forward_cache.last_prefill_graph_observation;
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
        }

        if (!forward_cache.prefill_graph_cache)
            return snapshot;

        const PrefillGraphCache &prefill_cache = *forward_cache.prefill_graph_cache;
        snapshot.prefill_cache_initialized = true;
        snapshot.phase = prefill_cache.phase(key);
        snapshot.cache_size = prefill_cache.size();
        snapshot.node_count = prefill_cache.nodeCount(key);
        snapshot.replay_count = prefill_cache.replayCount(key);
        snapshot.warmup_count = prefill_cache.warmupCount(key);
        snapshot.capture_count = prefill_cache.captureCount(key);
        snapshot.eviction_count += prefill_cache.evictionCount();
        return snapshot;
    }

} // namespace llaminar2
