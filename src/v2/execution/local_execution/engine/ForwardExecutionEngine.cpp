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
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
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

        /**
         * @brief Publish the ordering owner for one successful forward call.
         *
         * This helper deliberately writes the caller-owned ForwardOutput. The
         * execution-scoped record is the unambiguous handoff token for an
         * immediate consumer such as shifted-MTP prefill cache construction.
         * Longer-lived consumers use the orchestrator-owned durable timeline
         * event published before this producer stream can be invalidated.
         */
        void publishForwardOutputProvenance(
            ForwardOutput &output,
            DeviceId device,
            void *stream,
            bool is_decode,
            bool all_position_logits,
            int graph_seq_len,
            int graph_batch_size)
        {
            output.execution = {
                .valid = !device.is_gpu() || stream != nullptr,
                .device = device,
                .stream = stream,
                .is_decode = is_decode,
                .all_position_logits = all_position_logits,
                .graph_seq_len = graph_seq_len,
                .graph_batch_size = graph_batch_size,
            };
        }

        /**
         * @brief Resolve the device that actually produced a forward output.
         *
         * `TensorBase::home_device()` is immutable creation metadata. Activation
         * storage can be created on the host and subsequently become
         * device-authoritative, so using the home device here loses the exact
         * GPU stream/event boundary and permits stale host reads. Prefer the
         * tensor's coherence owner, then its live GPU allocation, and retain
         * the graph execution device when neither supplies stronger evidence.
         *
         * @param output Successful forward result whose hidden/logits storage
         *        carries the latest coherence state.
         * @param execution_device Device on which the graph was executed.
         * @return Runtime producer device suitable for stream provenance.
         */
        DeviceId resolveForwardOutputProducerDevice(
            const ForwardOutput &output,
            DeviceId execution_device)
        {
            TensorBase *tensor = output.hidden ? output.hidden : output.logits;
            if (!tensor)
                return execution_device;

            if (const auto authoritative = tensor->getAuthoritativeDevice();
                authoritative.has_value() && authoritative->is_valid())
            {
                return *authoritative;
            }
            if (const auto current = tensor->current_device();
                current.has_value() && current->is_valid())
            {
                return *current;
            }

            const DeviceId home = tensor->home_device();
            return home.is_gpu() ? home : execution_device;
        }

        std::mutex &gpuCacheMissGraphMaterializationMutex()
        {
            static std::mutex mutex;
            return mutex;
        }

        /**
         * @brief Install a host-owned worker resolver for one engine call.
         *
         * DeviceGraphExecutor also supports standalone use, where it resolves
         * the process GPU pool directly. ForwardExecutionEngine has a stronger
         * host boundary, so every nested eager/captured executor operation must
         * use the same host-selected worker. Restoring the prior resolver on all
         * returns prevents a short-lived test host from remaining captured by a
         * long-lived executor.
         */
        class ScopedWorkerGPUContextResolver
        {
        public:
            ScopedWorkerGPUContextResolver(
                DeviceGraphExecutor &executor,
                IForwardExecutionHost &host,
                DeviceId device)
                : executor_(executor),
                  previous_resolver_(executor.config().worker_gpu_context_resolver),
                  previous_uses_process_pool_(
                      executor.config().worker_gpu_context_uses_process_pool)
            {
                executor_.setWorkerGPUContextResolver(
                    [&host](DeviceId requested_device)
                    {
                        return host.getWorkerGPUContext(requested_device);
                    },
                    host.workerGPUContextUsesProcessPool(device));
            }

            ~ScopedWorkerGPUContextResolver()
            {
                executor_.setWorkerGPUContextResolver(
                    std::move(previous_resolver_),
                    previous_uses_process_pool_);
            }

            ScopedWorkerGPUContextResolver(const ScopedWorkerGPUContextResolver &) = delete;
            ScopedWorkerGPUContextResolver &operator=(const ScopedWorkerGPUContextResolver &) = delete;

        private:
            DeviceGraphExecutor &executor_;
            std::function<IWorkerGPUContext *(DeviceId)> previous_resolver_;
            bool previous_uses_process_pool_ = false;
        };

        /// @brief Return the absolute logical-token offset for prefill-style inputs.
        ///
        /// `token_offset` is the first-class request-range owner for prepared
        /// prefill chunks. Raw server/orchestrator inputs historically carried
        /// only `position_offset`; treating that as the logical-token offset
        /// keeps restored-prefix suffix prefill aligned with RoPE, KV append,
        /// and replay metadata until every caller is migrated to an explicit
        /// non-optional request range.
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
            chunk_input.position_policy = plan.position_policy;
            if (plan.position_policy == ForwardPositionPolicy::ContiguousOffset)
            {
                // The absolute range is represented by position_offset. Keeping
                // both pointers null prevents graph construction or replay from
                // recording a mutable host-row upload for this device-owned path.
                chunk_input.position_ids = nullptr;
                chunk_input.position_ids_device = nullptr;
            }
            else
            {
                chunk_input.position_ids = plan.chunk.position_ids.data();
                chunk_input.position_ids_device = nullptr;
            }
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
                effectiveTokenOffset(input)};
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
            case PrefillGraphPhase::Initialized:
                return "initialized";
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
            const std::string &recapture_reason,
            const std::string &reject_stage_name = std::string(),
            const std::string &reject_stage_type = std::string())
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
            observation.reject_stage_name = reject_stage_name;
            observation.reject_stage_type = reject_stage_type;
            return observation;
        }

        PerfStatsCollector::Tags prefillGraphObservationTags(
            const PrefillGraphExecutionObservation &observation,
            const char *cache_phase)
        {
            auto tags = PerfStatsCollector::Tags{
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
            if (!observation.reject_stage_name.empty())
                tags.emplace("reject_stage_name", observation.reject_stage_name);
            if (!observation.reject_stage_type.empty())
                tags.emplace("reject_stage_type", observation.reject_stage_type);
            return tags;
        }

        void publishPrefillGraphObservation(
            ForwardGraphCache &forward_cache,
            const ForwardInput &input,
            const PrefillGraphCacheKey &key,
            PrefillGraphPhase cache_phase,
            const char *capture_phase,
            const std::string &recapture_reason,
            const std::string &reject_stage_name = std::string(),
            const std::string &reject_stage_type = std::string())
        {
            auto observation = makePrefillGraphObservation(
                input,
                key,
                capture_phase,
                recapture_reason,
                reject_stage_name,
                reject_stage_type);
            forward_cache.last_prefill_graph_observation = observation;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_graph_lifecycle",
                1.0,
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
            PerfStatsCollector::addCounter(
                "forward_graph",
                "prefill_graph_phase",
                1.0,
                "prefill",
                input.device.toString(),
                prefillGraphObservationTags(observation, prefillGraphPhaseName(cache_phase)));
        }

        std::string forwardGraphPerfContext(const ForwardGraphSignature &signature)
        {
            if (!signature.decode)
                return signature.is_bucketed_prefill ? "prefill_bucket" : "prefill";
            if (signature.live_mtp_request_batch_condition)
                return "main_condition_batch";
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
                {"live_mtp_request_batch_condition",
                 boolTag(signature.live_mtp_request_batch_condition)},
                {"all_position_logit_rows", std::to_string(signature.all_position_logit_rows)},
                {"uses_device_token_ids", boolTag(signature.uses_device_token_ids)},
                {"uses_device_position_ids", boolTag(signature.uses_device_position_ids)},
                {"uses_device_sequence_lengths", boolTag(signature.uses_device_sequence_lengths)},
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
            bool moe_rebalancing_active,
            PrefillGraphPreflightMode mode = PrefillGraphPreflightMode::Default,
            bool collectives_graph_capturable = false,
            bool moe_rebalancing_graph_stable = false,
            bool host_policy_disabled = false,
            std::string *reject_stage_name = nullptr,
            std::string *reject_stage_type = nullptr)
        {
            if (reject_stage_name)
                reject_stage_name->clear();
            if (reject_stage_type)
                reject_stage_type->clear();
            if (host_policy_disabled)
                return PrefillGraphRejectReason::HostPolicyDisabled;
            return cache.preflight(
                graph,
                key,
                &collective_nodes,
                snapshots_active,
                moe_rebalancing_active,
                effectiveRealSeqLen(input),
                effectiveBucketSeqLen(input),
                mode,
                collectives_graph_capturable,
                moe_rebalancing_graph_stable,
                reject_stage_name,
                reject_stage_type);
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
            if (lhs.live_mtp_request_batch_condition !=
                rhs.live_mtp_request_batch_condition)
            {
                return lhs.live_mtp_request_batch_condition <
                       rhs.live_mtp_request_batch_condition;
            }
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

        const int token_offset = effectiveTokenOffset(input);

        plan.padding_required = !plan.selection.exact;
        plan.position_policy = input.device.is_gpu()
                                   ? ForwardPositionPolicy::ContiguousOffset
                                   : ForwardPositionPolicy::ExplicitRows;
        plan.chunk.token_offset = token_offset;
        plan.chunk.real_count = real_seq_len;
        plan.chunk.bucket_seq_len = plan.selection.bucket_seq_len;
        plan.chunk.token_ids = padPrefillTokensToBucket(
            input.token_ids,
            real_seq_len,
            plan.selection.bucket_seq_len,
            pad_token_id);
        if (plan.position_policy == ForwardPositionPolicy::ExplicitRows)
        {
            plan.chunk.position_ids = buildPrefillChunkPositionIds(
                real_seq_len,
                plan.selection.bucket_seq_len,
                token_offset,
                input.batch_size);
        }

        if (plan.chunk.token_ids.empty() ||
            (plan.position_policy == ForwardPositionPolicy::ExplicitRows &&
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

    size_t ForwardExecutionEngine::invalidateMoEPlacementSensitiveGraphsForStablePlacement()
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
        bool preserve_replay_safe_segmented_captures)
    {
        ReplayStateResetSummary summary;
        all_position_verifier_recapture_pending_ = false;
        for (auto &[signature, cache] : cache_)
        {
            const ForwardReplayStateCacheClass cache_class =
                classifyForwardReplayStateCache(signature);
            const ForwardReplayStateAction action =
                preserve_replay_safe_segmented_captures
                    ? chooseForwardReplayStateAction(
                          ForwardReplayStateMutationKind::RequestBoundaryStateReset,
                          signature,
                          !cache.collective_nodes.empty())
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
        ScopedWorkerGPUContextResolver worker_resolver_scope(
            executor_,
            host,
            input.device);
        last_executed_forward_graph_ = {};

        auto start = std::chrono::high_resolution_clock::now();

        const int first_position = input.position_ids ? input.position_ids[0] : input.position_offset;
        // Classify the execution path. Greedy MTP verifies short multi-token
        // continuations against an existing KV/GDN history; those must use the
        // decode graph path rather than the prompt-prefill path.
        const int decode_max_seq_len = std::max(1, config_.cache_config.decode_seq_len);
        const bool all_position_logits = host.computeAllPositionLogitsEnabled();
        const bool live_mtp_request_batch_condition =
            host.liveMTPRequestBatchConditionEnabled();
        if (all_position_logits)
        {
            /*
             * A new verifier attempt owns the publication handle. Clear any
             * earlier row-snapshot producer before execution so a failed verifier
             * cannot leave stale snapshots publishable.
             */
            clearLastAllPositionVerifierForwardGraph();
        }
        const bool mtp_spec_verifier_decode =
            all_position_logits &&
            host.mtpSpecVerifierInputPlanActive() &&
            input.seq_len > 0 &&
            input.seq_len <= decode_max_seq_len;
        const bool is_single_token_decode = (input.seq_len == 1 && input.batch_size <= 1);
        const bool is_short_continuation_decode =
            input.batch_size <= 1 &&
            input.seq_len > 1 &&
            input.seq_len <= decode_max_seq_len &&
            first_position > 0;
        const bool is_decode =
            is_single_token_decode ||
            is_short_continuation_decode ||
            mtp_spec_verifier_decode ||
            live_mtp_request_batch_condition;
        const bool decode_has_history = is_decode && first_position > 0;
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
                .seq_len = bucketed_prefill
                               ? bucketed_prefill_seq_len
                               : effective_input.seq_len,
                .batch_size = effective_input.batch_size,
                .device = effective_input.device,
                .decode = is_decode,
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
                .uses_device_token_ids = input.token_ids_device != nullptr,
                .uses_device_position_ids = input.position_ids_device != nullptr,
                .position_policy = effective_input.position_policy,
                .uses_device_sequence_lengths =
                    input.sequence_lengths_device != nullptr,
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
                .moe_placement_epoch = host.moePlacementEpoch()};

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
            {
                recordLastExecutedForwardGraph(forward_signature, /*cache_hit=*/true);
            }
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
        if (success && !output.execution.valid)
        {
            LOG_ERROR("[ForwardExecutionEngine] Successful forward did not publish execution provenance for "
                      << effective_input.device.toString());
            return false;
        }
        return success;
    }

    void ForwardExecutionEngine::recordLastExecutedForwardGraph(
        const ForwardGraphSignature &signature,
        bool cache_hit)
    {
        last_executed_forward_graph_.valid = true;
        last_executed_forward_graph_.signature = signature;
        last_executed_forward_graph_.cache_hit = cache_hit;
        if (signature.decode && signature.all_position_logits)
        {
            last_all_position_verifier_graph_.valid = true;
            last_all_position_verifier_graph_.signature = signature;
            last_all_position_verifier_graph_.cache_hit = cache_hit;
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
        view.stream = it->second.outputProducerStream(
            state.signature.decode,
            /*used_graph_replay=*/state.signature.decode);
        view.cache_hit = state.cache_hit;
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
            snapshot.eviction_count = bucketed_prefill_forward_eviction_count_;
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

            if (cache.prefill_graph_cache)
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

        if (decode_capture_allowed && stream_ctx && stream_ctx->deviceId().is_gpu() && stream_gpu_ctx)
        {
            const auto early_capture_policy = host.buildDecodeCapturePolicy(
                !forward_cache.collective_nodes.empty(),
                stream_ctx);

            if (early_capture_policy.allow_cached_graph_replay)
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
                    LOG_ERROR("[ForwardExecutionEngine] Failed to create the mandatory explicit decode graph stream for "
                              << stream_ctx->deviceId().toString());
                    return false;
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

        if (!host.prepareLiveStateForForwardGraphExecution(
                input,
                dynamic_param_stream,
                stream_device))
        {
            LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for cached forward graph execution");
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
        const int *replay_position_ids =
            selectForwardReplayHostPositionIds(forward_cache, input);
        const int position_row_count = forwardPositionRowCount(input);
        if ((input.position_ids_device || replay_position_ids) &&
            position_row_count <= 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Cached graph replay received invalid flattened position geometry: batch="
                      << input.batch_size << " seq_len=" << input.seq_len);
            return false;
        }
        for (auto *stage : forward_cache.dynamic_param_stages)
        {
            if (input.position_ids_device)
            {
                if (!stage->supportsDeviceResidentDynamicPositionReplay())
                {
                    LOG_ERROR("[ForwardExecutionEngine] Stage "
                              << stage->name()
                              << " cannot consume device-resident replay positions without host scalar state");
                    return false;
                }
                stage->updateDynamicDevicePositionIds(
                    input.position_ids_device,
                    position_row_count);
                /*
                 * RoPE consumes device position rows directly.  Calling the
                 * scalar update afterward would clear its device pointer.
                 * Other dynamic stages use this hook as a resident-state marker
                 * and still need their non-position replay scalars stamped.
                 */
                if (stage->type() == ComputeStageType::ROPE)
                    continue;
                stage->updateDynamicParams(input.position_offset, input.seq_len);
            }
            else if (replay_position_ids)
            {
                stage->updateDynamicParams(input.position_offset, input.seq_len);
                stage->updateDynamicPositionIds(
                    replay_position_ids,
                    position_row_count);
            }
            else
            {
                stage->updateDynamicParams(input.position_offset, input.seq_len);
            }
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
            // Prefill with graph capture/replay state machine. Snapshot capture
            // is handled as a post-graph drain inside executePrefillWithGraphCache()
            // so parity diagnostics do not force eager stage execution.
            success = executePrefillWithGraphCache(input, forward_cache, ctx, host);
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
            if (capture_policy.collective_segmented_enabled)
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
                host.shouldDeferMainDecodeFinalSync();
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
            requested_deferred_all_position_sync =
                wants_all_position_sync_defer;
            requested_deferred_main_decode_sync =
                wants_main_decode_sync_defer ||
                wants_captured_collective_sync_defer;
            executed_deferred_all_position_sync =
                wants_all_position_sync_defer && capture_policy.defer_final_sync;
            executed_deferred_main_decode_sync =
                requested_deferred_main_decode_sync && capture_policy.defer_final_sync;

            PerfStatsCollector::addCounter(
                "forward_graph",
                "decode_capture_policy",
                1.0,
                "decode",
                input.device.toString(),
                {{"context", all_position_verifier ? "main_verifier" : "main_decode"},
                 {"allow_graph_replay", boolTag(capture_policy.allow_cached_graph_replay)},
                 {"defer_final_sync", boolTag(capture_policy.defer_final_sync)},
                 {"has_collectives", boolTag(has_collective_nodes)},
                 {"collective_segmented", boolTag(capture_policy.collective_segmented_enabled)},
                 {"collectives_graph_capturable", boolTag(capture_policy.collectives_graph_capturable)},
                 {"replay_plan_policy",
                  capture_policy.graph_replay_plan_policy ==
                          DeviceGraphExecutor::GraphReplayPlanPolicy::
                              RequireFullGraph
                      ? "require_full_graph"
                      : "allow_heterogeneous_collective_segmentation"}});

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
                    {{"context", all_position_verifier ? "main_verifier" : "main_decode"},
                     {"old_epoch", std::to_string(forward_cache.graph_replay_live_state_epoch)},
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
                &used_graph_replay);
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

        const bool all_position_verifier_sync_deferred =
            success &&
            used_graph_replay &&
            executed_deferred_all_position_sync &&
            forward_cache.segment_cache.capture_stream != nullptr;
        const bool main_decode_sync_deferred =
            success &&
            used_graph_replay &&
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
                forward_cache.outputProducerStream(
                    is_decode,
                    used_graph_replay),
                is_decode,
                host.computeAllPositionLogitsEnabled(),
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
        }

        // Publish logits at the forward ownership boundary. Device consumers
        // inherit the producer stream; host consumers wait only on the exact
        // tensor or replay completion event.
        if (success)
        {
            if (!all_position_verifier_sync_deferred &&
                !main_decode_sync_deferred)
            {
                /*
                 * The cached graph's ForwardOutput is immutable graph metadata:
                 * it names the exact local-shard, gathered, or replicated logits
                 * tensor selected when the graph was built. Publish that tensor
                 * after every warmup/capture/replay launch. A device consumer can
                 * instead inherit the producer stream through the deferred path
                 * above, so neither route needs a host or stream synchronization.
                 */
                if (!host.publishLogitsAtBoundary(
                        output.logits,
                        ctx,
                        output.execution.stream))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish cached forward logits at the device boundary");
                    return false;
                }
            }

            const std::string stage_context =
                is_decode
                    ? (forward_cache.segment_cache.perf_context.empty()
                           ? "main_decode"
                           : forward_cache.segment_cache.perf_context)
                    : "prefill";
            collectTimeline(host, ctx, is_decode, input, start, stage_context);
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
        const bool moe_rebalancing_graph_stable =
            host.isMoeRebalancingGraphStableForPrefillCapture();
        const bool host_prefill_graph_disabled = host.prefillGraphCaptureDisabledByHost();
        const bool prefill_collectives_graph_capturable =
            !forward_cache.collective_nodes.empty() &&
            host.buildDecodeCapturePolicy(
                    /*has_collective_nodes=*/true,
                    ctx)
                .collectives_graph_capturable;
        bool padded_preflight_checked = false;
        PrefillGraphRejectReason padded_preflight_reason = PrefillGraphRejectReason::None;
        std::string padded_reject_stage_name;
        std::string padded_reject_stage_type;

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
                moe_rebalancing_active,
                PrefillGraphPreflightMode::ColdPaddedSupport,
                prefill_collectives_graph_capturable,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &padded_reject_stage_name,
                &padded_reject_stage_type);
            padded_preflight_checked = true;

            if (padded_preflight_reason != PrefillGraphRejectReason::None)
            {
                LOG_ERROR("[ForwardExecutionEngine] Padded prefill graph rejected by preflight: "
                          << toString(padded_preflight_reason)
                          << (padded_reject_stage_name.empty() ? "" : " stage=")
                          << padded_reject_stage_name
                          << (padded_reject_stage_type.empty() ? "" : " type=")
                          << padded_reject_stage_type
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
        }

        auto gpuContextForPrefill = [&]() -> IWorkerGPUContext *
        {
            if (!ctx || !ctx->deviceId().is_gpu())
                return nullptr;
            return host.getWorkerGPUContext(ctx->deviceId());
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

        auto preparePrefillGraphLaunchMetadata = [&](void *stream,
                                                     const char *phase_name) -> bool
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (!node || !node->stage || !node->stage->needsGraphLaunchPreparation())
                    continue;

                /*
                 * Monolithic prefill graphs have the same mutable-metadata
                 * contract as segmented decode graphs: tiny device-side
                 * scalars such as row-select indices must be uploaded on the
                 * explicit capture/replay stream before the graph is recorded
                 * or launched. The stage hook owns that upload; recording an
                 * H2D inside capture is a correctness bug because future graph
                 * replays would bake the first request's metadata.
                 */
                if (!node->stage->prepareGraphLaunch(ctx, stream))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Prefill graph "
                              << (phase_name ? phase_name : "launch")
                              << " metadata preparation failed for stage '"
                              << node_name << "'");
                    return false;
                }
            }
            return true;
        };

        auto publishPrefillCapturedTerminalState = [&](void *stream,
                                                       const char *context) -> bool
        {
            const int terminal_row = real_seq_len - 1;
            if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                    *forward_cache.graph,
                    terminal_row,
                    stream,
                    context,
                    input.sequence_lengths_device,
                    input.batch_size,
                    input.seq_len,
                    input.sequence_lengths))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph terminal state publication failed"
                          << (context ? std::string(" during ") + context : std::string{})
                          << " real_seq_len=" << real_seq_len
                          << " bucket_seq_len=" << bucket_seq_len);
                return false;
            }
            return true;
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

        if (phase == PrefillGraphPhase::Ready && !host_prefill_graph_disabled)
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

            if (!preparePrefillGraphLaunchMetadata(stream, "replay"))
                return false;

            if (!launchPrefillGraph(gpu_ctx, stream, "replay", PrefillGraphPhase::Ready, "replay", "none"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph replay FAILED for seq_len=" << input.seq_len);
                return false;
            }

            // Post-replay callbacks (KV cache head advance, histogram boundaries)
            for (auto *stage : forward_cache.replay_callback_stages)
                stage->onGraphReplayed();

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_replay"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_replay"))
            {
                return false;
            }

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

        const bool can_attempt_capture =
            phase == PrefillGraphPhase::Warmup ||
            phase == PrefillGraphPhase::Initialized;
        PrefillGraphRejectReason capture_ready_reason = PrefillGraphRejectReason::None;
        std::string capture_ready_reject_stage_name;
        std::string capture_ready_reject_stage_type;
        if (can_attempt_capture)
        {
            capture_ready_reason = preflightPrefillGraph(
                cache,
                *forward_cache.graph,
                key,
                forward_cache.collective_nodes,
                input,
                snapshots_active,
                moe_rebalancing_active,
                PrefillGraphPreflightMode::CaptureReady,
                prefill_collectives_graph_capturable,
                moe_rebalancing_graph_stable,
                host_prefill_graph_disabled,
                &capture_ready_reject_stage_name,
                &capture_ready_reject_stage_type);
        }

        if (can_attempt_capture && capture_ready_reason == PrefillGraphRejectReason::None)
        {
            // === CAPTURE PATH ===
            auto [gpu_ctx, stream] = ensurePrefillCaptureStream();
            if (!gpu_ctx || !stream)
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture missing explicit capture stream");
                return false;
            }

            /*
             * Apply the dedicated prefill stream to every stage before capture.
             * The warmup, metadata preparation, snapshot preparation, and
             * captured launch all use this same explicit stream. CUDA/HIP
             * stream order therefore supplies the complete producer-to-capture
             * dependency; a host stream drain here would only serialize the
             * Warmup -> Capture transition.
             */
            bindPrefillStreamToStages(stream);
            if (!preparePrefillGraphLaunchMetadata(stream, "capture"))
                return false;
            if (!executor_.prepareSnapshotsForGraphCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph snapshot preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            /*
             * Warmup may publish arena inputs from eager streams other than this
             * graph's dedicated stream. Join every such producer here, while
             * external event waits are legal, so the capture body contains only
             * graph-owned ordering. This is a nonblocking device-side fence; it
             * performs no stream or device synchronization.
             */
            if (!executor_.prepareInputsForGraphCapture(
                    *forward_cache.graph,
                    ctx,
                    stream,
                    "prefill_graph_capture"))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph input dependency preparation failed for seq_len="
                          << input.seq_len);
                return false;
            }

            const std::string capture_begin_boundary =
                "before_begin:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_begin_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture begin rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            /*
             * The cache owns begin/body/end as one scoped transaction. The
             * callback cannot strand the native stream in capture mode, and a
             * failed graph body cannot enter eager recovery.
             */
            if (!cache.captureAndInstantiate(
                    key,
                    gpu_ctx,
                    stream,
                    [&]() -> bool
                    {
                        return executor_.executeFastDecode(
                            *forward_cache.graph,
                            ctx,
                            &forward_cache.collective_nodes);
                    }))
            {
                LOG_ERROR(
                    "[ForwardExecutionEngine] Mandatory prefill graph capture transaction failed for seq_len="
                    << input.seq_len);
                return false;
            }

            const std::string capture_launch_boundary =
                "before_launch_after_capture:seq=" + std::to_string(input.seq_len) +
                ":real=" + std::to_string(real_seq_len) +
                ":bucket=" + std::to_string(bucket_seq_len) +
                ":offset=" + std::to_string(input.position_offset);
            if (!host.waitAtPrefillGraphCaptureBoundary(
                    input,
                    input.device,
                    capture_launch_boundary,
                    stream))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture launch rendezvous failed for seq_len="
                          << input.seq_len << " on " << input.device.toString());
                return false;
            }

            // Kernels recorded during HIP/CUDA stream capture are not executed
            // until the executable graph is launched. Launch once immediately so
            // the capture request produces logits and advances device state.
            const std::string recapture_reason =
                phase == PrefillGraphPhase::Initialized
                    ? "lazy_initialized_after_request_reset"
                    : "armed_warmup";
            if (!launchPrefillGraph(gpu_ctx, stream, "capture", PrefillGraphPhase::Ready, "launch_after_capture",
                                    recapture_reason))
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

            if (!publishPrefillCapturedTerminalState(
                    stream,
                    "prefill_graph_capture_launch"))
            {
                return false;
            }

            if (!executor_.publishSnapshotsAfterGraphExecution(
                    *forward_cache.graph,
                    stream,
                    "prefill_graph_capture_launch"))
            {
                return false;
            }

            LOG_INFO("[ForwardExecutionEngine] Prefill graph CAPTURED seq_len=" << input.seq_len
                                                                                << " nodes=" << cache.nodeCount(key));
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Ready,
                "capture",
                recapture_reason);
            return true;
        }

        if (can_attempt_capture && capture_ready_reason != PrefillGraphRejectReason::None)
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture readiness rejected after "
                      << prefillGraphPhaseName(phase)
                      << ": " << toString(capture_ready_reason)
                      << (capture_ready_reject_stage_name.empty() ? "" : " stage=")
                      << capture_ready_reject_stage_name
                      << (capture_ready_reject_stage_type.empty() ? "" : " type=")
                      << capture_ready_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                phase,
                "rejected",
                toString(capture_ready_reason),
                capture_ready_reject_stage_name,
                capture_ready_reject_stage_type);
            return false;
        }

        // === WARMUP/COLD PATH ===
        bool cold_capture_candidate = false;
        PrefillGraphRejectReason cold_reject_reason = PrefillGraphRejectReason::None;
        std::string cold_reject_stage_name;
        std::string cold_reject_stage_type;
        const bool cold_phase =
            phase == PrefillGraphPhase::Cold ||
            phase == PrefillGraphPhase::Initialized ||
            phase == PrefillGraphPhase::Warmup;
        if (cold_phase)
        {
            if (padded_preflight_checked)
            {
                cold_reject_reason = padded_preflight_reason;
                cold_reject_stage_name = padded_reject_stage_name;
                cold_reject_stage_type = padded_reject_stage_type;
            }
            else
            {
                cold_reject_reason = preflightPrefillGraph(
                    cache,
                    *forward_cache.graph,
                    key,
                    forward_cache.collective_nodes,
                    input,
                    snapshots_active,
                    moe_rebalancing_active,
                    padded_bucket
                        ? PrefillGraphPreflightMode::ColdPaddedSupport
                        : PrefillGraphPreflightMode::Default,
                    prefill_collectives_graph_capturable,
                    moe_rebalancing_graph_stable,
                    host_prefill_graph_disabled,
                    &cold_reject_stage_name,
                    &cold_reject_stage_type);
            }
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
            }
        }

        if (cold_phase && (!cold_capture_candidate || !cold_stream_ready))
        {
            LOG_ERROR("[ForwardExecutionEngine] Prefill graph capture rejected: "
                      << toString(cold_reject_reason)
                      << (cold_reject_stage_name.empty() ? "" : " stage=")
                      << cold_reject_stage_name
                      << (cold_reject_stage_type.empty() ? "" : " type=")
                      << cold_reject_stage_type
                      << " seq_len=" << input.seq_len
                      << "; refusing eager prefill fallback");
            publishPrefillGraphObservation(
                forward_cache,
                input,
                key,
                PrefillGraphPhase::Cold,
                "rejected",
                toString(cold_reject_reason),
                cold_reject_stage_name,
                cold_reject_stage_type);
            return false;
        }

        // Execute normally to warm up lazy allocations.
        bool exec_success = executor_.executeFastDecode(
            *forward_cache.graph, ctx, &forward_cache.collective_nodes);

        if (!exec_success)
            return false;

        if (!publishPrefillCapturedTerminalState(
                nullptr,
                "prefill_graph_warmup"))
        {
            return false;
        }

        if (!executor_.publishSnapshotsAfterGraphExecution(
                *forward_cache.graph,
                nullptr,
                "prefill_graph_warmup"))
        {
            return false;
        }

        // After successful warmup, check if graph capture is eligible
        if (cold_phase)
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

        std::unique_lock<std::mutex> materialization_lock;
        if (input_in.device.is_gpu())
        {
            materialization_lock = std::unique_lock<std::mutex>(
                gpuCacheMissGraphMaterializationMutex());
        }

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

        /*
         * A non-embedding pipeline stage consumes an activation produced by the
         * preceding stage. Perform that handoff before graph construction so
         * model graph builders remain declarative and never materialize GPU
         * activations through data()/mutable_data(). TransferEngine selects D2D,
         * peer transport, or the explicitly designed heterogeneous host-staged
         * boundary and publishes destination authority as one operation.
         */
        const auto initial_pp_copy = host.resolvePPCopyInfo(effective_input);
        if (initial_pp_copy.needs_copy)
        {
            auto pp_result = TransferEngine::instance().copyActivation(
                initial_pp_copy.external_hidden,
                initial_pp_copy.working_buffer,
                initial_pp_copy.device,
                initial_pp_copy.copy_bytes);
            if (!pp_result.success)
            {
                throw std::runtime_error(
                    "Initial PP hidden-state copy failed: " + pp_result.error);
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
        auto collective_nodes = graph.collectiveNodeNames();

        LOG_DEBUG("[ForwardExecutionEngine] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Empty forward graph");
            return false;
        }

        const bool bucketed_prefill_miss =
            should_cache && build_cache && signature.is_bucketed_prefill && !is_decode;
        bool bucketed_prefill_capture_candidate = false;
        PrefillGraphRejectReason bucketed_prefill_reject_reason = PrefillGraphRejectReason::None;
        std::string bucketed_prefill_reject_stage_name;
        std::string bucketed_prefill_reject_stage_type;

        if (bucketed_prefill_miss)
        {
            PrefillGraphCache preflight_cache(makePrefillGraphConfigFromEnv());
            PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
            bool prefill_collectives_graph_capturable = false;
            if (!collective_nodes.empty())
            {
                IDeviceContext *prefill_preflight_ctx = host.getDeviceContext(effective_input.device);
                prefill_collectives_graph_capturable =
                    host.buildDecodeCapturePolicy(
                        /*has_collective_nodes=*/true,
                        prefill_preflight_ctx)
                        .collectives_graph_capturable;
            }

            bucketed_prefill_reject_reason = preflightPrefillGraph(
                preflight_cache,
                graph,
                key,
                collective_nodes,
                effective_input,
                executor_.config().snapshot_callback != nullptr,
                host.isMoeRebalancingActive(),
                PrefillGraphPreflightMode::Default,
                prefill_collectives_graph_capturable,
                host.isMoeRebalancingGraphStableForPrefillCapture(),
                host.prefillGraphCaptureDisabledByHost(),
                &bucketed_prefill_reject_stage_name,
                &bucketed_prefill_reject_stage_type);
            bucketed_prefill_capture_candidate =
                (bucketed_prefill_reject_reason == PrefillGraphRejectReason::None);

            if (!bucketed_prefill_capture_candidate)
            {
                if (build_cache && !build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(makePrefillGraphConfigFromEnv());
                }
                if (build_cache && build_cache->prefill_graph_cache)
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }

                LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill graph rejected by preflight before execution: "
                          << toString(bucketed_prefill_reject_reason)
                          << (bucketed_prefill_reject_stage_name.empty() ? "" : " stage=")
                          << bucketed_prefill_reject_stage_name
                          << (bucketed_prefill_reject_stage_type.empty() ? "" : " type=")
                          << bucketed_prefill_reject_stage_type
                          << " real_seq_len=" << effectiveRealSeqLen(effective_input)
                          << " bucket_seq_len=" << effectiveBucketSeqLen(effective_input)
                          << "; refusing eager prefill fallback");
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

        if (materialization_lock.owns_lock())
        {
            materialization_lock.unlock();
        }

        // Notify host that graph is ready — allows releasing transient resources
        // (e.g., mmap pages) before execution allocates large activation buffers.
        if (!first_graph_ready_fired_)
        {
            first_graph_ready_fired_ = true;
            host.onFirstGraphReady();
        }

        bool success = false;

        /*
         * Retain the concrete producer stream independently of graph caching.
         * Short GPU prefills can be intentionally ineligible for caching, but
         * their hidden-state consumers need the same exact ordering contract as
         * Ready graph replays.
         */
        void *execution_stream_used = nullptr;

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

            void *execution_stream = nullptr;
            IWorkerGPUContext *execution_gpu_ctx = nullptr;
            if (ctx->deviceId().is_gpu())
            {
                try
                {
                    execution_gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
                    if (!execution_gpu_ctx)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = execution_gpu_ctx->defaultStream();
                }
                catch (const std::exception &e)
                {
                    LOG_ERROR("[ForwardExecutionEngine] Could not resolve forward graph stream for "
                              << ctx->deviceId().toString() << ": " << e.what());
                    return false;
                }

                if (bucketed_prefill_capture_candidate)
                {
                    /*
                     * The first bucketed-prefill cache miss is the warmup pass
                     * for the later monolithic graph capture.  Run it on the
                     * same explicit stream that the Warmup->Capture transition
                     * will record on; otherwise lazy first-use work may be
                     * ordered on one stream while capture later observes another.
                     */
                    if (!build_cache->prefill_capture_stream.ensure(execution_gpu_ctx) ||
                        !build_cache->prefill_capture_stream.stream)
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Bucketed prefill cache miss could not create an explicit capture stream for "
                                  << ctx->deviceId().toString());
                        return false;
                    }
                    execution_stream = build_cache->prefill_capture_stream.stream;
                }
                execution_stream_used = execution_stream;
            }

            /*
             * Cache-hit replay runs a stream/dynamic-parameter prelude before
             * launching the cached graph. Cache misses need the same contract
             * for their first execution: graph capture may start inside
             * DeviceGraphExecutor, so stages such as EmbeddingStage must upload
             * token IDs to their workspace before executor-side capture begins.
             *
             * The stream is Llaminar's owned worker stream, never the legacy
             * CUDA/HIP null stream. Stages that do not override
             * updateDynamicParams() simply ignore this setup.
             */
            std::vector<IComputeStage *> cache_miss_dynamic_param_stages;
            if (ctx->deviceId().is_gpu())
            {
                if (!execution_stream)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss graph execution requires an explicit stream for "
                              << ctx->deviceId().toString());
                    return false;
                }

                const auto &order = graph.getExecutionOrder();
                for (const auto &node_name : order)
                {
                    ComputeNode *node = graph.getNode(node_name);
                    if (!node || !node->stage)
                        continue;
                    node->stage->setGPUStream(execution_stream);
                    if (node->stage->hasDynamicParams())
                        cache_miss_dynamic_param_stages.push_back(node->stage.get());
                }

                if (!is_decode)
                {
                    /*
                     * Padded prefill replay params were pushed once before
                     * workspace allocation so host-only stages can see the
                     * metadata early.  GPU GDN/short-conv stages also upload a
                     * workspace-backed effective-length scalar, so repeat the
                     * update after the workspace has been bound and after every
                     * stage owns this explicit stream.  Otherwise a warmup run
                     * can leave the scalar at the previous real length and the
                     * following capture mutates live GDN state through one or
                     * more padding rows.
                     */
                    updatePrefillReplayParamStages(
                        effective_input,
                        prefill_replay_param_stages);
                }
            }

            if (!host.prepareLiveStateForForwardGraphExecution(
                    effective_input,
                    execution_stream,
                    ctx->deviceId()))
            {
                LOG_ERROR("[ForwardExecutionEngine] Failed to prepare live state for forward graph execution");
                return false;
            }

            if (host.computeAllPositionLogitsEnabled() &&
                host.allPositionLogitRows() > 0)
            {
                if (!host.prepareAllPositionVerifierGraphMetadata(
                        effective_input,
                        execution_stream,
                        ctx->deviceId()))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to prepare all-position verifier graph metadata");
                    return false;
                }
            }

            const int position_row_count =
                forwardPositionRowCount(effective_input);
            if ((effective_input.position_ids_device ||
                 effective_input.position_ids) &&
                position_row_count <= 0)
            {
                LOG_ERROR("[ForwardExecutionEngine] Cache-miss graph received invalid flattened position geometry: batch="
                          << effective_input.batch_size
                          << " seq_len=" << effective_input.seq_len);
                return false;
            }
            for (auto *stage : cache_miss_dynamic_param_stages)
            {
                /*
                 * Cache misses can still enter executor-side graph capture.
                 * Refresh explicit position rows after stream binding so RoPE
                 * captures the current request/chunk positions, not only the
                 * scalar offset baked into the graph build.
                 */
                if (effective_input.position_ids_device)
                {
                    if (!stage->supportsDeviceResidentDynamicPositionReplay())
                    {
                        LOG_ERROR("[ForwardExecutionEngine] Stage "
                                  << stage->name()
                                  << " cannot consume device-resident replay positions without host scalar state");
                        return false;
                    }
                    stage->updateDynamicDevicePositionIds(
                        effective_input.position_ids_device,
                        position_row_count);
                    if (stage->type() == ComputeStageType::ROPE)
                        continue;
                    stage->updateDynamicParams(
                        effective_input.position_offset,
                        effective_input.seq_len);
                }
                else if (effective_input.position_ids)
                {
                    stage->updateDynamicParams(
                        effective_input.position_offset,
                        effective_input.seq_len);
                    stage->updateDynamicPositionIds(
                        effective_input.position_ids,
                        position_row_count);
                }
                else
                {
                    stage->updateDynamicParams(
                        effective_input.position_offset,
                        effective_input.seq_len);
                }
            }

            if (!host.waitBeforeForwardGraphExecution(
                    effective_input,
                    ctx->deviceId(),
                    /*cache_miss=*/true))
            {
                LOG_ERROR("[ForwardExecutionEngine] Forward graph pre-execution rendezvous failed for "
                          << ctx->deviceId().toString());
                return false;
            }

            success = executor_.execute(graph, ctx);
        }

        DeviceId producer_device = effective_input.device;
        bool deferred_all_position_verifier_sync = false;
        bool deferred_main_decode_sync = false;
        if (success)
        {
            /*
             * A unified pipeline graph may finish on a device other than the
             * input token owner. Resolve that output device's worker stream
             * explicitly so provenance still names the stream which executes
             * its final hidden-state stages.
             */
            producer_device =
                resolveForwardOutputProducerDevice(
                    output,
                    effective_input.device);
            if (has_unified_pp && producer_device.is_gpu())
            {
                IWorkerGPUContext *producer_gpu_ctx =
                    host.getWorkerGPUContext(producer_device);
                if (!producer_gpu_ctx || !producer_gpu_ctx->defaultStream())
                {
                    LOG_ERROR("[ForwardExecutionEngine] Unified GPU forward completed without an explicit output producer stream on "
                              << producer_device.toString());
                    return false;
                }
                execution_stream_used = producer_gpu_ctx->defaultStream();
            }

            publishForwardOutputProvenance(
                output,
                producer_device,
                execution_stream_used,
                is_decode,
                host.computeAllPositionLogitsEnabled(),
                effective_input.seq_len,
                effective_input.batch_size);

            if (is_decode && producer_device.is_gpu())
            {
                const bool all_position_verifier =
                    host.computeAllPositionLogitsEnabled();
                const bool defer_all_position_verifier =
                    all_position_verifier &&
                    host.shouldDeferAllPositionVerifierFinalSync();
                const bool defer_main_decode =
                    !all_position_verifier &&
                    host.shouldDeferMainDecodeFinalSync();

                if ((defer_all_position_verifier || defer_main_decode) &&
                    !execution_stream_used)
                {
                    LOG_ERROR("[ForwardExecutionEngine] GPU cache-miss decode requested a device logits handoff without an explicit producer stream on "
                              << producer_device.toString());
                    return false;
                }

                if (defer_all_position_verifier)
                {
                    host.setPendingAllPositionVerifierStream(
                        execution_stream_used);
                    deferred_all_position_verifier_sync = true;
                }
                if (defer_main_decode)
                {
                    host.setPendingMainDecodeStream(execution_stream_used);
                    deferred_main_decode_sync = true;
                }
            }
        }

        /*
         * Host materialization remains the default public forward boundary.
         * MTP decode is different: its immediate consumer is a GPU sampler or
         * verifier-publication stage. Cache misses publish the exact producer
         * stream through the same typed handoff used by cached replay, so the
         * first execution cannot silently fall back to a host stream drain.
         */
        if (success)
        {
            if (!is_decode && effectiveRealSeqLen(effective_input) > 0)
            {
                const int terminal_row = effectiveRealSeqLen(effective_input) - 1;
                if (!executor_.publishCapturedTerminalStateAfterGraphExecution(
                        graph,
                        terminal_row,
                        nullptr,
                        should_cache
                            ? "prefill_cache_miss"
                            : "prefill_eager",
                        effective_input.sequence_lengths_device,
                        effective_input.batch_size,
                        effective_input.seq_len,
                        effective_input.sequence_lengths))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish captured terminal prefill state after cache miss");
                    return false;
                }
            }

            const bool device_consumer_owns_logits =
                deferred_all_position_verifier_sync ||
                deferred_main_decode_sync;
            IDeviceContext *sync_ctx =
                host.getDeviceContext(producer_device);
            if (sync_ctx && !device_consumer_owns_logits)
            {
                if (!host.publishLogitsAtBoundary(
                        output.logits,
                        sync_ctx,
                        execution_stream_used))
                {
                    LOG_ERROR("[ForwardExecutionEngine] Failed to publish cache-miss forward logits at the device boundary");
                    return false;
                }
            }

            const std::string stage_context =
                should_cache
                    ? forwardGraphPerfContext(signature)
                    : (is_decode ? "main_decode" : "prefill");
            collectTimeline(
                host,
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
            if (initial_pp_copy.needs_copy)
            {
                build_cache->pp_external_hidden_state =
                    initial_pp_copy.external_hidden;
                build_cache->pp_working_buffer =
                    initial_pp_copy.working_buffer;
                build_cache->pp_copy_bytes = initial_pp_copy.copy_bytes;
                build_cache->pp_device = initial_pp_copy.device;
                build_cache->pp_needs_copy = true;

                LOG_DEBUG("[ForwardExecutionEngine] Stored PP copy info: "
                          << initial_pp_copy.copy_bytes << " bytes on "
                          << initial_pp_copy.device.toString());
            }

            LOG_DEBUG("[ForwardExecutionEngine] Cached forward graph for signature "
                      << "[seq_len=" << signature.seq_len
                      << ", batch_size=" << signature.batch_size
                      << ", device=" << signature.device.to_string()
                      << ", decode=" << signature.decode
                      << ", decode_has_history=" << signature.decode_has_history
                      << "] (" << build_cache->graph->size() << " stages)");

            if (signature.is_bucketed_prefill)
            {
                if (!build_cache->prefill_graph_cache)
                {
                    build_cache->prefill_graph_cache =
                        std::make_unique<PrefillGraphCache>(makePrefillGraphConfigFromEnv());
                }

                PrefillGraphCacheKey key = makePrefillGraphKey(effective_input, host);
                if (bucketed_prefill_capture_candidate)
                {
                    build_cache->prefill_graph_cache->markWarmedUp(key);
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Warmup,
                        "warmup",
                        "none");
                    if (build_cache->prefill_graph_cache->config().trace)
                    {
                        LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED during cache miss warmup: seq_len="
                                 << effective_input.seq_len);
                    }
                }
                else
                {
                    publishPrefillGraphObservation(
                        *build_cache,
                        effective_input,
                        key,
                        PrefillGraphPhase::Cold,
                        "rejected",
                        toString(bucketed_prefill_reject_reason),
                        bucketed_prefill_reject_stage_name,
                        bucketed_prefill_reject_stage_type);
                }
                enforceBucketedPrefillForwardCapacity(&signature);
            }
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
        IForwardExecutionHost &host,
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

        IWorkerGPUContext *gpu_ctx = host.getWorkerGPUContext(ctx->deviceId());
        if (!gpu_ctx)
        {
            LOG_ERROR("[ForwardExecutionEngine] Host did not provide a worker GPU context for timeline collection on "
                      << ctx->deviceId().toString());
            return;
        }
        timeline.collect(gpu_ctx);

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

    void ForwardExecutionEngine::forEachCachedStage(
        const std::function<void(IComputeStage *)> &visitor) const
    {
        for (const auto &[sig, cache] : cache_)
        {
            (void)sig;
            if (!cache.valid || !cache.graph)
                continue;
            for (const auto &node_name : cache.graph->getExecutionOrder())
            {
                auto *node = cache.graph->getNode(node_name);
                if (node && node->stage)
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
        snapshot.reject_stage_name.clear();
        snapshot.reject_stage_type.clear();

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
            snapshot.reject_stage_name = observation.reject_stage_name;
            snapshot.reject_stage_type = observation.reject_stage_type;
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
        snapshot.initialized_count = prefill_cache.initializedCount(key);
        snapshot.capture_count = prefill_cache.captureCount(key);
        snapshot.eviction_count += prefill_cache.evictionCount();
        return snapshot;
    }

} // namespace llaminar2
