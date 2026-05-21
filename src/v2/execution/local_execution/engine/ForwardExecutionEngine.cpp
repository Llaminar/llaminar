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
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

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
            return chunk_input;
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

        return execute(chunk_input, output, host);
    }

    void ForwardExecutionEngine::invalidateAll()
    {
        for (auto &[_, cache] : cache_)
        {
            cache.invalidate();
        }
    }

    void ForwardExecutionEngine::clearCache()
    {
        invalidateAll();
        cache_.clear();
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
        auto start = std::chrono::high_resolution_clock::now();

        // Classify the execution path
        const bool is_decode = (input.seq_len == 1 && input.batch_size <= 1);
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
            ((input.token_ids != nullptr) && (input.position_ids != nullptr)) ||
            (is_pp_non_embedding_stage && (input.position_ids != nullptr));

        // Prefill graph caching: eligible for GPU devices with gpu_graphs enabled
        const bool prefill_cache_eligible =
            config_.cache_config.enabled &&
            !is_decode &&
            !has_unified_pp &&
            has_stable_forward_inputs &&
            (is_standard_path || is_partial_pp_path) &&
            input.device.is_gpu() &&
            debugEnv().execution.gpu_graphs;

        ForwardInput effective_input = input;
        std::optional<PrefillChunkRuntimePlan> raw_bucket_plan;
        bool bucketed_prefill = false;
        int bucketed_prefill_seq_len = input.seq_len;

        const auto &env = debugEnv();
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
            input.token_ids && input.batch_size == 1)
        {
            const int real_seq_len = effectiveRealSeqLen(input);
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
                LOG_ERROR("[ForwardExecutionEngine] Padded raw prefill input requires GPU graph bucket preflight: real_seq_len="
                          << real_seq_len << " bucket_seq_len=" << selection.bucket_seq_len
                          << " device=" << input.device.toString());
                return false;
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
                is_standard_path,
                config_.pp_stage_config.has_value(),
                pp_first_layer,
                pp_last_layer,
                pp_has_embedding,
                pp_has_lm_head,
                bucketed_prefill,
                bucketed_prefill ? bucketed_prefill_seq_len : 0};

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
            touchBucketedPrefillForwardCache(forward_signature, *active_forward_cache);
            const bool success = executeCacheHit(effective_input, output, *active_forward_cache, host,
                                                 is_decode, start);
            if (success && forward_signature.is_bucketed_prefill)
                enforceBucketedPrefillForwardCapacity(&forward_signature);
            return success;
        }

        // Cache MISS path
        ForwardGraphCache *build_cache = nullptr;
        bool should_cache_after_build = false;
        if (forward_cache_eligible)
        {
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

        return executeCacheMiss(build_input, output, forward_signature, build_cache,
                                should_cache_after_build, host, is_decode,
                                has_unified_pp, start);
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
            // Unified PP copy: data() handles all device coherence sync
            // automatically (including D2H via staging buffer).
            const void *src = forward_cache.pp_external_hidden_state->data();
            void *dst = forward_cache.pp_working_buffer->mutable_data();
            std::memcpy(dst, src, forward_cache.pp_copy_bytes);

            const auto &dev = forward_cache.pp_device;
            if (dev.is_gpu())
            {
                forward_cache.pp_working_buffer->ensureOnDevice(dev);
                forward_cache.pp_working_buffer->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE, dev);
            }
        }

        // For GPU graph replay: set the capture stream on all stages.
        // Normally the capture_stream never changes between decode steps,
        // but segment_cache.reset() (on capture retry or repeated replay
        // failures) destroys the stream and the next call creates a NEW one.
        // We track the last applied stream pointer and re-apply whenever it
        // changes so stages never hold a dangling/stale stream.
        //
        // Critical: if capture_stream was destroyed but not yet recreated
        // (e.g. between segment_cache.reset() and the next warmup phase),
        // applied_stream points at freed memory. Fall back to the device's
        // default stream for this forward pass so updateDynamicParams and
        // any stage kernels issue work on a live stream. The warmup phase
        // will re-apply the fresh capture_stream to all stages before
        // capture is attempted again.
        void *replay_stream = forward_cache.segment_cache.capture_stream;
        if (!replay_stream && forward_cache.applied_stream != nullptr)
        {
            // capture_stream was destroyed — previously applied stream is
            // now invalid. Fall back to the context's default stream.
            IDeviceContext *fallback_ctx = host.getDeviceContext(input.device);
            void *fallback_stream = nullptr;
            if (fallback_ctx && fallback_ctx->deviceId().is_gpu())
            {
                auto &pool = GPUDeviceContextPool::instance();
                IWorkerGPUContext &gpu_ctx = pool.getContext(fallback_ctx->deviceId());
                fallback_stream = gpu_ctx.defaultStream();
            }
            if (fallback_stream && fallback_stream != forward_cache.applied_stream)
            {
                const auto &order = forward_cache.graph->getExecutionOrder();
                for (const auto &node_name : order)
                {
                    ComputeNode *node = forward_cache.graph->getNode(node_name);
                    if (node && node->stage)
                        node->stage->setGPUStream(fallback_stream);
                }
                forward_cache.applied_stream = fallback_stream;
                // Leave gpu_stream_applied=true so warmup will still detect
                // the pointer change when it installs the new capture_stream.
            }
        }
        else if (replay_stream && (!forward_cache.gpu_stream_applied ||
                                   forward_cache.applied_stream != replay_stream))
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(replay_stream);
            }
            forward_cache.gpu_stream_applied = true;
            forward_cache.applied_stream = replay_stream;
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
        for (auto *stage : forward_cache.dynamic_param_stages)
        {
            stage->updateDynamicParams(input.position_offset, input.seq_len);
        }
        if (!is_decode)
        {
            updatePrefillReplayParamStages(input, forward_cache.prefill_replay_param_stages);
        }

        // Skip graph reset when Phase 3 replay is active — Phase 3 doesn't
        // call markCompleted(), so all flags are already false from last reset.
        if (!forward_cache.phase3_active)
        {
            forward_cache.graph->reset();
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

        // Graph capture (segmented/monolithic) is only beneficial for decode
        // graphs (seq_len=1) where the same fixed-shape graph replays thousands
        // of times. For prefill (seq_len>1), graph capture adds ~550ms one-shot
        // overhead (HIP graph capture + instantiation) that is never amortized
        // because prefill shapes change per prompt. Use executeFastDecode directly.
        bool used_segmented_capture = false;

        auto exec_t0 = std::chrono::high_resolution_clock::now();

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
            const auto capture_policy = host.buildDecodeCapturePolicy(
                has_collective_nodes,
                ctx,
                forward_cache.segment_cache.consecutive_failures);
            if (capture_policy.collective_segmented_enabled)
            {
                LOG_DEBUG("[ForwardExecutionEngine] Experimental collective segmented GPU-graph replay enabled");
            }

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
        }
        else
        {
            forward_cache.phase3_active = false;
        }

        // Sync the stream at the forward pass boundary so logits are
        // immediately available to the caller without per-access event waits.
        if (success)
        {
            if (forward_cache.phase3_active)
            {
                // Phase 3 replay already synchronized both capture_stream
                // and defaultStream at the end of executeReplayPhase().
                // Skip the redundant device-wide hipDeviceSynchronize and
                // just mark the mapped logits as host-visible.
                TensorBase *logits = host.logitsTensor();
                if (logits && logits->isMapped())
                {
                    logits->markMappedSynced();
                }
            }
            else
            {
                host.syncLogitsAtBoundary(ctx);
            }

            collectTimeline(ctx, is_decode, input, start);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

        // Decode step timing breakdown (enabled via TP_TIMING)
        if (is_decode && debugEnv().tp_timing)
        {
            double setup_us = std::chrono::duration<double, std::micro>(exec_t0 - start).count();
            double exec_us = std::chrono::duration<double, std::micro>(exec_t1 - exec_t0).count();
            double sync_us = std::chrono::duration<double, std::micro>(end - exec_t1).count();
            LOG_DEBUG("[DEVICE_DECODE] dev=" << input.device
                                             << " setup=" << std::fixed << std::setprecision(1) << setup_us << "us"
                                             << " exec=" << exec_us << "us"
                                             << " sync=" << sync_us << "us"
                                             << " total=" << (ms * 1000.0) << "us"
                                             << " phase3=" << forward_cache.phase3_active);
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

        PrefillGraphCacheKey key;
        key.seq_len = input.seq_len;
        key.device_id = input.device;
        // Tier 1 defaults: domain_id="single", placement_epoch=0, topology_signature=0

        auto phase = cache.phase(key);
        const int real_seq_len = effectiveRealSeqLen(input);
        const int bucket_seq_len = effectiveBucketSeqLen(input);
        const bool padded_bucket = isPaddedBucketExecution(input);
        const bool snapshots_active = (executor_.config().snapshot_callback != nullptr);
        const bool moe_rebalancing_active = host.isMoeRebalancingActive();
        bool padded_preflight_checked = false;
        PrefillGraphRejectReason padded_preflight_reason = PrefillGraphRejectReason::None;

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

            if (!cache.launch(key))
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
                GraphCaptureGuard capture_guard;
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
            if (!cache.launch(key))
            {
                LOG_ERROR("[ForwardExecutionEngine] Prefill graph launch-after-capture failed for seq_len=" << input.seq_len);
                return false;
            }

            for (auto *stage : forward_cache.replay_callback_stages)
                stage->onGraphReplayed();

            LOG_INFO("[ForwardExecutionEngine] Prefill graph CAPTURED seq_len=" << input.seq_len
                                                                                << " nodes=" << cache.nodeCount(key));
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
                if (cache.config().trace)
                    LOG_INFO("[ForwardExecutionEngine] Prefill graph ARMED for capture: seq_len=" << input.seq_len);
            }
            else if (cache.config().trace)
            {
                LOG_INFO("[ForwardExecutionEngine] Prefill graph capture rejected: "
                         << toString(cold_reject_reason) << " seq_len=" << input.seq_len);
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
            PrefillGraphCacheKey key;
            key.seq_len = effective_input.seq_len;
            key.device_id = effective_input.device;

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

        // Ensure GPU workspace is allocated for GEMM kernels (lazy initialization)
        host.ensureDeviceWorkspaceAllocated(graph);

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

            collectTimeline(
                host.getDeviceContext(effective_input.device),
                is_decode, effective_input, start);
        }

        // Cache the graph for future matching forward signatures
        if (should_cache && build_cache && success)
        {
            build_cache->graph = std::make_unique<ComputeGraph>(std::move(graph));
            build_cache->output = output;

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
        std::chrono::high_resolution_clock::time_point start)
    {
        if (!debugEnv().gpu_stage_timing ||
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
