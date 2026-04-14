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
#include "../../../backends/GPUDeviceContextPool.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <cstring>
#include <iomanip>

namespace llaminar2
{

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
        const bool forward_cache_eligible =
            config_.cache_config.enabled &&
            !has_unified_pp &&
            has_stable_forward_inputs &&
            (is_standard_path || is_partial_pp_path);

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
                input.seq_len,
                input.batch_size,
                input.device,
                is_decode,
                is_standard_path,
                config_.pp_stage_config.has_value(),
                pp_first_layer,
                pp_last_layer,
                pp_has_embedding,
                pp_has_lm_head};

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
            return executeCacheHit(input, output, *active_forward_cache, host,
                                   is_decode, start);
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

        return executeCacheMiss(input, output, forward_signature, build_cache,
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

        // For GPU graph replay: set the capture stream on all stages ONCE.
        // The capture_stream never changes between decode steps, so after the
        // first pass we skip this 339-stage loop entirely.
        void *replay_stream = forward_cache.segment_cache.capture_stream;
        if (replay_stream && !forward_cache.gpu_stream_applied)
        {
            const auto &order = forward_cache.graph->getExecutionOrder();
            for (const auto &node_name : order)
            {
                ComputeNode *node = forward_cache.graph->getNode(node_name);
                if (node && node->stage)
                    node->stage->setGPUStream(replay_stream);
            }
            forward_cache.gpu_stream_applied = true;
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
        for (auto *stage : forward_cache.dynamic_param_stages)
        {
            stage->updateDynamicParams(input.position_offset, input.seq_len);
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
            // Prefill: fast path without graph capture overhead
            success = executor_.executeFastDecode(
                *forward_cache.graph, ctx, &forward_cache.collective_nodes);
        }
        else
        {
            const auto capture_policy = host.buildDecodeCapturePolicy(
                has_collective_nodes,
                ctx,
                forward_cache.segment_cache.consecutive_failures);
            if (capture_policy.collective_segmented_enabled)
            {
                LOG_INFO("[ForwardExecutionEngine] Experimental collective segmented GPU-graph replay enabled");
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
            LOG_INFO("[DEVICE_DECODE] dev=" << input.device
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

        LOG_DEBUG("[ForwardExecutionEngine] Forward graph built with " << graph.size() << " stages");

        if (graph.size() == 0)
        {
            LOG_ERROR("[ForwardExecutionEngine] Empty forward graph");
            return false;
        }

        // Ensure GPU workspace is allocated for GEMM kernels (lazy initialization)
        host.ensureDeviceWorkspaceAllocated(graph);

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

            // Pre-compute collective node set for fast decode intercept
            build_cache->collective_nodes.clear();
            for (const auto &n : build_cache->graph->getExecutionOrder())
            {
                auto *nd = build_cache->graph->getNode(n);
                if (nd && nd->stage)
                {
                    auto t = nd->stage->type();
                    if (t == ComputeStageType::ALLREDUCE ||
                        t == ComputeStageType::ALLGATHER ||
                        t == ComputeStageType::ALLGATHER_V)
                    {
                        build_cache->collective_nodes.insert(n);
                    }
                }
            }

            build_cache->valid = true;

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

            LOG_INFO("[ForwardExecutionEngine] Cached forward graph for signature "
                     << "[seq_len=" << signature.seq_len
                     << ", batch_size=" << signature.batch_size
                     << ", device=" << signature.device.to_string()
                     << ", decode=" << signature.decode
                     << "] (" << build_cache->graph->size() << " stages)");
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
        if (!debugEnv().gpu_stage_timing || suppress_timeline_ ||
            !ctx || !ctx->deviceId().is_gpu())
        {
            return;
        }

        auto &timeline = executor_.stageTimeline();
        if (!timeline.isInitialized())
        {
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

} // namespace llaminar2
