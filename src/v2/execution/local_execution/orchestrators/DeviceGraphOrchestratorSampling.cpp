/**
 * @file DeviceGraphOrchestratorSampling.cpp
 * @brief DeviceGraphOrchestrator method shard (51042..55992).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    bool DeviceGraphOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        bool submitted_async_device_mutation = false;
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] restoreLivePrefixState failed: " << reason);
            if (state_.device_id.is_gpu() &&
                submitted_async_device_mutation)
            {
                /*
                 * Once restore reads or writes are queued, returning would
                 * release snapshot-owned source allocations without a
                 * completion event. There is no legal synchronous cleanup or
                 * fallback path for that ownership violation.
                 */
                std::terminate();
            }
            return false;
        };

        if (!snapshot.valid || !state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return fail("invalid snapshot, missing KV cache, or sequence index out of range");
        }
        if (snapshot.cached_tokens < 0 || snapshot.cached_tokens > state_.kv_cache->max_seq_len())
        {
            return fail("cached token count out of range: cached_tokens=" +
                        std::to_string(snapshot.cached_tokens) +
                        " max_seq_len=" + std::to_string(state_.kv_cache->max_seq_len()));
        }

        if (!validMTPShiftedCacheOrigin(snapshot.mtp_shifted_origin) ||
            (!state_.mtp_kv_caches.empty() &&
             snapshot.mtp_shifted_origin != graph_builder_->config().mtpShiftedCacheOrigin()))
        {
            return fail("snapshot shifted-cache origin differs from authenticated model ownership");
        }

        void *stream = explicitGPUStreamForOperation("restoreLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing explicit GPU stream");
        }
        if (!waitForPendingRequestStateReset(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "restore_live_prefix_request_state_reset"))
        {
            return fail("request-state reset wait failed");
        }
        if (!waitForPendingLivePrefixCheckpointReady(
                stream,
                "restore_live_prefix_state_source"))
        {
            return fail("pending live prefix checkpoint source wait failed");
        }
        if (!joinPublishedLiveStateHandoffs(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "restore_live_prefix_state_previous_mutation"))
        {
            return fail("pending live prefix mutation wait failed");
        }
        if (!waitForSnapshotReady(
                snapshot,
                stream,
                "restore_live_prefix_state_snapshot"))
        {
            return fail("prefix snapshot readiness wait failed");
        }
        if (!waitForPendingLiveGraphProducersBeforePrefixMutation(
                stream,
                "restore_live_prefix_state"))
        {
            return fail("pending live graph producer wait failed");
        }
        if (!prepareLivePrefixMutationReadyEvent(
                stream,
                "restore_live_prefix_state"))
        {
            return fail("could not preflight live-state readiness event");
        }

        PrefixArchivePublicationScope restore_completion([&](PrefixArchiveCompletion) {
            if (!state_.device_id.is_gpu())
                return true;
            std::vector<PrefixPayloadReadLease> sources;
            std::vector<std::shared_ptr<void>> device_sources;
            for (const auto &source : snapshot.blocks)
                sources.push_back(PrefixPayloadReadLease::wholeArchive(source));
            for (const auto &source : snapshot.mtp_blocks)
                sources.push_back(PrefixPayloadReadLease::wholeArchive(source));
            for (const auto &source : snapshot.device_sequence_state_checkpoints)
                device_sources.push_back(source.storage);
            const char *producer = snapshot.logical_checkpoint
                ? "restore_logical_checkpoint"
                : snapshot.cached_tokens == 0
                    ? "restore_payload_checkpoint_zero"
                    : "restore_payload_checkpoint";
            return recordLivePrefixMutationReady(
                stream, producer, std::move(sources), std::move(device_sources));
        });

        /*
         * Restoring KV/GDN/position state invalidates any previously current
         * terminal-hidden sidecar input unless this snapshot explicitly
         * carries a matching terminal-hidden payload.  Leaving the flag set
         * after a hidden-less restore can combine fresh restored caches with an
         * old PREFIX_TERMINAL_HIDDEN row on the next MTP sidecar step.
         */
        if (graph_builder_ && graph_builder_->config().mtp.enabled)
        {
            state_.mtp_terminal_hidden_publication.invalidate();
        }

        if (snapshot.logical_checkpoint)
        {
            // Validate the complete CPU endpoint map before truncating any
            // cache. A logical token rollback cannot reconstruct lexical or
            // incomplete-pool history, including an accepted zero-token state.
            std::vector<const HostCacheAuxiliaryCheckpoint *> host_auxiliary(
                state_.mtp_kv_caches.size() + 1, nullptr);
            if (state_.device_id.is_gpu() && !snapshot.host_auxiliary_checkpoints.empty())
                return fail("GPU logical checkpoint unexpectedly carries CPU auxiliary state");
            for (const auto &checkpoint : snapshot.host_auxiliary_checkpoints)
            {
                const int index = checkpoint.cache_depth + 1;
                if (index < 0 || index >= static_cast<int>(host_auxiliary.size()) || host_auxiliary[index])
                    return fail("CPU logical checkpoint has an invalid or duplicate auxiliary cache depth");
                IKVCache *cache = index == 0 ? state_.kv_cache.get() : state_.mtp_kv_caches[index - 1].get();
                if (!cache || !cache->auxiliaryState())
                    return fail("CPU logical checkpoint carries auxiliary state for a missing owner");
                checkpoint.validate(*cache->auxiliaryState(), seq_idx);
                host_auxiliary[index] = &checkpoint;
            }
            if (!state_.device_id.is_gpu())
                for (size_t index = 0; index < host_auxiliary.size(); ++index)
                {
                    const IKVCache *cache = index == 0 ? state_.kv_cache.get() : state_.mtp_kv_caches[index - 1].get();
                    const auto *owner = cache ? cache->auxiliaryState() : nullptr;
                    const bool required = owner && owner->checkpointCapacity(seq_idx) > 0;
                    if (required != bool(host_auxiliary[index]))
                        return fail("CPU logical checkpoint is missing its exact accepted auxiliary endpoint");
                }
            if (snapshot.blocks.size() > 1)
            {
                return fail("logical checkpoint contains more than one hybrid payload block");
            }
            const PrefixBlockHandle *hybrid_handle = nullptr;
            if (!snapshot.blocks.empty())
            {
                hybrid_handle = &snapshot.blocks.front();
                const bool includes_hybrid_state =
                    hybrid_handle->layout.includes_hybrid_state &&
                    hybrid_handle->layout.hybrid_state_bytes > 0;
                const bool includes_terminal_hidden =
                    hybrid_handle->layout.includes_terminal_hidden &&
                    hybrid_handle->layout.terminal_hidden_bytes > 0;
                const bool terminal_hidden_source_available =
                    state_.device_id.is_gpu()
                        ? hybrid_handle->deviceTerminalHiddenData() != nullptr
                        : hybrid_handle->terminal_hidden != nullptr;
                if ((!includes_hybrid_state && !includes_terminal_hidden) ||
                    (includes_hybrid_state &&
                     (!hybrid_handle->has_hybrid_state ||
                      (hybrid_handle->layout.hybrid_host_state_bytes > 0 && !hybrid_handle->hybrid_payload) ||
                      (hybrid_handle->layout.hybrid_device_state_bytes > 0 && !hybridPayloadDevicePtr(*hybrid_handle)))) ||
                    (includes_terminal_hidden &&
                     (!hybrid_handle->has_terminal_hidden ||
                      !terminal_hidden_source_available)) ||
                    hybrid_handle->key.token_start != 0 ||
                    hybrid_handle->key.token_count != snapshot.cached_tokens)
                {
                    return fail("logical checkpoint payload block is invalid");
                }
            }
            if (snapshot.mtp_cached_tokens.size() != state_.mtp_kv_caches.size())
            {
                return fail("logical checkpoint MTP cache count does not match runner state");
            }

            const DeviceKVSequenceStateCheckpoint *main_sequence_checkpoint =
                nullptr;
            std::vector<const DeviceKVSequenceStateCheckpoint *>
                shifted_sequence_checkpoints(
                    state_.mtp_kv_caches.size(),
                    nullptr);
            if (state_.device_id.is_gpu())
            {
                for (const DeviceKVSequenceStateCheckpoint &checkpoint :
                     snapshot.device_sequence_state_checkpoints)
                {
                    if (!checkpoint.valid() ||
                        checkpoint.device != state_.device_id ||
                        checkpoint.sequence_index != seq_idx)
                    {
                        return fail(
                            "logical checkpoint contains an invalid or foreign device sequence-state payload");
                    }

                    IKVCache *checkpoint_cache = nullptr;
                    if (checkpoint.cache_depth < 0)
                    {
                        if (main_sequence_checkpoint)
                        {
                            return fail(
                                "logical checkpoint contains duplicate main device sequence state");
                        }
                        main_sequence_checkpoint = &checkpoint;
                        checkpoint_cache = state_.kv_cache.get();
                    }
                    else
                    {
                        const int depth = checkpoint.cache_depth;
                        if (depth >= static_cast<int>(
                                         state_.mtp_kv_caches.size()) ||
                            shifted_sequence_checkpoints[
                                static_cast<size_t>(depth)])
                        {
                            return fail(
                                "logical checkpoint contains an invalid or duplicate shifted device sequence-state depth");
                        }
                        checkpoint_cache =
                            state_.mtp_kv_caches[
                                static_cast<size_t>(depth)]
                                .get();
                        shifted_sequence_checkpoints[
                            static_cast<size_t>(depth)] = &checkpoint;
                    }

                    if (!checkpoint_cache ||
                        checkpoint.bytes !=
                            checkpoint_cache
                                ->deviceSequenceStateCheckpointBytes())
                    {
                        return fail(
                            "logical checkpoint device sequence-state layout does not match its live cache");
                    }
                }

                if (!main_sequence_checkpoint)
                {
                    return fail(
                        "GPU logical checkpoint is missing main device sequence state");
                }
                for (size_t depth = 0;
                     depth < state_.mtp_kv_caches.size();
                     ++depth)
                {
                    if (state_.mtp_kv_caches[depth] &&
                        !shifted_sequence_checkpoints[depth])
                    {
                        return fail(
                            "GPU logical checkpoint is missing shifted device sequence state for depth=" +
                            std::to_string(depth));
                    }
                }
            }
            else if (!snapshot.device_sequence_state_checkpoints.empty())
            {
                return fail(
                    "CPU logical checkpoint unexpectedly carries GPU sequence state");
            }
            // A PP shard of a hybrid model can contain only FA layers. The
            // wrapper type alone does not imply any recurrent state to reset;
            // its captured sequence-metadata restore is sufficient in that case.
            if (const auto *hybrid = dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
                snapshot.cached_tokens == 0 &&
                state_.device_id.is_gpu() &&
                hybrid && hybrid->gdnLayerCount() > 0)
            {
                return fail(
                    "zero-token hybrid logical checkpoint restore requires a streamful recurrent-state reset");
            }

            /*
             * Shifted MTP caches have independent GDN/short-conv recurrent
             * banks. Truncating their logical token counts cannot reconstruct
             * those banks, so every non-empty hybrid shifted cache must carry
             * exactly one hybrid-only checkpoint handle. Validate the complete
             * map before mutating any live cache; a partial snapshot must never
             * leave half of the transaction restored.
             */
            std::vector<const PrefixBlockHandle *> shifted_hybrid_handles(
                state_.mtp_kv_caches.size(),
                nullptr);
            for (const PrefixBlockHandle &handle : snapshot.mtp_blocks)
            {
                const int depth = handle.key.block_index;
                if (depth < 0 ||
                    depth >= static_cast<int>(state_.mtp_kv_caches.size()))
                {
                    return fail(
                        "logical checkpoint shifted hybrid depth out of range: " +
                        std::to_string(depth));
                }
                const auto &cache =
                    state_.mtp_kv_caches[static_cast<size_t>(depth)];
                const int mtp_cached_tokens =
                    snapshot.mtp_cached_tokens[static_cast<size_t>(depth)];
                if (!cache || mtp_cached_tokens <= 0)
                {
                    return fail(
                        "logical checkpoint carries shifted hybrid state for an empty or missing cache at depth=" +
                        std::to_string(depth));
                }
                if (shifted_hybrid_handles[static_cast<size_t>(depth)])
                {
                    return fail(
                        "logical checkpoint contains duplicate shifted hybrid state at depth=" +
                        std::to_string(depth));
                }

                const PrefixPayloadLayout expected =
                    liveHybridCheckpointLayout(
                        buildDensePrefixPayloadLayout(
                            *cache,
                            state_.device_id,
                            mtp_cached_tokens),
                        /*device_only=*/state_.device_id.is_gpu(), *cache);
                const bool source_available =
                    state_.device_id.is_gpu()
                        ? hybridPayloadDevicePtr(handle) != nullptr
                        : hybridPayloadHostPtr(handle) != nullptr;
                if (!handle.valid() ||
                    handle.key.token_start != 0 ||
                    handle.key.token_count != mtp_cached_tokens ||
                    !expected.includes_hybrid_state ||
                    expected.hybrid_state_bytes == 0 ||
                    !handle.layout.compatiblePayloadShape(expected) ||
                    !handle.has_hybrid_state ||
                    !source_available)
                {
                    return fail(
                        "logical checkpoint shifted hybrid payload is invalid at depth=" +
                        std::to_string(depth));
                }
                shifted_hybrid_handles[static_cast<size_t>(depth)] = &handle;
            }

            bool main_sequence_state_restored = false;
            if (state_.device_id.is_gpu())
            {
                std::string checkpoint_error;
                submitted_async_device_mutation = true;
                main_sequence_state_restored =
                    state_.kv_cache
                        ->restoreDeviceSequenceStateCheckpoint(
                            seq_idx,
                            main_sequence_checkpoint->data(),
                            main_sequence_checkpoint->bytes,
                            stream,
                            &checkpoint_error);
                if (!main_sequence_state_restored)
                {
                    return fail(
                        "main device sequence-state restore failed: " +
                        checkpoint_error);
                }
                PerfStatsCollector::addCounter(
                    "mtp",
                    "live_prefix_checkpoint_device_sequence_state_restores",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"cache", "main"},
                     {"depth", "-1"},
                     {"bytes",
                      std::to_string(main_sequence_checkpoint->bytes)},
                     {"implementation", "device_kernel"}});
            }
            else
            {
                main_sequence_state_restored =
                    state_.kv_cache->truncateSequence(
                        seq_idx,
                        snapshot.cached_tokens,
                        stream);
            }
            if (!main_sequence_state_restored)
            {
                return fail("main KV logical checkpoint truncate failed for tokens=" +
                            std::to_string(snapshot.cached_tokens));
            }
            if (snapshot.cached_tokens == 0)
            {
                if (!resetHybridPrefixPayloadState(
                        *state_.kv_cache,
                        IKVCache::StateResetContext{
                            .boundary =
                                IKVCache::StateResetBoundary::PrefixReplacement,
                            .execution_stream = stream,
                            .reason = "restore-logical-checkpoint-zero",
                        }))
                {
                    return fail(
                        "zero-token logical checkpoint GDN reset failed");
                }
            }
            else if (hybrid_handle)
            {
                if (hybrid_handle->layout.includes_hybrid_state &&
                    hybrid_handle->layout.hybrid_state_bytes > 0)
                {
                    PerfStatsCollector::ScopedTimer timer("mtp",
                                                          "restore_live_prefix_checkpoint_hybrid_import",
                                                          "decode",
                                                          state_.device_id.toString());
                    if (!importHybridPrefixPayload(
                            *state_.kv_cache,
                            *hybrid_handle,
                            seq_idx,
                            /*restore_for_suffix_prefill=*/false,
                            /*synchronize=*/false,
                            stream, HybridPrefixStatePurpose::LocalCheckpoint))
                    {
                        return fail("logical checkpoint hybrid payload import failed");
                    }
                }
                if (hybrid_handle->layout.includes_terminal_hidden &&
                    hybrid_handle->layout.terminal_hidden_bytes > 0)
                {
                    if (!restoreLiveCheckpointTerminalHidden(
                            *hybrid_handle,
                            stream,
                            "restore_logical_checkpoint",
                            /*publish_mailbox=*/false,
                            /*destination_row=*/seq_idx))
                    {
                        return fail(
                            "logical checkpoint terminal hidden restore failed");
                    }
                }
            }

            for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
            {
                auto &cache = state_.mtp_kv_caches[depth];
                const int mtp_cached_tokens = snapshot.mtp_cached_tokens[depth];
                if (!cache)
                {
                    if (mtp_cached_tokens >= 0)
                    {
                        return fail("logical checkpoint contains MTP tokens for a missing cache");
                    }
                    continue;
                }
                if (mtp_cached_tokens < 0 || mtp_cached_tokens > cache->max_seq_len())
                {
                    return fail("logical checkpoint MTP token count out of range for depth=" +
                                std::to_string(depth) +
                                " tokens=" + std::to_string(mtp_cached_tokens));
                }
                bool shifted_sequence_state_restored = false;
                if (state_.device_id.is_gpu())
                {
                    const DeviceKVSequenceStateCheckpoint *checkpoint =
                        shifted_sequence_checkpoints[depth];
                    std::string checkpoint_error;
                    submitted_async_device_mutation = true;
                    shifted_sequence_state_restored =
                        cache->restoreDeviceSequenceStateCheckpoint(
                            seq_idx,
                            checkpoint->data(),
                            checkpoint->bytes,
                            stream,
                            &checkpoint_error);
                    if (!shifted_sequence_state_restored)
                    {
                        return fail(
                            "MTP device sequence-state restore failed for depth=" +
                            std::to_string(depth) +
                            ": " + checkpoint_error);
                    }
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "live_prefix_checkpoint_device_sequence_state_restores",
                        1.0,
                        "decode",
                        state_.device_id.toString(),
                        {{"cache", "shifted"},
                         {"depth", std::to_string(depth)},
                         {"bytes", std::to_string(checkpoint->bytes)},
                         {"implementation", "device_kernel"}});
                }
                else
                {
                    shifted_sequence_state_restored =
                        cache->truncateSequence(
                            seq_idx,
                            mtp_cached_tokens,
                            stream);
                }
                if (!shifted_sequence_state_restored)
                {
                    return fail("MTP KV logical checkpoint truncate failed for depth=" +
                                std::to_string(depth) +
                                " tokens=" + std::to_string(mtp_cached_tokens));
                }

                const bool expects_hybrid_checkpoint =
                    liveCheckpointHasHybridState(
                        *cache,
                        state_.device_id,
                        mtp_cached_tokens);
                const PrefixBlockHandle *shifted_hybrid_handle =
                    shifted_hybrid_handles[depth];
                if (expects_hybrid_checkpoint && !shifted_hybrid_handle)
                {
                    return fail(
                        "logical checkpoint is missing shifted hybrid state for depth=" +
                        std::to_string(depth));
                }
                if (!expects_hybrid_checkpoint && shifted_hybrid_handle)
                {
                    return fail(
                        "logical checkpoint has unexpected shifted hybrid state for depth=" +
                        std::to_string(depth));
                }
                if (shifted_hybrid_handle &&
                    !importHybridPrefixPayload(
                        *cache,
                        *shifted_hybrid_handle,
                        seq_idx,
                        /*restore_for_suffix_prefill=*/false,
                        /*synchronize=*/false,
                        stream, HybridPrefixStatePurpose::LocalCheckpoint))
                {
                    return fail(
                        "logical checkpoint shifted hybrid payload import failed for depth=" +
                        std::to_string(depth));
                }
            }

            for (size_t index = 0; index < host_auxiliary.size(); ++index)
                if (host_auxiliary[index])
                {
                    IKVCache &cache = index == 0 ? *state_.kv_cache : *state_.mtp_kv_caches[index - 1];
                    host_auxiliary[index]->restore(*cache.auxiliaryState(), seq_idx);
                }

            state_.positions[seq_idx] = snapshot.cached_tokens;
            state_.sequence_lengths[seq_idx] = snapshot.cached_tokens;
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::PrefixRestore,
                "restore_logical_checkpoint");
            restore_completion.commit();
            submitted_async_device_mutation = false;
            if (!recordRestoredDeviceResidentMTPTransaction(
                    stream,
                    "restore_logical_checkpoint_shifted_kv"))
            {
                return fail("logical checkpoint restore could not publish restored shifted-MTP KV ownership");
            }
            return true;
        }

        if (state_.device_id.is_gpu())
        {
            submitted_async_device_mutation = true;
        }
        if (!state_.kv_cache->truncateSequence(seq_idx, 0, stream))
        {
            return fail("main KV payload checkpoint clear failed");
        }
        for (auto &cache : state_.mtp_kv_caches)
        {
            if (cache)
            {
                if (!cache->truncateSequence(seq_idx, 0, stream))
                {
                    return fail("MTP KV payload checkpoint clear failed");
                }
            }
        }

        if (snapshot.cached_tokens == 0)
        {
            if (!snapshot.mtp_blocks.empty())
            {
                return fail("zero-token snapshot unexpectedly contains MTP blocks");
            }
            if (const auto *hybrid = dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
                state_.device_id.is_gpu() && hybrid && hybrid->gdnLayerCount() > 0)
            {
                return fail("zero-token hybrid payload checkpoint restore requires streamful GDN reset");
            }
            if (!resetHybridPrefixPayloadState(
                    *state_.kv_cache,
                    IKVCache::StateResetContext{
                        .boundary =
                            IKVCache::StateResetBoundary::PrefixReplacement,
                        .execution_stream = stream,
                        .reason = "restore-payload-checkpoint-zero",
                    }))
            {
                return fail("zero-token payload checkpoint GDN reset failed");
            }
            auto reset_auxiliary = [&](IKVCache &cache)
            {
                if (auto *owner = cache.auxiliaryState())
                    owner->reset({cache.first_layer_index(), cache.n_layers(), seq_idx, 1}, stream);
            };
            reset_auxiliary(*state_.kv_cache);
            for (auto &cache : state_.mtp_kv_caches) if (cache) reset_auxiliary(*cache);
            state_.positions[seq_idx] = 0;
            state_.sequence_lengths[seq_idx] = 0;
            handleLivePrefixReplayStateAfterMutation(
                LivePrefixMutationReason::PrefixRestore,
                "restore_payload_checkpoint_zero");
            restore_completion.commit();
            submitted_async_device_mutation = false;
            if (!recordRestoredDeviceResidentMTPTransaction(
                    stream,
                    "restore_payload_checkpoint_zero_shifted_kv"))
            {
                return fail("zero-token payload checkpoint restore could not publish restored shifted-MTP KV ownership");
            }
            return true;
        }

        for (const auto &handle : snapshot.blocks)
        {
            if (!handle.valid() || !handle.kvKData() || !handle.kvVData())
            {
                return fail("main KV block handle is invalid");
            }

            PrefixPayloadLayout expected = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                handle.key.token_count,
                handle.layout.terminal_hidden_bytes,
                handle.layout.terminal_logits_bytes);
            applyLocalHybridCheckpointExtent(expected, *state_.kv_cache);
            if (!handle.layout.compatiblePayloadShape(expected))
            {
                return fail("main KV block layout is incompatible");
            }

            for (int local_layer = 0; local_layer < handle.layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("main KV block layer index is invalid");
                }
                const uint8_t *k_src = handle.kvKData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_k;
                const uint8_t *v_src = handle.kvVData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                if (!state_.kv_cache->importLogicalBlock(desc, k_src, v_src))
                {
                    return fail("main KV logical block import failed for layer=" +
                                std::to_string(global_layer) +
                                " tokens=" + std::to_string(handle.key.token_count));
                }
            }
            importPrefixAuxiliaryState(*state_.kv_cache, seq_idx, PrefixAuxiliaryLane::Main, handle, stream);
        }

        if (!snapshot.blocks.empty() &&
            !importHybridPrefixPayload(
                *state_.kv_cache,
                snapshot.blocks.back(),
                seq_idx,
                /*restore_for_suffix_prefill=*/false,
                /*synchronize=*/true,
                stream, HybridPrefixStatePurpose::LocalCheckpoint))
        {
            return fail("hybrid payload import failed");
        }

        if (!snapshot.blocks.empty())
        {
            const PrefixBlockHandle &terminal_handle = snapshot.blocks.back();
            if (terminal_handle.layout.includes_terminal_hidden &&
                terminal_handle.layout.terminal_hidden_bytes > 0)
            {
                if (!restoreLiveCheckpointTerminalHidden(
                        terminal_handle,
                        stream,
                        "restore_payload_checkpoint",
                        /*publish_mailbox=*/false,
                        /*destination_row=*/seq_idx))
                {
                    return fail(
                        "payload checkpoint terminal hidden restore failed");
                }
            }
        }

        for (const auto &handle : snapshot.mtp_blocks)
        {
            if (!handle.valid() || !handle.kvKData() || !handle.kvVData())
            {
                return fail("MTP KV block handle is invalid");
            }

            const int depth = handle.key.block_index;
            if (depth < 0 || depth >= static_cast<int>(state_.mtp_kv_caches.size()))
            {
                return fail("MTP KV block depth out of range: depth=" +
                            std::to_string(depth) +
                            " caches=" + std::to_string(state_.mtp_kv_caches.size()));
            }
            auto &cache = state_.mtp_kv_caches[static_cast<size_t>(depth)];
            if (!cache || handle.key.token_count < 0 || handle.key.token_count > cache->max_seq_len())
            {
                return fail("MTP KV block token count out of range: tokens=" +
                            std::to_string(handle.key.token_count));
            }

            PrefixPayloadLayout expected = buildDensePrefixPayloadLayout(
                *cache,
                state_.device_id,
                handle.key.token_count);
            if (!handle.layout.compatiblePayloadShape(expected))
            {
                return fail("MTP KV block layout is incompatible");
            }

            for (int local_layer = 0; local_layer < handle.layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("MTP KV block layer index is invalid");
                }
                const uint8_t *k_src = handle.kvKData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_k;
                const uint8_t *v_src = handle.kvVData() +
                                       static_cast<size_t>(local_layer) * handle.layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                if (!cache->importLogicalBlock(desc, k_src, v_src))
                {
                    return fail("MTP KV logical block import failed for layer=" +
                                std::to_string(global_layer) +
                                " tokens=" + std::to_string(handle.key.token_count));
                }
            }
            importPrefixAuxiliaryState(*cache, seq_idx, PrefixAuxiliaryLane::Main, handle, stream);
        }

        state_.positions[seq_idx] = snapshot.cached_tokens;
        state_.sequence_lengths[seq_idx] = snapshot.cached_tokens;
        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::PrefixRestore,
            "restore_payload_checkpoint");
        restore_completion.commit();
        submitted_async_device_mutation = false;
        if (!recordRestoredDeviceResidentMTPTransaction(
                stream,
                "restore_payload_checkpoint_shifted_kv"))
        {
            return fail("payload checkpoint restore could not publish restored shifted-MTP KV ownership");
        }
        return true;
    }

    bool DeviceGraphOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        if (!state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return false;
        }
        void *stream = explicitGPUStreamForOperation("truncateLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return false;
        }
        if (!waitForPendingRequestStateReset(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "truncate_live_prefix_request_state_reset"))
        {
            return false;
        }
        if (!waitForPendingLivePrefixCheckpointReady(
                stream,
                "truncate_live_prefix_source"))
        {
            return false;
        }
        if (!joinPublishedLiveStateHandoffs(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "truncate_live_prefix_previous_mutation"))
        {
            return false;
        }
        if (!waitForPendingLiveGraphProducersBeforePrefixMutation(
                stream,
                "truncate_live_prefix"))
        {
            return false;
        }
        if (!prepareLivePrefixMutationReadyEvent(
                stream,
                "truncate_live_prefix"))
        {
            return false;
        }
        if (!state_.kv_cache->truncateSequence(seq_idx, cached_tokens, stream))
        {
            return false;
        }
        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }
            const int shifted_count=mtpShiftedPrefixRows(cached_tokens,cache->stateOwnership().shifted_origin,static_cast<int>(depth));
            if(shifted_count<0 || (cache->stateOwnership().shifted_origin==MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero &&
                shifted_count>cache->max_seq_len()))
                throw std::invalid_argument("QFN prefix truncate must fit its exact model-owned cache origin/depth/capacity");
            const int bounded_count = std::min(shifted_count, cache->max_seq_len());
            if (!cache->truncateSequence(seq_idx, bounded_count, stream))
            {
                return false;
            }
        }
        state_.positions[seq_idx] = cached_tokens;
        state_.sequence_lengths[seq_idx] = cached_tokens;
        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::PrefixTruncate,
            "truncate_live_prefix");
        if (!recordLivePrefixMutationReady(
                stream,
                "truncate_live_prefix"))
        {
            return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::mtpSpecStatePublicationRequiresCapturedStage() const
    {
        if (!graph_builder_ || !graph_builder_->config().mtp.enabled || !state_.kv_cache)
        {
            return false;
        }

        auto *hybrid = dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
        if (!hybrid)
        {
            return false;
        }

        // The cache's compressed layer map already owns the local count. Do
        // not reconstruct it by querying ambiguous local/global layer IDs.
        return hybrid->gdnLayerCount() > 0;
    }

    bool DeviceGraphOrchestrator::requiresMTPDecodeEquivalentVerifierReplay() const
    {
        return mtpSpecStatePublicationRequiresCapturedStage() &&
               !supportsMTPSpecStatePublication();
    }

    int DeviceGraphOrchestrator::sampleGreedyOnDevice()
    {
        TensorBase *main_logits = requireCurrentMainLogits(
            DeviceLogitsSource::Main, "sampleGreedyOnDevice");

        // The producer, not the allocated weight/shard geometry, owns the row.
        // CPU GlobalTP forward and prefix restore publish a gathered full row;
        // a local projection may publish a shard. Using the dormant local
        // allocation would add a collective that followers never enter.
        const bool column_parallel =
            current_main_logits_publication_.descriptor
                .isColumnParallelStorage();
        if (column_parallel)
        {
            const int token_offset = vocabOffsetForTPConfig(graph_builder_->config());
            void *stream = nullptr;
            auto device_opt = main_logits->current_device();
            if (device_opt.has_value() && device_opt->is_gpu())
            {
                stream = prepareMainLogitsDeviceConsumer(
                    "sampleGreedyOnDeviceLocal",
                    MainLogitsHandoffMode::Consume);
                if (!stream)
                {
                    return -1;
                }
            }
            return coordinateGreedyCandidate(
                sampleGreedyCandidateFromTensor(main_logits,
                                                0,
                                                token_offset,
                                                argmax_partial_vals_dev_,
                                                argmax_partial_idxs_dev_,
                                                argmax_partial_capacity_,
                                                stream,
                                                "logits_local"),
                globalTPContextForMTPCoordination());
        }

        void *stream = nullptr;
        auto device_opt = main_logits->current_device();
        if (device_opt.has_value() && device_opt->is_gpu())
        {
            stream = prepareMainLogitsDeviceConsumer(
                "sampleGreedyOnDevice",
                MainLogitsHandoffMode::Consume);
            if (!stream)
            {
                return -1;
            }
        }
        /*
         * `state_.logits` is already the complete vocabulary row. In CPU
         * GlobalTP/NodeTP the graph's terminal allgather produced that row on
         * every participant, so only the coordinated root samples it. Entering
         * candidate allgather here would disagree with
         * requiresMPICoordinatedDecodeSampling(), whose worker correctly skips
         * sampling, and pair the root's allgather with the worker's later
         * post-sampling barrier. Only the column-parallel branch above owns a
         * sampling-time collective.
         */
        const GreedyLogitCandidate replicated_candidate =
            sampleGreedyCandidateFromTensor(main_logits,
                                            0,
                                            0,
                                            argmax_partial_vals_dev_,
                                            argmax_partial_idxs_dev_,
                                            argmax_partial_capacity_,
                                            stream,
                                            "logits");
        return replicated_candidate.valid ? replicated_candidate.token : -1;
    }

    int DeviceGraphOrchestrator::sampleOnDevice(const SamplingParams &params)
    {
        return sampleOnDeviceAtLogicalPosition(params, get_position());
    }

    int DeviceGraphOrchestrator::sampleOnDeviceAtLogicalPosition(
        const SamplingParams &params,
        int logical_position)
    {
        if (params.is_greedy())
        {
            return sampleGreedyOnDevice();
        }
        if (!state_.device_id.is_gpu())
        {
            return -1;
        }
        TensorBase *main_logits = requireCurrentMainLogits(
            DeviceLogitsSource::Main,
            "sampleOnDeviceAtLogicalPosition");
        if (current_main_logits_publication_.descriptor
                .isColumnParallelStorage())
        {
            return -1;
        }
        if (params.top_k <= 0 || params.top_k > 256)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic sampling requires 1 <= top_k <= 256; got top_k="
                      << params.top_k);
            return -1;
        }
        if (!main_logits->deviceValid())
        {
            return -1;
        }

        void *gpu_ptr = main_logits->gpu_data_ptr();
        if (!gpu_ptr)
        {
            return -1;
        }

        const auto &shape = main_logits->shape();
        if (shape.empty())
        {
            return -1;
        }
        const size_t rows = shape.size() >= 2 ? shape[0] : 1;
        const size_t cols = shape.size() >= 2 ? shape[1] : shape[0];
        if (rows < 1 || cols == 0 || cols > static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return -1;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            return -1;
        }
        void *stream = prepareMainLogitsDeviceConsumer(
            "sampleOnDevice",
            MainLogitsHandoffMode::Consume);
        if (!stream)
        {
            return -1;
        }

        int token = -1;
        const bool seeded = params.seed != 0;
        const uint64_t seed =
            seeded ? static_cast<uint64_t>(params.seed)
                   : (0xD1B54A32D192ED03ull ^
                      (session_epoch_ * 0x9E3779B97F4A7C15ull));
        const uint64_t offset =
            seeded
                ? static_cast<uint64_t>(std::max(0, logical_position)) *
                      sampling_math::kMTPSpecDrawPurposesPerToken
                : device_sampling_counter_++;
        const bool ok = backend->sampleTopKTopPF32(
            static_cast<const float *>(gpu_ptr),
            static_cast<int>(cols),
            params.top_k,
            params.top_p,
            params.temperature,
            seed,
            offset,
            state_.device_id.gpu_ordinal(),
            &token,
            stream);
        if (!ok)
        {
            LOG_DEBUG("[DeviceGraphOrchestrator] GPU stochastic sampling failed on "
                      << state_.device_id.toString());
            return -1;
        }
        return token;
    }

    bool DeviceGraphOrchestrator::requiresMPICoordinatedDecodeSampling(
        const SamplingParams &params) const
    {
        /*
         * This is the same published-layout decision as the greedy consumer,
         * not a CPU/GPU capability guess. A full row needs no candidate gather;
         * a distributed shard requires every peer, including when each peer
         * applies its own portion of the root's penalty map first. CPU
         * stochastic sampling uses the already-gathered host distribution and
         * does not enter this greedy candidate protocol.
         */
        if (!params.is_greedy())
        {
            return false;
        }
        (void)requireCurrentMainLogits(
            DeviceLogitsSource::Main, "requiresMPICoordinatedDecodeSampling");
        if (!current_main_logits_publication_.descriptor.isColumnParallelStorage())
        {
            return false;
        }
        IGlobalTPContext *global_ctx = globalTPContextForMTPCoordination();
        return global_ctx && global_ctx->degree() > 1;
    }

    bool DeviceGraphOrchestrator::publishDeviceGenerationStateReady(
        void *producer_stream,
        int request_count,
        DeviceGenerationStatePublicationKind kind)
    {
        if (!state_.device_id.is_gpu())
            return true;

        const char *const producer_name = [&]() -> const char *
        {
            switch (kind)
            {
            case DeviceGenerationStatePublicationKind::Admission:
                return "device_generation_admission";
            case DeviceGenerationStatePublicationKind::CommittedTransaction:
                return "generation_response_and_state_commit";
            case DeviceGenerationStatePublicationKind::Terminal:
                return "device_generation_parent_terminal";
            }
            return "invalid_device_generation_publication";
        }();
        const char *const verifier_identity_state =
            kind == DeviceGenerationStatePublicationKind::Admission
                ? "reset"
                : "committed";

        auto &ready = device_generation_state_ready_;
        if (!producer_stream || request_count <= 0 ||
            request_count != device_generation_storage_.active_request_count ||
            !device_generation_storage_.validFor(request_count) ||
            !ready.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation state publication has invalid lifecycle ownership"
                      << " producer="
                      << (producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown")
                      << " request_count=" << request_count
                      << " active_request_count="
                      << device_generation_storage_.active_request_count
                      << " event=" << static_cast<bool>(ready.event)
                      << " handoff_phase="
                      << static_cast<int>(ready.handoff.phase()));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !ready.handoff.publish(
                kind,
                producer_stream,
                request_count,
                [&]() -> bool
                {
                    if (kind == DeviceGenerationStatePublicationKind::Terminal &&
                        active_device_generation_admission_ &&
                        active_device_generation_admission_->depth_policy.isOrdinary() &&
                        !recordDeviceResidentLogicalSequenceStateMailbox(request_count, producer_stream,
                            DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTerminal))
                        return false;
                    return publishDeviceGenerationArenaState(
                               producer_stream,
                               producer_name) &&
                           mtp_committed_verifier_identity_dev_ &&
                           publishPreparedArenaGraphInput(
                               BufferId::MTP_COMMITTED_VERIFIER_IDENTITY,
                               mtp_committed_verifier_identity_dev_,
                               producer_stream,
                               state_.device_id,
                               producer_name) &&
                           DeviceEventEdge::at(
                               DeviceTimelinePoint::DeviceGenerationStateReady)
                               .from(
                                   DeviceTimelineRole::DeviceGenerationController)
                               .publish(
                                   *backend,
                                   state_.device_id,
                                   ready.event.get(),
                                   producer_stream);
                }))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish device-generation controller readiness"
                      << " kind=" << static_cast<int>(kind)
                      << " handoff_phase="
                      << static_cast<int>(ready.handoff.phase()));
            return false;
        }

        if (kind == DeviceGenerationStatePublicationKind::Terminal)
        {
            if (const auto& resources=deviceGenerationLoopGraph().read_ahead_resources;
                resources && resources->hasOutstandingSubmission())
            {
                // The original handoff above is the sole event producer. This
                // unforgeable capability records its exact successful original
                // event/controlblock/stream/submission, with no second record.
                try {
                    resources->acceptTerminalPublication(
                        ReadAheadNativeResourceFamily::PublishedTerminal(resources,ready.event,producer_stream));
                } catch (...) { std::terminate(); } // The original terminal is already enqueued.
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_state_publications",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer", producer_name},
             {"request_count", std::to_string(request_count)},
             {"verifier_identity", verifier_identity_state},
             {"ordering", "event_wait"}});
        return true;
    }

    bool DeviceGraphOrchestrator::consumeDeviceGenerationStateReady(
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        int expected_request_count,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &ready = device_generation_state_ready_;
        if (!consumer_stream || expected_request_count <= 0 ||
            !ready.event ||
            device_generation_storage_.active_request_count !=
                expected_request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation state consumption has invalid lifecycle ownership"
                      << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown")
                      << " expected_requests=" << expected_request_count
                      << " published_requests="
                      << ready.handoff.requestCount()
                      << " handoff_phase="
                      << static_cast<int>(ready.handoff.phase()));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !ready.handoff.borrow(
                consumer_stream,
                consumer_role,
                expected_request_count,
                [&](void *producer, void *consumer) -> bool
                {
                    return DeviceEventEdge::at(
                               DeviceTimelinePoint::
                                   DeviceGenerationStateReady)
                        .from(
                            DeviceTimelineRole::
                                DeviceGenerationController)
                        .to(consumer_role)
                        .enqueueWait(
                            *backend,
                            state_.device_id,
                            ready.event.get(),
                            producer,
                            consumer);
                }))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to consume device-generation controller readiness"
                      << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        return true;
    }

    bool DeviceGraphOrchestrator::
        retireDeviceGenerationStateForRequestReset(
            void *reset_stream,
            int expected_request_count,
            const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &ready = device_generation_state_ready_;
        IBackend *const backend = getBackendFor(state_.device_id);
        if (!backend || !ready.event || !reset_stream ||
            expected_request_count <= 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-generation reset retirement "
                "has incomplete event or stream ownership"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown")
                << " expected_requests=" << expected_request_count
                << " handoff_phase="
                << static_cast<int>(ready.handoff.phase()));
            return false;
        }

        const auto phase_before = ready.handoff.phase();
        const auto borrower_before = ready.handoff.borrowerRole();
        if (!ready.handoff.retireForReset(
                reset_stream,
                expected_request_count,
                [&](DeviceGenerationStateHandoffPhase phase,
                    void *frontier_stream,
                    DeviceTimelineRole,
                    void *consumer_stream) -> bool
                {
                    const auto edge =
                        DeviceEventEdge::at(
                            DeviceTimelinePoint::
                                DeviceGenerationStateReady)
                            .from(
                                DeviceTimelineRole::
                                    DeviceGenerationController);
                    if (phase ==
                            DeviceGenerationStateHandoffPhase::Borrowed &&
                        !edge.publish(
                            *backend,
                            state_.device_id,
                            ready.event.get(),
                            frontier_stream))
                    {
                        return false;
                    }
                    return edge.to(DeviceTimelineRole::RequestStateReset)
                        .enqueueWait(
                            *backend,
                            state_.device_id,
                            ready.event.get(),
                            frontier_stream,
                            consumer_stream);
                }))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Device-generation reset could not "
                "retire its exact controller frontier"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown")
                << " expected_requests=" << expected_request_count
                << " handoff_phase="
                << static_cast<int>(phase_before)
                << " borrower_role="
                << deviceTimelineRoleName(borrower_before));
            return false;
        }

        PerfStatsCollector::addCounter(
            "request_reset",
            "device_generation_frontier_retirements",
            1.0,
            "request_reset",
            state_.device_id.toString(),
            {{"phase",
              phase_before == DeviceGenerationStateHandoffPhase::Published
                  ? "published"
                  : "borrowed"},
             {"borrower_role",
              std::string(deviceTimelineRoleName(borrower_before))},
             {"ordering", "event_wait"}});
        return true;
    }

    DeviceGenerationExecutionPolicy
    DeviceGraphOrchestrator::declaredDeviceGenerationExecutionPolicy() const noexcept
    {
        if (!state_.device_id.is_gpu() || !graph_builder_)
            return DeviceGenerationExecutionPolicy::Unsupported;
        DeviceGenerationDomainDeclaration declaration;
        declaration.observe(state_.device_id.type);
        if (hasHeterogeneousCollectiveExecutionDomain())
            declaration.requireHostedBoundary();
        if (pp_stage_config_)
        {
            switch (pp_stage_config_->generation_domain)
            {
            case DeviceGenerationExecutionDomain::NativeDomain:
                break;
            case DeviceGenerationExecutionDomain::HostedBoundary:
                declaration.requireHostedBoundary();
                break;
            default:
                return DeviceGenerationExecutionPolicy::Unsupported;
            }
        }
        // The immutable execution plan is available before capture. A later
        // coordinator binding cannot retroactively enlarge the admitted family.
        const auto &execution = graph_builder_->config().moe.expert_overlay_execution_plan;
        if (execution && execution->generationExecutionDomain() ==
                DeviceGenerationExecutionDomain::HostedBoundary)
            declaration.requireHostedBoundary();
        return llaminar2::declaredDeviceGenerationExecutionPolicy(
            state_.device_id.type, declaration.domain());
    }

    DeviceGenerationExecutionPolicy
    DeviceGraphOrchestrator::deviceGenerationExecutionPolicy(
        DeviceGenerationLoopTopology topology) const noexcept
    {
        const auto declared = declaredDeviceGenerationExecutionPolicy();
        if (declared == DeviceGenerationExecutionPolicy::Unsupported)
            return declared;
        const auto &capture = deviceGenerationLoopGraph().capture;
        if (!capture)
            return DeviceGenerationExecutionPolicy::Unsupported;
        if (declared == DeviceGenerationExecutionPolicy::NativeConditionalGraph)
        {
            // Capabilities validate the admitted architecture; they never
            // select a differently sized hosted graph family on failure.
            if (!capture->supportsDeviceControlledWhileLoop() ||
                (topology == DeviceGenerationLoopTopology::DynamicDepth &&
                 !capture->supportsDeviceControlledSelectorWhileLoop()))
                return DeviceGenerationExecutionPolicy::Unsupported;
        }
        switch (topology)
        {
        case DeviceGenerationLoopTopology::FixedDepth:
        case DeviceGenerationLoopTopology::DynamicDepth:
            return declared;
        }
        return DeviceGenerationExecutionPolicy::Unsupported;
    }

    bool DeviceGraphOrchestrator::beginDeviceResidentGeneration(
        const DeviceGenerationAdmissionRequest &request)
    {
        if (!state_.device_id.is_gpu())
            return true;
        const auto &depth_policy = request.depth_policy;
        if (mtp_depth_transaction_trace_dev_ &&
            (!sampling_math::validMTPDepthTransactionTraceAdmission(request.kind, request.request_count,
                request.max_new_tokens) || depth_policy.isOrdinary() || depth_policy.isForwardOnly()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Depth transaction trace requires fresh scalar speculative n<=256 admission");
            return false;
        }
        const bool speculative = !depth_policy.isOrdinary() &&
                                 !depth_policy.isForwardOnly();
        if (!graph_builder_ ||
            (speculative && !retainsMTPGraphCapacity(graph_builder_->config().mtp)))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation admission requires an initialized graph owner and retained capacity for speculative algorithms");
            return false;
        }
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!request.valid() ||
            session_epoch_ == 0 || generation == 0 ||
            !depth_policy.valid() ||
            depth_policy.maximum_depth > mtp_max_draft_depth_ ||
            !device_generation_storage_.validFor(request.request_count) ||
            request.max_new_tokens >
                device_generation_storage_.response_token_stride)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Invalid device-resident generation admission"
                      << " requests=" << request.request_count
                      << " max_new_tokens=" << request.max_new_tokens
                      << " leading_row_disposition="
                      << static_cast<int>(
                             request.initial_leading_row_disposition)
                      << " session_epoch=" << session_epoch_
                      << " workspace_generation=" << generation
                      << " admitted_depth_range=["
                      << depth_policy.minimum_depth << ','
                      << depth_policy.maximum_depth << ']'
                      << " retained_max_depth=" << mtp_max_draft_depth_
                      << " response_capacity="
                      << device_generation_storage_.response_token_stride);
            return false;
        }

        if(speculative){
            const auto source=graph_builder_->retainedStorageBoundary(state_.device_id);
            if(source && source->retainedParticipantSource())
                qwen4exp::PooledImmutableParticipantExecutionPolicy::requireAdmission(depth_policy, graph_builder_->config().mtp);
        }

        IBackend *backend = getBackendFor(state_.device_id);
        void *stream = explicitGPUStreamForOperation(
            "beginDeviceResidentGeneration");
        if (!backend || !stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-resident generation admission requires an explicit backend stream");
            return false;
        }

        if (device_generation_storage_.active_request_count != 0 ||
            active_device_generation_admission_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation admission attempted before the prior request crossed reset"
                      << " active_requests="
                      << device_generation_storage_.active_request_count
                      << " active_policy="
                      << active_device_generation_admission_.has_value());
            return false;
        }
        if (sparseQueryParentLaunched())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation admission found an unretired loop launch");
            return false;
        }
        if (!device_generation_state_ready_.handoff.inactive())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation admission found an unretired controller handoff without an active request"
                      << " phase="
                      << static_cast<int>(
                             device_generation_state_ready_
                                 .handoff.phase()));
            return false;
        }
        if (!hosted_device_generation_cursor_.inactive() ||
            !deviceGenerationLoopGraph().hosted_advance.idle())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation admission found an unretired hosted scheduler lifecycle");
            return false;
        }

        // A fresh admission can directly follow request reset, without a
        // synthetic forward consuming that reset's event first. Join the
        // existing reset authority before overwriting any resident bank.
        if (!waitForPendingRequestStateReset(
                stream, DeviceTimelineRole::RequestAdmissionTransfer,
                "beginDeviceResidentGeneration"))
            return false;

        admitSparseQueryProgram(request);

        const auto forward_failure_policy=graph_builder_->deviceGenerationForwardFailurePolicy();
        using ForwardFailurePolicy=sampling_math::DeviceGenerationForwardFailurePolicy;
        if (forward_failure_policy!=ForwardFailurePolicy::ResetAtResponseAdmission &&
            forward_failure_policy!=ForwardFailurePolicy::PreserveUntilRequestReset)
            return false;
        // The native initializer must observe the last real prefill/forward's
        // error writes before deciding admission. This is an event join only;
        // prefix hits with no new forward have no publication to join.
        if (forward_failure_policy==ForwardFailurePolicy::PreserveUntilRequestReset &&
            !waitForForwardGraphOutputReady(stream,DeviceTimelineRole::DeviceGenerationController,
                ForwardGraphOutputKind::Any,"forward_failure_response_admission"))
            return false;

        // Prefix-hit admission may have no new prompt-input upload. Publish
        // the same immutable request controls here as at ordinary prefill;
        // the existing session epoch makes this once-only for either route.
        if (!publishMTPRequestStopTokensOnDevice(stream, "device_generation_admission") ||
            !publishMTPRequestPenaltyPolicyOnDevice(stream, "device_generation_admission"))
            return false;

        // Seed metadata is admitted once, never embedded as a graph scalar.
        // TransferEngine retains the tensor's host source through the exact
        // upload event. The existing reset edge retires all prior GPU readers;
        // no caller-owned vector or stack temporary can outlive an async copy.
        const auto seed_tensor = std::dynamic_pointer_cast<INT32Tensor>(
            arena_->getSharedTensor(BufferId::SAMPLING_REQUEST_SEEDS));
        const GenerationRequestSeedGeometry seed_geometry(
            state_.batch_size, stochastic_batch_output_request_capacity_);
        if (!seed_tensor || seed_tensor->numel() !=
                seed_geometry.requests() * GenerationRequestSeedGeometry::words_per_request ||
            static_cast<size_t>(request.request_count) > seed_geometry.requests() ||
            !seed_tensor->gpu_data_ptr() ||
            reinterpret_cast<uintptr_t>(seed_tensor->gpu_data_ptr()) % alignof(uint64_t) != 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Generation seed bank does not match admitted request capacity");
            return false;
        }
        try
        {
            // Clear inactive rows too: a narrower request must not retain a
            // previously admitted seed which an invalid consumer could reuse.
            // Mutable tensor access joins only an outstanding host-source copy,
            // never a stream/device-wide fence or a download of live GPU state.
            auto *const seed_words = seed_tensor->raw_mutable_data();
            std::memset(seed_words, 0, seed_geometry.bytes());
            if (request.sampling_seeds)
            {
                const auto seeds = request.sampling_seeds->values();
                std::memcpy(seed_words, seeds.data(), seeds.size_bytes());
            }
            TransferEngine::publishHostWrite(seed_tensor);
            TransferEngine::prepareDeviceInput(seed_tensor.get(), state_.device_id, stream);
            if (!publishPreparedArenaGraphInput(BufferId::SAMPLING_REQUEST_SEEDS,
                    seed_tensor->gpu_data_ptr(), stream, state_.device_id,
                    "generation_request_seed_admission"))
                return false;
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Generation seed publication failed: " << error.what());
            return false;
        }

        if (depth_policy.isOrdinary() && request.initial_leading_row_disposition ==
                sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse)
        {
            // Prefill and full-prefix restore expose logits, not a sampled
            // condition. Snapshot canonical device KV positions directly into
            // the existing logical frontier; no dummy forward or host count
            // is needed to manufacture a valid first sampling boundary.
            auto &rows = device_resident_logical_sequence_state_storage_;
            if (!state_.kv_cache || !rows.validFor(request.request_count))
                return false;
            for (int row = 0; row < request.request_count; ++row)
                if (!state_.kv_cache->deviceSequenceCachedTokenCountPtr(row))
                    return false;
            if (!waitForLiveInferenceStateReadyForObservation(stream,
                    "ordinary_unsampled_frontier_admission", DeviceTimelineRole::DeviceGenerationController) ||
                !waitForDeviceResidentLogicalSequenceStateRowReuse(stream,
                    "ordinary_unsampled_frontier_admission"))
                return false;
            // IKVCache owns each pointer and does not promise adjacent rows.
            // Resolve each request at admission only; never infer a cross-row
            // stride or add host dispatch inside the retained generation loop.
            for (int row = 0; row < request.request_count; ++row)
                if (!backend->enqueueInitializeGenerationLogicalState({
                        .source = GenerationInitialFrontier::UnsampledLogits,
                        .positions = state_.kv_cache->deviceSequenceCachedTokenCountPtr(row),
                        .request_count = 1,
                        .output = {
                            .base_cached_tokens = rows.initialization_base_cached_tokens_scratch_device + row,
                            .target_positions = rows.target_cached_tokens_device + row,
                            .accepted_state_counts = rows.accepted_state_counts_device + row,
                            .next_condition_tokens = rows.next_condition_tokens_device + row,
                            .all_drafts_accepted_flags = rows.all_drafts_accepted_flags_device + row,
                            .stopped_flags = rows.stopped_flags_device + row,
                            .publication_ok_flags = rows.publication_ok_flags_device + row}},
                        state_.device_id.gpu_ordinal(), stream))
                    return false;
            std::string error;
            if (!recordDeviceResidentLogicalSequenceStateMailbox(request.request_count, stream,
                    DeviceResidentLogicalStatePublicationKind::UnsampledLogitsInitialization, &error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Unsampled generation frontier publication failed: " << error);
                return false;
            }
        }

        /*
         * Request admission mutates contents behind stable device addresses; it
         * does not replace graph identity. Seed, sampling policy, geometry, and
         * pointer bindings are authenticated by the child caches during parent
         * materialization. A changed child invalidates and rebuilds the parent,
         * while an identical request reuses the proven executable.
         *
         * The committed-verifier identity is request-owned diagnostic evidence,
         * so admission invalidates it on this exact admission stream before the
         * controller is published. This asynchronous reset neither blocks nor
         * copies state to the host; the following state-ready event orders every
         * consumer behind both the reset and controller initialization.
         */

        device_generation_storage_.active_request_count =
            request.request_count;
        active_device_generation_admission_ = request;
        if ((speculative && (!mtp_committed_verifier_identity_dev_ ||
             !backend->memset(
                mtp_committed_verifier_identity_dev_,
                0,
                static_cast<size_t>(request.request_count) *
                    sizeof(
                        sampling_math::
                            MTPCommittedVerifierIdentityRecord),
                state_.device_id.gpu_ordinal(),
                stream))) ||
            !backend->enqueueInitializeDeviceGeneration(
                request.request_count,
                request.max_new_tokens,
                depth_policy,
                request.initial_leading_row_disposition,
                device_generation_storage_.response_token_stride,
                device_generation_storage_.response_tokens_device,
                device_generation_storage_.control_stride,
                device_generation_storage_.control_device,
                state_.device_id.gpu_ordinal(),
                stream,
                {.kind = request.kind,
                 .session_epoch = session_epoch_,
                 .workspace_generation = generation,
                 .prior_tickets = device_generation_storage_.dispatch_tickets_device,
                 .forward_failure_policy = forward_failure_policy}) ||
            !backend->enqueueInitializeDeviceGenerationDispatchTicket(
                session_epoch_,
                generation,
                device_generation_storage_.control_device,
                device_generation_storage_.control_stride,
                request.request_count,
                device_generation_storage_.dispatch_tickets_device,
                state_.device_id.gpu_ordinal(),
                stream) ||
            !beginRetainedStorageAdmission(stream) ||
            !publishDeviceGenerationStateReady(
                stream,
                request.request_count,
                DeviceGenerationStatePublicationKind::Admission))
        {
            device_generation_storage_.active_request_count = 0;
            active_device_generation_admission_.reset();
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to initialize and publish the device-generation controller");
            return false;
        }

        last_device_generation_dispatch_ticket_.reset();
        hosted_device_generation_cursor_.reset();

        if (speculative)
        {
            // Admission is participant-local even when a rank coordinates
            // dispatch. Publish the policy beside the authenticated controller
            // initialization on every device, before any draft can consume it.
            const auto topology = depth_policy.mode == sampling_math::DeviceGenerationPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth : DeviceGenerationLoopTopology::FixedDepth;
            PerfStatsCollector::addCounter(
                "mtp", "device_generation_execution_policy_selections", 1.0,
                "decode", state_.device_id.toString(),
                {{"policy", deviceGenerationExecutionPolicyName(deviceGenerationExecutionPolicy(topology))},
                 {"topology", topology == DeviceGenerationLoopTopology::DynamicDepth ? "dynamic_depth" : "fixed_depth"},
                 {"selection_boundary", "pre_first_draft"}});
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_admissions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"sparse_query_program", sparseQueryGraphProgram() == SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
             {"request_count", std::to_string(request.request_count)},
             {"admission_boundary", request.kind == sampling_math::DeviceGenerationAdmissionKind::ContinueResponse
                 ? "continue_response" : "new_request"},
             {"max_new_tokens", std::to_string(request.max_new_tokens)},
             {"minimum_draft_depth",
              std::to_string(depth_policy.minimum_depth)},
             {"maximum_draft_depth",
              std::to_string(depth_policy.maximum_depth)},
             {"depth_policy_mode",
              std::to_string(static_cast<int>(depth_policy.mode))},
             {"initial_leading_committed_output_count",
              std::to_string(
                  sampling_math::
                      device_generation_leading_committed_output_count(
                          request.initial_leading_row_disposition))},
             {"response_owner", "persistent_device_ledger"}});
        return true;
    }

    bool DeviceGraphOrchestrator::materializeDeviceResidentGeneration(
        int request_count,
        int draft_depth,
        DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling_mode)
    {
        if (!state_.device_id.is_gpu() || request_count <= 0 ||
            draft_depth < 0 ||
            !isValidDeviceGenerationSamplingMode(sampling_mode) ||
            device_generation_storage_.active_request_count != request_count ||
            !device_generation_storage_.validFor(request_count) ||
            !device_generation_state_ready_.handoff.published() ||
            device_generation_state_ready_.handoff.requestCount() !=
                request_count)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation parent preparation requires one committed admitted transaction"
                      << " device=" << state_.device_id.toString()
                      << " requests=" << request_count
                      << " active_requests="
                      << device_generation_storage_.active_request_count
                      << " draft_depth=" << draft_depth
                      << " sampling_mode="
                      << deviceGenerationSamplingModeName(sampling_mode)
                      << " controller_handoff_phase="
                      << static_cast<int>(
                             device_generation_state_ready_
                                 .handoff.phase()));
            return false;
        }

        if (!active_device_generation_admission_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation materialization has no admitted depth policy");
            return false;
        }
        const auto &configured_depth_policy =
            active_device_generation_admission_->depth_policy;
        const DeviceGenerationLoopTopology configured_topology =
            configured_depth_policy.mode ==
                    sampling_math::DeviceGenerationPolicyMode::Dynamic
                ? DeviceGenerationLoopTopology::DynamicDepth
                : DeviceGenerationLoopTopology::FixedDepth;
        if (topology != configured_topology)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation materialization topology disagrees with the resident depth controller"
                      << " requested=" << static_cast<int>(topology)
                      << " configured="
                      << static_cast<int>(configured_topology));
            return false;
        }

        /*
         * Both ordinary and speculative parents place maintenance after each
         * committed transaction. The
         * externally orchestrated first transaction therefore crosses the same
         * boundary here, on each participant's persistent worker, before
         * composition. Besides preserving exact once-only cadence semantics,
         * that boundary is the atomic first-use operation which leaves the
         * maintenance child replay-ready. Refusing to compose without its
         * outstanding event makes it impossible to substitute an uncaptured
         * graph or silently omit the first maintenance transaction.
         */
        const bool prepare_first_transaction_boundary = !compiling_sparse_query_program_ ||
            sparseQueryExecutionProgramIndex(*compiling_sparse_query_program_) == 0;
        const bool owns_dynamic_device_maintenance =
            usesParticipantLocalDeviceMoERebalanceController();
        /*
         * This is the first committed transaction's own boundary. The
         * completion_event_in_flight flag tracks diagnostic export of the
         * newest maintenance event; it can still describe prefill (or an
         * earlier request) and cannot prove this transaction was serviced.
         * Submit exactly once here. The device-owned cadence predicate makes
         * a non-due boundary cheap, while a due edge must reset its countdown
         * before the parent's next verifier admission.
         */
        if (prepare_first_transaction_boundary && owns_dynamic_device_maintenance &&
            !maybeRunDeviceMoERebalanceMaintenanceGraph())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation parent preparation could not launch the first committed MoE maintenance boundary"
                      << " device=" << state_.device_id.toString());
            return false;
        }

        /*
         * Boundary children must already own strict captured executables before
         * they can be cloned into a native/hosted parent. The externally
         * scheduled first transaction captured acquire on entry. Join its full
         * committed state here, consume any maintenance publication, and close
         * that reader on the future parent stream; this both captures release on
         * first use and makes the first internal acquire legal. Static overlay
         * reaches the same path without manufacturing a maintenance graph.
         */
        if (prepare_first_transaction_boundary && moe_overlay_epoch_execution_binding_)
        {
            auto &loop = deviceGenerationLoopGraph();
            void *const parent_stream = loop.stream.get();
            const char *const parent_name =
                configured_depth_policy.isOrdinary()
                    ? "ordinary_device_generation_parent_materialization"
                    : "device_generation_parent_materialization";
            if (!parent_stream ||
                !waitForLiveInferenceStateReadyForObservation(
                    parent_stream,
                    parent_name,
                    DeviceTimelineRole::DeviceGenerationController) ||
                (owns_dynamic_device_maintenance &&
                 !waitForPendingDeviceMoERebalanceMaintenance(
                     parent_stream,
                     DeviceTimelineRole::MTPSidecarGraph,
                     parent_name)) ||
                !prepareMoEOverlayEpochForInternalParent(
                    parent_stream,
                    parent_name))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-generation parent preparation could not close and capture the externally scheduled ExpertOverlay boundary"
                          << " device=" << state_.device_id.toString());
                return false;
            }
        }

        const auto compile_program = [&]() -> bool
        {
            std::string error;
            if (configured_depth_policy.isOrdinary())
            {
                // Pipeline placement and sparse request ownership are independent
                // axes. Their combined terminal retains both captured PP edges and
                // the ExpertOverlay semantic replay used to bind sparse identity.
                const bool pipeline_domain = pipeline_forward_edges_ &&
                    pipeline_forward_edges_->hasDomainBoundary();
                const auto composition = moe_overlay_epoch_execution_binding_
                    ? (pipeline_domain ? OrdinaryGenerationComposition::PipelineDomainExpertOverlayTail
                                       : OrdinaryGenerationComposition::ExpertOverlay)
                    : (pipeline_domain ? OrdinaryGenerationComposition::PipelineDomainTail
                                       : OrdinaryGenerationComposition::CompleteLocal);
                if (draft_depth != 0 ||
                    !materializeOrdinaryDeviceGenerationLoopGraph(
                        sampling_mode,
                        &error,
                        composition))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Ordinary generation preparation failed: " << error);
                    return false;
                }
                return true;
            }
            if (!materializeMTPDeviceGenerationLoopGraph(
                    request_count,
                    draft_depth,
                    sampling_mode,
                    &error))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-generation parent preparation failed: "
                          << error);
                return false;
            }
            return true;
        };
        // A pipeline coordinator already scopes the same program on every
        // participant; it owns the outer two-family composition order.
        if (compiling_sparse_query_program_) return compile_program();
        for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
        {
            SparseQueryProgramCompilationScope program_scope(*this,
                program == 0 ? SparseQueryExecutionProgram::RankingPossible : SparseQueryExecutionProgram::ProvenNoRanking);
            if (!compile_program()) return false;
        }
        return true;
    }

    bool DeviceGraphOrchestrator::prepareDeviceResidentGenerationContinuation(
        DeviceGenerationSamplingMode sampling_mode)
    {
        using namespace sampling_math;
        const auto &admission = active_device_generation_admission_;
        const auto reject = [&](const std::string &reason) {
            LOG_ERROR("[DeviceGraphOrchestrator] ContinueResponse retained entry rejected: " << reason);
            return false;
        };
        if (!admission || admission->kind != DeviceGenerationAdmissionKind::ContinueResponse ||
            admission->request_count != 1 || admission->depth_policy.mode != DeviceGenerationPolicyMode::Dynamic ||
            deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::DynamicDepth) !=
                DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions ||
            mtpVerifierGraphFamily() != MTPVerifierGraphFamily::ExactRows ||
            !isValidDeviceGenerationSamplingMode(sampling_mode) ||
            device_generation_storage_.active_request_count != 1 ||
            !device_generation_state_ready_.handoff.published() ||
            !hosted_device_generation_cursor_.inactive())
            return reject("admission, exact hosted policy, or lifecycle is incomplete");

        // Rebuild only the host candidate identity list. Missing/changed children
        // fail before topology replacement; no cold materializer or old outcome runs.
        const auto authenticate = [&]() {
            auto &loop = deviceGenerationLoopGraph();
            if (!loop.valid || loop.launched || !loop.hosted_advance.idle())
                return reject("retained query-program parent is invalid, launched, or still owns an unfinished branch: program=" +
                    std::to_string(sparseQueryExecutionProgramIndex(sparseQueryGraphProgram())));
            std::string error;
            if (!materializeMTPDeviceGenerationLoopGraph(1, admission->depth_policy.maximum_depth,
                    sampling_mode, &error, MTPGenerationFamilyPreparation::RequireRetained))
                return reject(error);
            return true;
        };
        if (compiling_sparse_query_program_)
        {
            if (!authenticate()) return false;
        }
        else
        {
            for (std::size_t program = 0; program < sparseQueryProgramCount(); ++program)
            {
                SparseQueryProgramCompilationScope scope(*this, program == 0 ?
                    SparseQueryExecutionProgram::RankingPossible : SparseQueryExecutionProgram::ProvenNoRanking);
                if (!authenticate()) return false;
            }
        }

        // A direct-emission M1 forward invalidates the previously accepted
        // prefix row. Its one-row selector was admitted with the model family;
        // continuation may replay it but may never build a missing selector.
        if (!state_.mtp_terminal_hidden_publication.current())
        {
            if (state_.last_forward_seq_len != 1 || state_.last_forward_batch_size != 1 ||
                mtp_terminal_hidden_contiguous_rows_select_caches_.empty() ||
                !mtp_terminal_hidden_contiguous_rows_select_caches_.front())
                return reject("Full terminal-hidden refresh has no retained single-row source");
            const auto &selector = *mtp_terminal_hidden_contiguous_rows_select_caches_.front();
            if (!selector.valid || !selector.graph || !selector.segment_cache.initialized ||
                selector.segment_cache.needs_capture || selector.workspace_generation != workspaceGeneration(state_.device_id))
                return reject("Full terminal-hidden refresh requires its unchanged retained one-row selector");
        }
        TensorBase *terminal_hidden = nullptr;
        BufferId terminal_hidden_id = BufferId::HIDDEN_STATE;
        if (!resolveMTPTerminalHiddenInput(&terminal_hidden, &terminal_hidden_id,
                "hosted_continuation_full_input") ||
            terminal_hidden != mtp_sidecar_depth0_device_token_cache_.terminal_hidden ||
            terminal_hidden_id != BufferId::PREFIX_TERMINAL_HIDDEN)
            return reject("Full sidecar lost its canonical current terminal-hidden owner");
        const auto *live_position = state_.kv_cache
            ? state_.kv_cache->deviceSequenceCachedTokenCountPtr(0) : nullptr;
        auto &loop = deviceGenerationLoopGraph();
        auto &full = mtp_sidecar_depth0_device_token_cache_;
        auto *backend = getBackendFor(state_.device_id);
        void *const stream = loop.stream.get();
        if (!backend || !stream || !live_position || !arena_ ||
            !device_generation_state_ready_.event ||
            !DeviceEventEdge::at(DeviceTimelinePoint::DeviceGenerationStateReady)
                .from(DeviceTimelineRole::DeviceGenerationController)
                .to(DeviceTimelineRole::DeviceGenerationController)
                .enqueueWait(*backend, state_.device_id, device_generation_state_ready_.event.get(),
                    device_generation_state_ready_.handoff.frontierStream(), stream) ||
            !stochastic_target_sample_tokens_dev_ ||
            !publishDeviceResidentLogicalSequenceStateFromTargetSample(0, live_position) ||
            !waitForRequiredStochasticTargetSampleReady(0, stream, "hosted_continuation_full_input") ||
            !arena_->prepareForRead(terminal_hidden_id, state_.device_id, stream) ||
            !backend->enqueuePrepareMTPBatchedSidecarInputs(stochastic_target_sample_tokens_dev_, 1,
                live_position, 0, 1, state_.device_id.gpu_ordinal(), stream,
                full.condition_token_device, const_cast<void *>(full.position_ids_device)))
            return reject("prepared target/position/terminal-hidden inputs lack an ordered device source");
        arena_->markWritten(BufferId::MTP_CONDITION_TOKEN, state_.device_id, stream);
        arena_->markWritten(BufferId::MTP_POSITION_IDS, state_.device_id, stream);
        if (!hosted_device_generation_cursor_.armContinuationAdmission(admission->kind))
            return reject("ticket-zero authority was already consumed");
        return true;
    }

    bool DeviceGraphOrchestrator::
        enqueueDeviceGenerationDispatchTicketObservation()
    {
        auto &loop = deviceGenerationLoopGraph();
        const int request_count =
            device_generation_storage_.active_request_count;
        IBackend *const backend = getBackendFor(state_.device_id);
        if (loop.execution_kind !=
                MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                    HostedDispatchTicketPublisher ||
            !loop.valid || !loop.capture || !loop.capture->hasExecutable() ||
            !loop.stream || request_count != 1 ||
            !device_generation_storage_.validFor(request_count) ||
            !device_generation_dispatch_ticket_host_scratch_ ||
            !device_generation_dispatch_ticket_host_scratch_->canServe(
                request_count) ||
            !device_generation_dispatch_ticket_ready_event_ ||
            !hosted_device_generation_cursor_
                 .maySubmitTicketObservation() ||
            !backend)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket observation has incomplete or overlapping ownership"
                      << " device=" << state_.device_id.toString()
                      << " requests=" << request_count
                      << " loop_valid=" << loop.valid
                      << " lifecycle_state="
                      << static_cast<int>(
                             hosted_device_generation_cursor_.state()));
            return false;
        }

        void *const scheduler_stream = loop.stream.get();
        const size_t ticket_bytes =
            sizeof(sampling_math::DeviceGenerationDispatchTicket) *
            static_cast<size_t>(request_count);
        // Admission already published the genuine native ticket with maintenance=0.
        // Replaying the completed-transaction publisher here would read a prior
        // maintenance predicate and manufacture a predecessor boundary.
        if ((!hosted_device_generation_cursor_.admissionTicketPending() &&
             !loop.capture->launchOnStream(scheduler_stream)) ||
            !backend->deviceToHostOnStream(
                device_generation_dispatch_ticket_host_scratch_->tickets,
                device_generation_storage_.dispatch_tickets_device,
                ticket_bytes,
                state_.device_id.gpu_ordinal(),
                scheduler_stream) ||
            !backend->recordEvent(
                device_generation_dispatch_ticket_ready_event_.get(),
                state_.device_id.gpu_ordinal(),
                scheduler_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket observation submission failed on "
                      << state_.device_id.toString());
            return false;
        }
        if (!hosted_device_generation_cursor_
                 .markTicketObservationSubmitted())
        {
            /*
             * The immutable ticket copy is already in flight. Re-entering the
             * scheduler here could duplicate a retained transaction, so a
             * failed post-submit transition is a fatal internal invariant.
             */
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted ticket submission could not acquire its typed lifecycle state");
            std::terminate();
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_dispatch_ticket_d2h_submissions",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"bytes", std::to_string(ticket_bytes)},
             {"abi_version",
              std::to_string(
                  sampling_math::DeviceGenerationDispatchTicket::kABIVersion)},
             {"word_count",
              std::to_string(
                  sampling_math::DeviceGenerationDispatchTicket::kWordCount)},
             {"authority", "immutable_scheduler_snapshot"},
             {"state_payload", "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::observeDeviceGenerationDispatchTicket(
        sampling_math::DeviceGenerationDispatchTicket *out_ticket)
    {
        using namespace sampling_math;
        if (out_ticket)
            *out_ticket = DeviceGenerationDispatchTicket{};

        auto &loop = deviceGenerationLoopGraph();
        const int request_count =
            device_generation_storage_.active_request_count;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        IBackend *const backend = getBackendFor(state_.device_id);
        if (!out_ticket || !backend || request_count != 1 ||
            loop.execution_kind !=
                MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                    HostedDispatchTicketPublisher ||
            !loop.valid || !loop.stream || loop.workspace_generation == 0 ||
            loop.workspace_generation != generation ||
            !hosted_device_generation_cursor_.mayObserveTicket())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket observation has stale lifecycle identity");
            return false;
        }

        if (!hosted_device_generation_cursor_.schedulerStarted())
        {
            if (loop.launched ||
                !consumeDeviceGenerationStateReady(
                    loop.stream.get(),
                    DeviceTimelineRole::DeviceGenerationController,
                    request_count,
                    "hosted_device_generation_ticket_scheduler") ||
                (usesParticipantLocalDeviceMoERebalanceController() &&
                 !waitForPendingDeviceMoERebalanceMaintenance(
                     loop.stream.get(),
                     DeviceTimelineRole::MTPSidecarGraph,
                     "hosted_device_generation_ticket_scheduler")))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation scheduler could not consume the first committed transaction");
                return false;
            }
            if (!hosted_device_generation_cursor_.continuationAdmissionReady() &&
                !prepareMoEOverlayEpochForInternalParent(
                    loop.stream.get(),
                    "hosted_device_generation_ticket_scheduler"))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation scheduler could not close the externally scheduled ExpertOverlay transaction");
                return false;
            }
            if (!hosted_device_generation_cursor_.startScheduler())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation scheduler rejected its initial typed transition");
                return false;
            }
            if (loop.read_ahead_resources)
            {
                // The original family spans the authenticated request cursor,
                // including every heterogeneous forward/maintenance boundary.
                // Tickets select retained work; none retires this commitment.
                loop.read_ahead_resources->beginSubmission();
            }
            // The ticket protocol observes completed transactions. Ordinary
            // requests have not executed their initial sample externally, so
            // submit that complete retained boundary on the controller stream
            // before asking for ticket one. No token or KV data is observed.
            if (loop.ordinary_sampling_identity)
            {
                if (!waitForDeviceResidentLogicalSequenceStateRowReuse(loop.stream.get(),
                        "ordinary_hosted_generation_writer"))
                    return false;
                const bool pending_response = loop.ordinary_leading ==
                    sampling_math::DeviceGenerationLeadingRowDisposition::PendingResponse;
                if (pending_response)
                {
                    /*
                     * Prefill already held and released the overlay epoch. Its
                     * scalar sample reads the retained logits only, so it must
                     * not manufacture a second lease boundary.
                     */
                    if (!loop.ordinary_children[
                            OrdinaryGenerationGraphPlan::prefill_sampler] ||
                        !loop.ordinary_children[
                            OrdinaryGenerationGraphPlan::prefill_sampler]
                             ->launchOnStream(loop.stream.get()))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Ordinary initial prefill sample failed");
                        std::terminate();
                    }
                }
                else if (loop.ordinary_hosted_forward)
                {
                    /*
                     * ExpertOverlay deliberately has no monolithic local
                     * hosted_transaction child: its retained forward enters a
                     * rank-wide sparse graph group and must be replayed by all
                     * participants together. RankOrchestrator has already
                     * admitted that shared sequence and fans these calls out
                     * on the persistent TP workers.
                     */
                    std::string replay_error;
                    void *producer_stream = nullptr;
                    const auto &sampler = loop.ordinary_children[
                        OrdinaryGenerationGraphPlan::decode_sampler];
                    if (!loop.ordinary_forward_identity || !sampler ||
                        !replayHostedOrdinaryExpertOverlayTransaction(
                            *loop.ordinary_forward_identity,
                            sampler.get(),
                            &producer_stream,
                            &replay_error))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Ordinary initial ExpertOverlay transaction failed: "
                                  << replay_error);
                        std::terminate();
                    }
                }
                else if (!loop.ordinary_children[
                                 OrdinaryGenerationGraphPlan::hosted_transaction] ||
                         !loop.ordinary_children[
                              OrdinaryGenerationGraphPlan::hosted_transaction]
                              ->launchOnStream(loop.stream.get()))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Ordinary initial captured transaction failed");
                    std::terminate();
                }
            }
            loop.launched = true;
            if (loop.ordinary_sampling_identity &&
                !recordDeviceResidentLogicalSequenceStateMailbox(
                    request_count, loop.stream.get(),
                    DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTransaction))
            {
                // The initial captured sample/forward has been submitted.
                // Publish its exact stream before maintenance can observe this
                // transaction; a request-long writer would starve that reader.
                LOG_ERROR("[DeviceGraphOrchestrator] Initial ordinary hosted transaction could not publish logical state");
                std::terminate();
            }
            if (!enqueueDeviceGenerationDispatchTicketObservation())
                return false;
            if (loop.read_ahead_resources)
                PerfStatsCollector::addCounter("ngram", "read_ahead_complete_request_submissions", 1.0,
                    "decode", state_.device_id.toString(),
                    {{"execution", "hosted_ticket_selected_captured_transactions"}});
        }

        if (!hosted_device_generation_cursor_.ticketObservationPending() ||
            !backend->waitForEvent(
                device_generation_dispatch_ticket_ready_event_.get(),
                state_.device_id.gpu_ordinal()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket event did not complete on "
                      << state_.device_id.toString());
            return false;
        }

        const DeviceGenerationDispatchTicket ticket =
            device_generation_dispatch_ticket_host_scratch_->tickets[0];
        const std::int32_t last_transaction_count =
            hosted_device_generation_cursor_.lastTransactionCount();
        const bool transaction_ordinal_available =
            last_transaction_count <
            std::numeric_limits<std::int32_t>::max();
        const bool admission_ticket = hosted_device_generation_cursor_.admissionTicketPending();
        const std::int32_t expected_transaction_count = admission_ticket ? 0 :
            transaction_ordinal_available
                ? last_transaction_count + 1
                : std::numeric_limits<std::int32_t>::min();
        const bool valid_ticket = hosted_device_generation_cursor_.acceptsTicket(ticket,
            session_epoch_, generation, loop.minimum_draft_depth, loop.maximum_draft_depth);
        if (!valid_ticket)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation dispatch ticket failed authentication"
                      << " device=" << state_.device_id.toString()
                      << " abi=" << ticket.abi_version
                      << " session=" << ticket.sessionEpoch()
                      << " expected_session=" << session_epoch_
                      << " workspace=" << ticket.workspaceGeneration()
                      << " expected_workspace=" << generation
                      << " healthy=" << ticket.healthy
                      << " complete=" << ticket.complete
                      << " transaction=" << ticket.transaction_count
                      << " expected_transaction="
                      << expected_transaction_count
                      << " next_depth=" << ticket.next_draft_depth
                      << " depth_range=[" << loop.minimum_draft_depth << ','
                      << loop.maximum_draft_depth << ']'
                      << " error=" << ticket.error_code
                      << " maintenance_due=" << ticket.maintenance_due);
            return false;
        }

        if (!hosted_device_generation_cursor_.acceptTicket(
                ticket.transaction_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Authenticated hosted ticket could not enter its typed observed state");
            return false;
        }
        last_device_generation_dispatch_ticket_ = ticket;
        *out_ticket = ticket;
        PerfStatsCollector::Tags observation_tags{
            {"next_depth", std::to_string(ticket.next_draft_depth)},
            {"complete", ticket.complete ? "true" : "false"},
            {"maintenance_due", ticket.maintenance_due ? "true" : "false"},
            {"entry_source", admission_ticket ? "native_continue_response_admission" : "committed_transaction_ticket"}};
        if (admission_ticket)
        {
            // Lifecycle IDs are fixed-width witness words, never cumulative map
            // keys. All three admission producers append the same ordered words
            // so saved consumers can join exact counts/digests/ranges per owner.
            PerfStatsCollector::addCounterWithSequence("mtp", "device_generation_continuation_admission_tickets", 1.0,
                {0u, ticket.sessionEpoch(), ticket.workspaceGeneration()}, "decode", state_.device_id.toString(),
                {{"depth", std::to_string(ticket.next_draft_depth)},
                 {"physical_rows", std::to_string(ticket.next_draft_depth + 1)},
                 {"family", "exact_rows"}, {"entry_source", "native_continue_response_admission"}});
            observation_tags.emplace("physical_rows", std::to_string(ticket.next_draft_depth + 1));
            observation_tags.emplace("family", "exact_rows");
            PerfStatsCollector::addCounterWithSequence("mtp", "device_generation_dispatch_tickets_observed", 1.0,
                {0u, ticket.sessionEpoch(), ticket.workspaceGeneration()}, "decode", state_.device_id.toString(),
                std::move(observation_tags));
        }
        else
            PerfStatsCollector::addCounterWithSequence("mtp", "device_generation_dispatch_tickets_observed", 1.0,
                {static_cast<uint64_t>(ticket.transaction_count)}, "decode", state_.device_id.toString(),
                std::move(observation_tags));
        return true;
    }

    bool DeviceGraphOrchestrator::beginHostScheduledDeviceGenerationAdvance(
        const sampling_math::DeviceGenerationDispatchTicket &ticket,
        size_t *out_fragment_count)
    {
        if (out_fragment_count)
            *out_fragment_count = 0;
        auto &loop = deviceGenerationLoopGraph();
        const int request_count =
            device_generation_storage_.active_request_count;
        if (!out_fragment_count || !last_device_generation_dispatch_ticket_ ||
            !ticket.matchesLifecycle(
                last_device_generation_dispatch_ticket_->sessionEpoch(),
                last_device_generation_dispatch_ticket_
                    ->workspaceGeneration()) ||
            !ticket.hasSameDispatchDecision(
                *last_device_generation_dispatch_ticket_) ||
            request_count != 1 || !loop.valid || !loop.launched ||
            !loop.stream || !loop.hosted_advance.idle() ||
            loop.execution_kind !=
                MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                    HostedDispatchTicketPublisher ||
            !hosted_device_generation_cursor_.mayBeginAdvance(
                ticket.transaction_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation advance rejected a stale, altered, or overlapping ticket");
            return false;
        }

        const size_t branch_index =
            static_cast<size_t>(ticket.next_draft_depth);
        const size_t branch_offset = loop.branch_offsets[branch_index];
        const size_t branch_count =
            loop.branch_fragment_counts[branch_index];
        if (branch_count == 0 ||
            branch_offset > loop.hosted_fragments.size() ||
            branch_count > loop.hosted_fragments.size() - branch_offset)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket selected an unmaterialized semantic branch depth="
                      << ticket.next_draft_depth);
            return false;
        }

        size_t unconditional_count = 0;
        size_t conditional_count = 0;
        for (size_t index = 0; index < branch_count; ++index)
        {
            const HostedDeviceGenerationFragment &fragment =
                loop.hosted_fragments[branch_offset + index];
            if (fragment.execution ==
                DeviceControlledLoopFragmentExecution::Always)
            {
                ++unconditional_count;
                continue;
            }
            ++conditional_count;
            if (index + 1 != branch_count || !fragment.capture ||
                !fragment.condition_word_device)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation branch has an ambiguous conditional tail");
                return false;
            }
        }
        if (conditional_count > 1 ||
            (ticket.maintenance_due != 0 && conditional_count != 1) ||
            (ticket.complete == 0 && unconditional_count == 0))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation ticket disagrees with branch fragment topology");
            return false;
        }

        /*
         * The authenticated device ticket is the sole sequence-transition
         * authority. Every symmetric child presents it to the shared
         * coordinator; the first caller applies the transition and siblings
         * consume that same idempotent decision. No participant index or host
         * worker scheduling order carries lifecycle meaning.
         */
        if (moe_overlay_inference_transaction_coordinator_)
        {
            using HostedTransition =
                MoEOverlayInferenceTransactionCoordinator::
                    HostedGraphSequenceTransition;
            std::string sequence_error;
            const HostedTransition transition =
                ticket.complete != 0
                    ? HostedTransition::terminal()
                    : loop.ordinary_sampling_identity
                          ? HostedTransition::ordinaryDecode()
                          : HostedTransition::speculativeMTP(
                                ticket.next_draft_depth);
            const bool sequence_ready = ticket.transaction_count == 0
                ? moe_overlay_inference_transaction_coordinator_->beginHostedContinuationGraphSequence(
                      transition, &sequence_error)
                : moe_overlay_inference_transaction_coordinator_->advanceHostedGraphSequence(
                      static_cast<std::uint64_t>(ticket.transaction_count), transition,
                      static_cast<std::uint64_t>(ticket.committed_output_tokens), &sequence_error);
            if (!sequence_ready)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted ExpertOverlay graph-sequence transition failed: "
                          << sequence_error);
                return false;
            }
        }

        const DeviceControlledLoopTicketSelection selection{
            .next_iteration_admitted = ticket.complete == 0,
            .completed_iteration_word_nonzero = ticket.maintenance_due != 0,
            .selector = ticket.next_draft_depth,
        };
        const std::span<const HostedDeviceGenerationFragment>
            ordered_branch(
                loop.hosted_fragments.data() + branch_offset,
                branch_count);
        const size_t selected_fragment_count =
            selection.countSelected(ordered_branch);
        if (loop.ordinary_sampling_identity &&
            !waitForDeviceResidentLogicalSequenceStateRowReuse(
                loop.stream.get(), "ordinary_hosted_transaction_writer"))
        {
            // The topology coordinator has accepted the completed ticket.
            // Acquire the next writer only after that boundary, joining the
            // previous publication and its actual readers through existing
            // event edges. The terminal branch also owns its final publication.
            LOG_ERROR("[DeviceGraphOrchestrator] Ordinary hosted transaction could not acquire its logical-state writer");
            return false;
        }
        if (!hosted_device_generation_cursor_.beginAdvance(
                ticket.transaction_count) ||
            !loop.hosted_advance.begin(selected_fragment_count))
        {
            /*
             * The cross-participant graph-sequence coordinator has already
             * accepted this ticket. A partial local transition cannot be
             * retried safely because it could submit the branch twice.
             */
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted branch could not atomically acquire its typed submission cursors");
            std::terminate();
        }
        *out_fragment_count = loop.hosted_advance.fragmentCount();
        return true;
    }

    bool DeviceGraphOrchestrator::
        submitHostScheduledDeviceGenerationFragment(
            const sampling_math::DeviceGenerationDispatchTicket &ticket,
            size_t fragment_index)
    {
        auto &loop = deviceGenerationLoopGraph();
        if (!hosted_device_generation_cursor_.submitting() ||
            !last_device_generation_dispatch_ticket_ ||
            !ticket.hasSameDispatchDecision(
                *last_device_generation_dispatch_ticket_) ||
            !loop.hosted_advance.maySubmit(fragment_index))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted fragment submission is stale, duplicated, or out of order");
            return false;
        }

        const size_t branch_index =
            static_cast<size_t>(ticket.next_draft_depth);
        const size_t branch_offset = loop.branch_offsets[branch_index];
        const size_t branch_count =
            loop.branch_fragment_counts[branch_index];
        const DeviceControlledLoopTicketSelection selection{
            .next_iteration_admitted = ticket.complete == 0,
            .completed_iteration_word_nonzero = ticket.maintenance_due != 0,
            .selector = ticket.next_draft_depth,
        };
        const std::span<const HostedDeviceGenerationFragment>
            ordered_branch(
                loop.hosted_fragments.data() + branch_offset,
                branch_count);
        const HostedDeviceGenerationFragment *const selected =
            selection.selectOrdinal(ordered_branch, fragment_index);
        if (!selected)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted fragment ordinal has no semantic branch member index="
                      << fragment_index);
            return false;
        }

        bool submitted = false;
        std::string replay_error;
        void *producer_stream = nullptr;
        switch (selected->kind)
        {
        case HostedDeviceGenerationFragment::Kind::CapturedLocal:
            submitted = selected->capture &&
                selected->capture->launchOnStream(loop.stream.get());
            break;
        case HostedDeviceGenerationFragment::Kind::OrdinaryExpertOverlayReplay:
            submitted = replayHostedOrdinaryExpertOverlayTransaction(
                selected->forward_signature,
                selected->capture,
                &producer_stream,
                &replay_error);
            break;
        case HostedDeviceGenerationFragment::Kind::MTPFullSidecarReplay:
            submitted = replayHostedMTPSidecar(
                MTPSidecarCaptureRole::Full,
                ticket.next_draft_depth,
                &producer_stream,
                &replay_error);
            break;
        case HostedDeviceGenerationFragment::Kind::MTPChainedSidecarReplay:
            submitted = replayHostedMTPSidecar(
                MTPSidecarCaptureRole::Chained,
                ticket.next_draft_depth,
                &producer_stream,
                &replay_error);
            break;
        case HostedDeviceGenerationFragment::Kind::MTPGroupedVerifierReplay:
            submitted = replayHostedMTPGroupedVerifier(
                selected->forward_signature,
                ticket.next_draft_depth,
                &producer_stream,
                &replay_error);
            break;
        case HostedDeviceGenerationFragment::Kind::MoEOverlayEpochAcquire:
            submitted = acquireMoEOverlayEpochForHostedParent(
                loop.stream.get(),
                selected->capture,
                "hosted_device_generation_branch_entry");
            break;
        case HostedDeviceGenerationFragment::Kind::MoEOverlayEpochRelease:
            submitted = releaseMoEOverlayEpochForHostedParent(
                loop.stream.get(),
                selected->capture,
                "hosted_device_generation_branch_terminal");
            break;
        }
        if (!submitted)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation fragment submission failed"
                      << " device=" << state_.device_id.toString()
                      << " fragment="
                      << (selected->name ? selected->name : "unnamed")
                      << " depth=" << ticket.next_draft_depth
                      << (replay_error.empty() ? "" : " error=")
                      << replay_error);
            return false;
        }

        if (!loop.hosted_advance.recordSubmission(fragment_index))
        {
            /* The fragment ran; permitting a retry would duplicate it. */
            LOG_ERROR("[DeviceGraphOrchestrator] Submitted hosted fragment could not advance its exact ordinal cursor");
            std::terminate();
        }
        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "hosted_device_generation_fragment_submissions",
            1.0,
            {static_cast<uint64_t>(ticket.transaction_count)},
            "decode",
            state_.device_id.toString(),
            {{"depth", std::to_string(ticket.next_draft_depth)},
             {"fragment_index", std::to_string(fragment_index)},
             {"fragment", selected->name ? selected->name : "unnamed"}});
        return true;
    }

    bool DeviceGraphOrchestrator::finishHostScheduledDeviceGenerationAdvance(
        const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        auto &loop = deviceGenerationLoopGraph();
        const int request_count =
            device_generation_storage_.active_request_count;
        if (!hosted_device_generation_cursor_.submitting() ||
            !loop.hosted_advance.mayFinish() ||
            !last_device_generation_dispatch_ticket_ ||
            !ticket.hasSameDispatchDecision(
                *last_device_generation_dispatch_ticket_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted advance cannot finish before every selected fragment is submitted");
            return false;
        }

        if (ticket.complete != 0)
        {
            if (!publishDeviceGenerationStateReady(
                    loop.stream.get(),
                    request_count,
                    DeviceGenerationStatePublicationKind::Terminal))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation terminal event publication failed");
                std::terminate();
            }
            if (!loop.hosted_advance.finish() ||
                !hosted_device_generation_cursor_.finishAdvance(
                    ticket.transaction_count,
                    /*terminal=*/true))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Hosted terminal branch could not retire its typed submission cursors");
                std::terminate();
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "hosted_device_generation_terminal_submissions",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"transactions",
                  std::to_string(ticket.transaction_count)}});
            // This participant owns the authenticated terminal frontier in
            // both direct and rank-coordinated hosted execution. Recording at
            // the direct launch caller would omit every TP participant.
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_loop_graph_launches",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"sparse_query_program", sparseQueryGraphProgram() == SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
                 {"backend", loop.capture->backendName()},
                 {"requests", std::to_string(request_count)},
                 {"draft_depth", std::to_string(loop.draft_depth)},
                 {"minimum_draft_depth", std::to_string(loop.minimum_draft_depth)},
                 {"maximum_draft_depth", std::to_string(loop.maximum_draft_depth)},
                 {"fragments", std::to_string(loop.fragment_count)},
                 {"conditional_fragments", std::to_string(loop.native_conditional_fragment_count)},
                 {"selector_gated_fragments", std::to_string(loop.native_selector_gated_fragment_count)},
                 {"ticket_conditioned_fragments", std::to_string(loop.ticket_conditioned_fragment_count)},
                 {"execution", "hosted_ticket_selected_captured_transactions"}});
            return true;
        }

        if (loop.ordinary_sampling_identity &&
            !recordDeviceResidentLogicalSequenceStateMailbox(
                request_count, loop.stream.get(),
                DeviceResidentLogicalStatePublicationKind::OrdinaryGenerationTransaction))
        {
            // All selected captured fragments precede this event on the exact
            // controller stream. Closing each writer lets background snapshots
            // consume a completed transaction while the request remains live.
            LOG_ERROR("[DeviceGraphOrchestrator] Ordinary hosted transaction could not publish logical state");
            std::terminate();
        }
        if (!loop.hosted_advance.finish() ||
            !hosted_device_generation_cursor_.finishAdvance(
                ticket.transaction_count,
                /*terminal=*/false))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted transaction branch could not retire its typed submission cursors");
            std::terminate();
        }
        last_device_generation_dispatch_ticket_.reset();
        if (!enqueueDeviceGenerationDispatchTicketObservation())
            return false;
        PerfStatsCollector::Tags submission_tags{
            {"depth", std::to_string(ticket.next_draft_depth)},
            {"entry_source", ticket.transaction_count == 0 ? "native_continue_response_admission" : "committed_transaction_ticket"},
            {"fragments", std::to_string(loop.branch_fragment_counts[static_cast<size_t>(ticket.next_draft_depth)])},
            {"dynamic_depth_source", "device_controller_ticket"}};
        if (ticket.transaction_count == 0)
        {
            submission_tags.emplace("physical_rows", std::to_string(ticket.next_draft_depth + 1));
            submission_tags.emplace("family", "exact_rows");
            PerfStatsCollector::addCounterWithSequence("mtp", "hosted_device_generation_transaction_submissions", 1.0,
                {0u, ticket.sessionEpoch(), ticket.workspaceGeneration()}, "decode", state_.device_id.toString(),
                std::move(submission_tags));
        }
        else
            PerfStatsCollector::addCounter("mtp", "hosted_device_generation_transaction_submissions", 1.0,
                "decode", state_.device_id.toString(), std::move(submission_tags));
        return true;
    }

    bool DeviceGraphOrchestrator::
        submitHostScheduledDeviceGenerationAdvance(
            const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        size_t fragment_count = 0;
        if (!beginHostScheduledDeviceGenerationAdvance(
                ticket, &fragment_count))
        {
            return false;
        }
        for (size_t fragment = 0; fragment < fragment_count; ++fragment)
        {
            if (!submitHostScheduledDeviceGenerationFragment(
                    ticket, fragment))
            {
                return false;
            }
        }
        return finishHostScheduledDeviceGenerationAdvance(ticket);
    }

    bool DeviceGraphOrchestrator::launchDeviceResidentGeneration()
    {
        auto &loop = deviceGenerationLoopGraph();
        const int request_count =
            device_generation_storage_.active_request_count;
        const uint64_t generation = workspaceGeneration(state_.device_id);
        if (!state_.device_id.is_gpu() || request_count <= 0 ||
            !device_generation_storage_.validFor(request_count) ||
            !loop.valid || loop.launched || !loop.stream || !loop.capture ||
            !loop.capture->hasExecutable() ||
            loop.capture->executionStream() != loop.stream.get() ||
            loop.request_count != request_count ||
            loop.workspace_generation == 0 ||
            loop.workspace_generation != generation)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation loop launch has incomplete or stale ownership"
                      << " device=" << state_.device_id.toString()
                      << " requests=" << request_count
                      << " loop_valid=" << loop.valid
                      << " launched=" << loop.launched
                      << " graph_executable="
                      << (loop.capture && loop.capture->hasExecutable())
                      << " loop_generation=" << loop.workspace_generation
                      << " workspace_generation=" << generation);
            return false;
        }

        if (loop.execution_kind ==
            MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                HostedDispatchTicketPublisher)
        {
            const int maximum_transaction_count =
                device_generation_storage_.response_token_stride;
            const int admission_observation = hosted_device_generation_cursor_.continuationAdmissionReady() ? 1 : 0;
            for (std::int64_t observation = 0;
                 observation - admission_observation < maximum_transaction_count;
                 ++observation)
            {
                sampling_math::DeviceGenerationDispatchTicket ticket;
                if (!observeDeviceGenerationDispatchTicket(&ticket) ||
                    !submitHostScheduledDeviceGenerationAdvance(ticket))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation loop failed at observation "
                              << observation << " on "
                              << state_.device_id.toString());
                    return false;
                }
                if (ticket.complete != 0)
                {
                    return true;
                }
            }
            LOG_ERROR("[DeviceGraphOrchestrator] Hosted device-generation controller exceeded the response-capacity transaction bound without completing");
            return false;
        }
        if (loop.execution_kind !=
            MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                NativeConditionalParent)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation launch has no materialized execution kind");
            return false;
        }

        void *const loop_stream = loop.stream.get();
        if (!consumeDeviceGenerationStateReady(
                loop_stream,
                DeviceTimelineRole::DeviceGenerationController,
                request_count,
                "device_generation_parent_loop"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation loop could not consume the first committed transaction");
            return false;
        }

        if (loop.ordinary_sampling_identity &&
            !waitForDeviceResidentLogicalSequenceStateRowReuse(loop_stream,
                "ordinary_native_generation_writer"))
            return false;

        if (usesParticipantLocalDeviceMoERebalanceController() &&
            !waitForPendingDeviceMoERebalanceMaintenance(
                loop_stream,
                DeviceTimelineRole::MTPSidecarGraph,
                "device_generation_parent_loop"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation loop could not join prior MoE maintenance");
            return false;
        }
        if (!prepareMoEOverlayEpochForInternalParent(
                loop_stream,
                "device_generation_parent_loop"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device-generation loop could not close the externally scheduled ExpertOverlay transaction");
            return false;
        }

        /*
         * Once cudaGraphLaunch accepts the parent, every remaining transaction
         * and its terminal controller publication are in flight on this exact
         * stream. Failure to publish the post-launch event would leave no legal
         * consumer edge for terminal materialization, so either backend failure
         * is process-fatal rather than a recoverable boolean result.
         */
        if (loop.read_ahead_resources)
            loop.read_ahead_resources->beginSubmission();
        if (!loop.capture->launch())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Native device-generation parent launch failed after lifecycle consumption on "
                      << state_.device_id.toString());
            std::terminate();
        }
        if (!publishDeviceGenerationStateReady(
                loop_stream,
                request_count,
                DeviceGenerationStatePublicationKind::Terminal))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Native device-generation parent could not publish its terminal event on "
                      << state_.device_id.toString());
            std::terminate();
        }
        loop.launched = true;
        if (loop.read_ahead_resources)
            PerfStatsCollector::addCounter("ngram", "read_ahead_complete_parent_submissions", 1.0,
                "decode", state_.device_id.toString(),
                {{"scope", "complete_native_parent"},
                 {"native_family_identity", std::to_string(loop.read_ahead_resources->captureIdentity())}});


        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_loop_graph_launches",
            1.0,
            "decode",
            state_.device_id.toString(),
             {{"sparse_query_program", sparseQueryGraphProgram() == SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
                 {"backend", loop.capture->backendName()},
              {"requests", std::to_string(request_count)},
              {"draft_depth", std::to_string(loop.draft_depth)},
              {"minimum_draft_depth",
               std::to_string(loop.minimum_draft_depth)},
              {"maximum_draft_depth",
               std::to_string(loop.maximum_draft_depth)},
              {"fragments", std::to_string(loop.fragment_count)},
              {"conditional_fragments",
               std::to_string(loop.native_conditional_fragment_count)},
              {"selector_gated_fragments",
               std::to_string(
                   loop.native_selector_gated_fragment_count)},
              {"ticket_conditioned_fragments",
               std::to_string(loop.ticket_conditioned_fragment_count)},
              {"execution",
               loop.minimum_draft_depth != loop.maximum_draft_depth
                   ? "single_async_native_selector_while_launch"
                   : "single_async_native_while_launch"}});
        return true;
    }

    std::span<int> DeviceGraphOrchestrator::deviceGenerationTerminalHostControls(
        int request_count)
    {
        if (request_count <= 0 || !device_generation_terminal_host_scratch_ ||
            !device_generation_terminal_host_scratch_->meta ||
            !device_generation_terminal_host_scratch_->canServe(
                request_count,
                device_generation_storage_.response_token_stride,
                device_generation_storage_.control_stride))
            return {};
        const auto prefix=localHotTerminalMetadataBytes(request_count)?sizeof(qwen4exp::local_hot::TerminalObservation)/sizeof(int):0;
        return {device_generation_terminal_host_scratch_->meta+prefix,
                static_cast<size_t>(request_count) *
                    static_cast<size_t>(device_generation_storage_.control_stride)};
    }

    bool DeviceGraphOrchestrator::observeMTPDepthTransactionTraceTerminal(
        std::span<const int, sampling_math::kDeviceGenerationControlCount> control, void *result_stream)
    {
        if (!mtp_depth_transaction_trace_dev_) return true;
        if (!result_stream || !state_.device_id.is_gpu() || !active_device_generation_admission_ ||
            !sampling_math::validMTPDepthTransactionTraceAdmission(active_device_generation_admission_->kind,
                active_device_generation_admission_->request_count, active_device_generation_admission_->max_new_tokens))
            return false;
        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend) return false;
        const int records = control[sampling_math::kDeviceGenerationControlTransactionCount];
        const size_t bytes = sampling_math::mtpDepthTransactionTraceBytes(records);
        sampling_math::MTPDiagnosticRecordStorage observation;
        if (!bytes || !backend->deviceToHostOnStream(&observation.depth_transaction_trace,
                mtp_depth_transaction_trace_dev_, bytes, state_.device_id.gpu_ordinal(), result_stream) ||
            !backend->recordEvent(device_generation_terminal_host_ready_event_.get(), state_.device_id.gpu_ordinal(), result_stream) ||
            !backend->waitForEvent(device_generation_terminal_host_ready_event_.get(), state_.device_id.gpu_ordinal()))
            return false;
        const auto &trace = observation.depth_transaction_trace;
        if (trace.magic != sampling_math::kMTPDepthTransactionTraceMagic ||
            trace.version != sampling_math::kMTPDepthTransactionTraceVersion ||
            trace.record_words != sampling_math::kMTPDepthTransactionTraceRecordWords ||
            trace.capacity != sampling_math::kMTPDepthTransactionTraceCapacity ||
            trace.initial_control[sampling_math::kDeviceGenerationControlTransactionCount] != 0)
            return false;
        const std::string identity = std::to_string(session_epoch_) + ":" +
            std::to_string(workspaceGeneration(state_.device_id)) + ":0";
        const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
        const auto lookup_mode = placement ? placement->terminalObservationMode() : ImmutableRowTableObservationMode::None;
        publishMTPDepthTransactionInitial(trace, state_.device_id.toString(), identity, "native_resident", lookup_mode);
        int prior_depth = trace.initial_control[sampling_math::kDeviceGenerationControlCurrentDraftDepth];
        int prior_response = trace.initial_control[sampling_math::kDeviceGenerationControlResponseTokenCount];
        const int64_t response_limit = static_cast<int64_t>(trace.initial_control[sampling_math::kDeviceGenerationControlResponseTokenCount]) +
            trace.initial_control[sampling_math::kDeviceGenerationControlRemainingTokenCount];
        if (response_limit <= 0 || response_limit > sampling_math::kMTPDepthTransactionTraceCapacity) return false;
        int prior_windows = trace.initial_control[sampling_math::kDeviceGenerationControlDepthEvaluatedWindows];
        int prior_leading = trace.initial_control[sampling_math::kDeviceGenerationControlNextLeadingCommittedOutputCount];
        int64_t published = trace.initial_control[sampling_math::kDeviceGenerationControlPublishedStateCommitCount];
        int last_emitted = 0;
        bool model_stopped = false;
        if ((prior_leading != 0 && prior_leading != 1) || published < 0) return false;
        for (int index = 0; index < records; ++index)
        {
            sampling_math::MTPDepthTransactionTraceValues v;
            if (!sampling_math::unpackMTPDepthTransactionTraceRecord(trace.records[index], &v) ||
                v.host_observation || v.pre_depth != prior_depth || v.token_witness <= prior_response ||
                v.token_witness > response_limit ||
                v.evaluated_windows < prior_windows || v.evaluated_windows > prior_windows + 1)
                return false;
            const int newly_emitted = v.token_witness - prior_response;
            MTPPublicationWitness publication;
            if (model_stopped || v.incoming_debt != (prior_leading != 0) ||
                newly_emitted + static_cast<int>(v.incoming_debt) > v.pre_depth + 1 ||
                !MTPPublicationWitness::fromEmitted(static_cast<uint32_t>(newly_emitted),
                    v.incoming_debt, v.outgoing_debt, v.model_stopped, &publication) ||
                (v.model_stopped && index + 1 != records)) return false;
            published += publication.publishedStateTokens();
            model_stopped = publication.modelStopped();
            prior_leading = !model_stopped && publication.outgoingDebt() ? 1 : 0;
            last_emitted = newly_emitted;
            publishMTPDepthTransactionRecord(trace.records[index], index + 1, state_.device_id.toString(), identity, "native_resident");
            prior_depth = v.post_depth; prior_response = v.token_witness; prior_windows = v.evaluated_windows;
        }
        if (prior_depth != control[sampling_math::kDeviceGenerationControlCurrentDraftDepth] ||
            prior_response != control[sampling_math::kDeviceGenerationControlResponseTokenCount] ||
            prior_windows != control[sampling_math::kDeviceGenerationControlDepthEvaluatedWindows] ||
            published != control[sampling_math::kDeviceGenerationControlPublishedStateCommitCount] ||
            prior_leading != control[sampling_math::kDeviceGenerationControlNextLeadingCommittedOutputCount] ||
            static_cast<int>(model_stopped) != control[sampling_math::kDeviceGenerationControlModelStopped] ||
            last_emitted != control[sampling_math::kDeviceGenerationControlLastTransactionEmittedTokenCount])
            return false;
        publishMTPDepthTransactionFinal(control, state_.device_id.toString(), identity, "native_resident", lookup_mode);
        return true;
    }

    bool DeviceGraphOrchestrator::finishDeviceResidentGeneration(DeviceGenerationTerminalResult* out_result)
    {
        return finishDeviceResidentGenerationImpl(out_result,RetainedStorageTerminalPhase::PublishAndAwait);
    }

    bool DeviceGraphOrchestrator::finishDeviceResidentGenerationImpl(
        DeviceGenerationTerminalResult *out_result,RetainedStorageTerminalPhase phase)
    {
        using namespace sampling_math;
        if (out_result)
            *out_result = DeviceGenerationTerminalResult{};
        const int request_count =
            device_generation_storage_.active_request_count;
        if (!out_result || !state_.device_id.is_gpu() || request_count <= 0 ||
            !active_device_generation_admission_ ||
            active_device_generation_admission_->request_count != request_count ||
            !device_generation_storage_.validFor(request_count) ||
            !device_generation_terminal_host_scratch_ ||
            !device_generation_terminal_host_scratch_->canServe(
                request_count,
                device_generation_storage_.response_token_stride,
                device_generation_storage_.control_stride) ||
            !device_generation_terminal_host_ready_event_ ||
            !stochastic_outcome_response_bridge_stream_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal device-generation result has incomplete lifecycle storage"
                      << " device=" << state_.device_id.toString()
                      << " requests=" << request_count
                      << " storage_valid="
                      << device_generation_storage_.validFor(request_count)
                      << " pinned_scratch="
                      << static_cast<bool>(
                             device_generation_terminal_host_scratch_)
                      << " result_stream="
                      << stochastic_outcome_response_bridge_stream_.get());
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        void *const result_stream =
            stochastic_outcome_response_bridge_stream_.get();
        if (!backend ||
            !consumeDeviceGenerationStateReady(
                result_stream,
                DeviceTimelineRole::HostResultBridge,
                request_count,
                "terminal_device_generation_response"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal response could not consume the final device-generation publication on "
                      << state_.device_id.toString());
            return false;
        }

        const DeviceMoECurrentBatchLLEPEvidenceSource llep_evidence_source =
            graph_builder_
                ? graph_builder_->deviceMoECurrentBatchLLEPEvidenceSource(
                      state_.device_id)
                : DeviceMoECurrentBatchLLEPEvidenceSource{};
        if (llep_evidence_source.valid() &&
            !backend->
                enqueuePublishMoECurrentBatchLLEPEvidenceToGenerationControl(
                    llep_evidence_source.runtime_layers_device,
                    llep_evidence_source.layer_count,
                    device_generation_storage_.control_device,
                    device_generation_storage_.control_stride,
                    request_count,
                    state_.device_id.gpu_ordinal(),
                    result_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal response failed to enqueue current-batch LLEP evidence publication on "
                      << state_.device_id.toString());
            return false;
        }

        const size_t response_elements =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(
                device_generation_storage_.response_token_stride);
        const size_t control_elements =
            static_cast<size_t>(request_count) *
            static_cast<size_t>(device_generation_storage_.control_stride);
        int32_t *const host_response_tokens =
            device_generation_terminal_host_scratch_->output_tokens;
        const auto localTerminalBytes=localHotTerminalMetadataBytes(request_count);
        const auto localPrefixWords=localTerminalBytes?sizeof(qwen4exp::local_hot::TerminalObservation)/sizeof(int):0;
        int *const host_controls=device_generation_terminal_host_scratch_->meta+localPrefixWords;
        const void* terminalControlSource=device_generation_storage_.control_device;
        std::size_t terminalControlBytes=sizeof(int)*control_elements;
        if(localTerminalBytes){
            auto* output=arena_->getDevicePtr(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION,state_.device_id);
            if(!output || arena_->getCols(BufferId::MTP_LOCAL_HOT_TERMINAL_OBSERVATION)*sizeof(int)<localTerminalBytes)
                throw std::logic_error("Local terminal observation lost its admitted arena output");
            // Private terminal-only output has no graph consumer. The exact
            // result stream and SAME ready event publish pack+copy together;
            // no additional coherence event or live observation is created.
            packLocalHotTerminalObservation(request_count,output,localTerminalBytes,result_stream);
            terminalControlSource=output;terminalControlBytes=localTerminalBytes;
        }

        {
            PerfStatsCollector::ScopedTimer enqueue_timer(
                "mtp",
                "device_generation_terminal_d2h_enqueue",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_count)}});
            if (!backend->deviceToHostOnStream(
                    host_response_tokens,
                    device_generation_storage_.response_tokens_device,
                    sizeof(int32_t) * response_elements,
                    state_.device_id.gpu_ordinal(),
                    result_stream) ||
                !backend->deviceToHostOnStream(
                    localTerminalBytes?device_generation_terminal_host_scratch_->meta:host_controls,
                    terminalControlSource,
                    terminalControlBytes,
                    state_.device_id.gpu_ordinal(),
                    result_stream) ||
                !backend->recordEvent(
                    device_generation_terminal_host_ready_event_.get(),
                    state_.device_id.gpu_ordinal(),
                    result_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Terminal response failed to enqueue its response/control copies on "
                          << state_.device_id.toString());
                return false;
            }
        }
        {
            PerfStatsCollector::ScopedTimer wait_timer(
                "mtp",
                "device_generation_terminal_d2h_wait",
                "decode",
                state_.device_id.toString(),
                {{"requests", std::to_string(request_count)}});
            if (!backend->waitForEvent(
                    device_generation_terminal_host_ready_event_.get(),
                    state_.device_id.gpu_ordinal()))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Terminal response event wait failed on "
                          << state_.device_id.toString());
                return false;
            }
        }

        // The existing terminal-result wait above closes the complete parent,
        // including its side-stream join. Query only this original family event
        // here; never add a token/transaction wait or infer retirement from a
        // cancelled host request.
        if (const auto& resources = deviceGenerationLoopGraph().read_ahead_resources)
        {
            try {
                const bool retiring = resources->hasOutstandingSubmission();
                if (retiring && !resources->observeTerminal())
                    throw std::logic_error("Read-ahead parent terminal remained pending after the original result join");
                if (retiring)
                    PerfStatsCollector::addCounter("ngram", "read_ahead_terminal_retirements", 1.0,
                        "decode", state_.device_id.toString(),
                        {{"scope", "complete_request_terminal"},
                         {"native_family_identity", std::to_string(resources->captureIdentity())}});
            } catch (...) { std::terminate(); }
        }

        if(phase==RetainedStorageTerminalPhase::PublishAndAwait)
            awaitRetainedStorageTerminal();

        /*
         * The first speculative transaction is retained in arena-owned device
         * storage when phase snapshots are explicitly requested.  Observe it
         * only after the terminal publication event has completed: this is an
         * intentional diagnostic D2H boundary, never part of graph dispatch or
         * inference authority.  Keeping the readback beside the ordinary
         * terminal bridge lets successful runs distinguish grouped-forward,
         * sampling, and reducer faults without manufacturing a mirrored-rank
         * failure or walking the much larger KV/tensor digest inventory.
         */
        const bool terminal_transaction_diagnostic_requested =
            DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_TERMINAL_TRANSACTION_DIAGNOSTICS");
        if (terminal_transaction_diagnostic_requested)
        {
            using Identity =
                sampling_math::MTPCommittedVerifierIdentityRecord;
            Identity identity{};
            const bool identity_available =
                mtp_committed_verifier_identity_dev_ &&
                backend->deviceToHostFast(
                    &identity,
                    mtp_committed_verifier_identity_dev_,
                    sizeof(identity),
                    state_.device_id.gpu_ordinal(),
                    result_stream) &&
                identity.valid == 1 &&
                identity.version ==
                    sampling_math::kMTPCommittedVerifierIdentityVersion &&
                identity.transaction_count > 0 &&
                identity.draft_depth > 0 &&
                identity.draft_depth <=
                    sampling_math::kSpeculativeBatchMaxRows;
            if (!identity_available)
            {
                LOG_WARN("[MTPTerminalTransactionDiagnostic] unavailable on "
                         << state_.device_id.toString());
            }
            else
            {
                const int verifier_rows = identity.draft_depth + 1;
                std::vector<int32_t> verifier_inputs(
                    static_cast<size_t>(verifier_rows), -1);
                std::vector<int32_t> verifier_targets(
                    static_cast<size_t>(verifier_rows), -1);
                std::vector<int32_t> compact_tokens(
                    static_cast<size_t>(verifier_rows), -1);
                std::array<
                    int32_t,
                    sampling_math::kSpeculativeBatchMetaCount>
                    compact_meta{};
                std::array<int32_t, 4> publication_meta{};
                const auto &logical_state =
                    device_resident_logical_sequence_state_storage_;
                const bool payload_available =
                    mtp_verifier_input_tokens_dev_ &&
                    stochastic_verify_tokens_dev_ &&
                    stochastic_batch_output_tokens_dev_ &&
                    stochastic_batch_output_meta_dev_ &&
                    logical_state.validFor(/*request_count=*/1) &&
                    backend->deviceToHostFast(
                        verifier_inputs.data(),
                        mtp_verifier_input_tokens_dev_,
                        verifier_inputs.size() * sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        verifier_targets.data(),
                        stochastic_verify_tokens_dev_,
                        verifier_targets.size() * sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        compact_tokens.data(),
                        stochastic_batch_output_tokens_dev_,
                        compact_tokens.size() * sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        compact_meta.data(),
                        stochastic_batch_output_meta_dev_,
                        compact_meta.size() * sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    /*
                     * These arena rows are not contiguous as a four-field
                     * tuple when request capacity exceeds one. Copy each
                     * scalar independently so the diagnostic reflects the
                     * exact publication record consumed by the KV commit.
                     */
                    backend->deviceToHostFast(
                        &publication_meta[0],
                        logical_state.target_cached_tokens_device,
                        sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        &publication_meta[1],
                        logical_state.accepted_state_counts_device,
                        sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        &publication_meta[2],
                        logical_state.next_condition_tokens_device,
                        sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream) &&
                    backend->deviceToHostFast(
                        &publication_meta[3],
                        logical_state.publication_ok_flags_device,
                        sizeof(int32_t),
                        state_.device_id.gpu_ordinal(),
                        result_stream);
                if (!payload_available)
                {
                    LOG_WARN("[MTPTerminalTransactionDiagnostic] payload unavailable on "
                             << state_.device_id.toString());
                }
                else
                {
                    auto append_ints = [](std::ostringstream &out,
                                          const int32_t *values,
                                          int count)
                    {
                        out << '[';
                        for (int index = 0; index < count; ++index)
                        {
                            if (index != 0)
                                out << ',';
                            out << values[index];
                        }
                        out << ']';
                    };
                    std::ostringstream detail;
                    detail
                        << "[MTPTerminalTransactionDiagnostic] device="
                        << state_.device_id.toString()
                        << " transaction=" << identity.transaction_count
                        << " draft_depth=" << identity.draft_depth
                        << " committed_identity=";
                    append_ints(
                        detail,
                        identity.verifier_input_tokens,
                        verifier_rows);
                    detail << " live_verifier_inputs=";
                    append_ints(
                        detail,
                        verifier_inputs.data(),
                        verifier_rows);
                    detail << " verifier_targets=";
                    append_ints(
                        detail,
                        verifier_targets.data(),
                        verifier_rows);
                    detail << " compact_tokens=";
                    append_ints(
                        detail,
                        compact_tokens.data(),
                        verifier_rows);
                    detail << " compact_meta=";
                    append_ints(
                        detail,
                        compact_meta.data(),
                        sampling_math::kSpeculativeBatchMetaCount);
                    detail << " publication={target_cached_tokens="
                           << publication_meta[0]
                           << ",accepted_state_count="
                           << publication_meta[1]
                           << ",next_condition_token="
                           << publication_meta[2]
                           << ",ok=" << publication_meta[3] << '}';
                    LOG_INFO(detail.str());
                }
            }
        }

        const bool terminal_layer_diagnostics =
            DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_MIRROR_LAYER_DIAGNOSTICS");
        if (terminal_layer_diagnostics ||
            terminal_transaction_diagnostic_requested)
        {
            /*
             * The terminal result event above has closed every retained
             * transaction submitted for this request. It is therefore safe
             * for this opt-in diagnostic to inspect graph-owned checkpoint
             * rows without introducing a host edge into inference. Emit one
             * stable record per row so grouped verifier M>1 can be compared
             * with the independent serial-M1 oracle by semantic boundary.
             */
            for (const auto &digest :
                 captureMirroredMTPDigests(result_stream))
            {
                const bool layer_digest =
                    terminal_layer_diagnostics &&
                    (digest.name.starts_with("main_checkpoint_") ||
                     digest.name == "terminal_hidden");
                const bool state_digest =
                    terminal_transaction_diagnostic_requested &&
                    (digest.name.starts_with("main_live_kv_") ||
                     digest.name.starts_with("main_live_gdn_") ||
                     digest.name == "main_live_position");
                if (!layer_digest && !state_digest)
                    continue;
                LOG_INFO(
                    (layer_digest
                         ? "[MTPTerminalLayerDigest] device="
                         : "[MTPTerminalStateDigest] device=")
                    << state_.device_id.toString()
                    << " name=" << digest.name
                    << " available="
                    << (digest.available ? "true" : "false")
                    << " bytes=" << digest.byte_count
                    << " hash=0x" << std::hex << digest.hash << std::dec
                    << (digest.value_preview.empty()
                            ? std::string{}
                            : " values=" + digest.value_preview));
            }
        }

        const bool first_transaction_diagnostic_requested = mtp_diagnostic_record_mode_ == sampling_math::MTPDiagnosticRecordMode::FirstTransaction;
        if (first_transaction_diagnostic_requested &&
            mtp_first_transaction_diagnostic_dev_)
        {
            sampling_math::MTPFirstTransactionDiagnosticRecord record{};
            const bool record_available =
                backend->deviceToHostFast(
                    &record,
                    mtp_first_transaction_diagnostic_dev_,
                    sizeof(record),
                    state_.device_id.gpu_ordinal(),
                    result_stream) &&
                record.valid == 1 &&
                record.version ==
                    sampling_math::kMTPFirstTransactionDiagnosticVersion &&
                record.transaction_count == 0 &&
                record.comparison_row_count > 0 &&
                record.comparison_row_count <=
                    sampling_math::kSpeculativeBatchMaxRows &&
                record.top_k > 0 &&
                record.top_k <= sampling_math::kMaxTopK;
            if (!record_available)
            {
                std::ostringstream partial;
                partial << "[MTPFirstTransactionDiagnostic] unavailable on "
                        << state_.device_id.toString()
                        << " valid=" << record.valid
                        << " version=" << record.version
                        << " transaction=" << record.transaction_count
                        << " rows=" << record.comparison_row_count
                        << " top_k=" << record.top_k
                        << " draft_depth="
                        << record.draft_diagnostic_depth
                        << " draft_conditions=[";
                const int retained_draft_depth = std::clamp(
                    record.draft_diagnostic_depth,
                    0,
                    sampling_math::kSpeculativeBatchMaxRows);
                for (int slot = 0; slot < retained_draft_depth; ++slot)
                {
                    if (slot != 0)
                        partial << ',';
                    partial << record.draft_condition_tokens[slot];
                }
                partial << "] draft_positions=[";
                for (int slot = 0; slot < retained_draft_depth; ++slot)
                {
                    if (slot != 0)
                        partial << ',';
                    partial << record.draft_position_ids[slot];
                }
                partial << ']';
                LOG_WARN(partial.str());
            }
            else
            {
                const int sample_rows = record.comparison_row_count + 1;
                auto append_ints = [](std::ostringstream &out,
                                      const int32_t *values,
                                      int count)
                {
                    out << '[';
                    for (int index = 0; index < count; ++index)
                    {
                        if (index != 0)
                            out << ',';
                        out << values[index];
                    }
                    out << ']';
                };
                std::ostringstream detail;
                detail << "[MTPFirstTransactionDiagnostic] device="
                       << state_.device_id.toString()
                       << " seed=0x" << std::hex << record.threshold_seed
                       << std::dec
                       << " base_position="
                       << record.threshold_base_position
                       << " position_offset="
                       << record.threshold_position_offset
                       << " comparison_rows="
                       << record.comparison_row_count
                       << " top_k=" << record.top_k
                       << " budget="
                       << record.transaction_commit_budget
                       << " leading="
                       << record.leading_committed_output_count
                       << " verifier_inputs=";
                append_ints(
                    detail,
                    record.verifier_input_tokens,
                    sample_rows);
                detail << " sampled_targets=";
                append_ints(
                    detail,
                    record.sampled_target_tokens,
                    sample_rows);
                detail << " matches=";
                append_ints(
                    detail,
                    record.sampled_matches_verifier_input,
                    record.comparison_row_count);
                detail << " output_tokens=";
                append_ints(detail, record.output_tokens, sample_rows);
                detail << " output_meta=";
                append_ints(
                    detail,
                    record.output_meta,
                    sampling_math::kSpeculativeBatchMetaCount);
                detail << " draft_conditions=";
                append_ints(
                    detail,
                    record.draft_condition_tokens,
                    std::clamp(
                        record.draft_diagnostic_depth,
                        0,
                        sampling_math::kSpeculativeBatchMaxRows));
                detail << " draft_positions=";
                append_ints(
                    detail,
                    record.draft_position_ids,
                    std::clamp(
                        record.draft_diagnostic_depth,
                        0,
                        sampling_math::kSpeculativeBatchMaxRows));
                LOG_INFO(detail.str());
            }
        }

        DeviceGenerationTerminalResult parsed;
        parsed.device = state_.device_id;
        parsed.requests.reserve(static_cast<size_t>(request_count));
        int total_response_tokens = 0;
        int total_transactions = 0;
        int total_accepted_speculative_tokens = 0;
        int total_rejected_transactions = 0;
        int total_consumed_verifier_rows = 0;
        int total_published_state_commits = 0;
        int total_attempted_draft_tokens = 0;
        int total_verifier_tokens = 0;
        int total_current_batch_llep_movement_layers = 0;
        int total_current_batch_llep_non_owner_assignment_layers = 0;
        int total_depth_evaluated_windows = 0;
        int total_depth_updates = 0;
        int total_depth_promotions = 0;
        int total_depth_demotions = 0;
        std::vector<qwen4exp::QwenSparseCounterTotals> qsa_terminal_totals;
        qsa_terminal_totals.reserve(request_count);
        std::vector<qwen4exp::QwenFixedHeadCounterTotals> fixed_head_terminal_totals;
        fixed_head_terminal_totals.reserve(request_count);
        std::vector<qwen4exp::QwenAdaptiveCounterTotals> adaptive_terminal_totals;
        adaptive_terminal_totals.reserve(request_count);
        const auto placement = graph_builder_ ? graph_builder_->config().ngram_table_placement : nullptr;
        using LookupObservation = ImmutableRowTableObservationMode;
        const auto lookup_observation = placement ? placement->terminalObservationMode() : LookupObservation::None;
        const bool adaptive_lookup =
            lookup_observation == LookupObservation::GPUAdaptive &&
            std::find(placement->consumers().begin(), placement->consumers().end(),
                      PhysicalMemoryAllocatorIdentity{placement->worldRank(), state_.device_id}) !=
                placement->consumers().end();
        const auto adaptive_row_bytes = adaptive_lookup ? placement->extent().rowBytes() : 0;
        const bool compact_lookup = adaptive_lookup && placement->gpuAdaptiveRequest().local_hot_cache.has_value();
        const auto adaptive_query = compact_lookup ? qwen4exp::QwenAdaptiveQuerySemantics::CompactNativeMisses
                                                  : qwen4exp::QwenAdaptiveQuerySemantics::OriginalNgramRows;
        // The original coordinator consumes concatenated compact miss prefixes;
        // its N envelope may exceed one query's U bound. Neither is wire volume.
        const auto adaptive_maximum_ids = !adaptive_lookup ? 0 : compact_lookup
            ? qwen4exp::PooledImmutableLocalHotTransportGeometry::compactOriginalCapacity(
                static_cast<std::uint32_t>(placement->gpuAdaptiveMaximumLiveRows()),placement->consumers().size())
            : placement->gpuAdaptiveMaximumLiveRows();
        const bool fixed_head_lookup = lookup_observation == LookupObservation::FixedHeadStandalone;
        const auto fixed_row_bytes=fixed_head_lookup && graph_builder_->config().flash_next
            ? graph_builder_->config().flash_next->ngrams().row_bytes : 0;
        const auto fixed_participants=fixed_head_lookup
            ? graph_builder_->config().ngram_table_placement->consumers().size() : 0;
        if(fixed_head_lookup && (!fixed_row_bytes || (fixed_participants!=2 && fixed_participants!=4 && fixed_participants!=8)))return false;
        for (int request_index = 0;
             request_index < request_count;
             ++request_index)
        {
            const int *const control =
                host_controls +
                static_cast<size_t>(request_index) *
                    static_cast<size_t>(
                        device_generation_storage_.control_stride);
            const int response_count =
                control[kDeviceGenerationControlResponseTokenCount];
            const int remaining_count =
                control[kDeviceGenerationControlRemainingTokenCount];
            const int model_stopped =
                control[kDeviceGenerationControlModelStopped];
            const int request_complete =
                control[kDeviceGenerationControlRequestComplete];
            const int transaction_count =
                control[kDeviceGenerationControlTransactionCount];
            const int next_leading_count =
                control[
                    kDeviceGenerationControlNextLeadingCommittedOutputCount];
            const int transaction_budget =
                control[
                    kDeviceGenerationControlTransactionCommitBudget];
            const int accepted_count =
                control[
                    kDeviceGenerationControlAcceptedSpeculativeTokenCount];
            const int rejected_count =
                control[
                    kDeviceGenerationControlRejectedTransactionCount];
            const int consumed_rows =
                control[
                    kDeviceGenerationControlConsumedVerifierRowCount];
            const int published_state_commits =
                control[
                    kDeviceGenerationControlPublishedStateCommitCount];
            const int attempted_draft_tokens =
                control[
                    kDeviceGenerationControlAttemptedDraftTokenCount];
            const int verifier_tokens =
                control[kDeviceGenerationControlVerifierTokenCount];
            const int last_transaction_draft_depth =
                control[
                    kDeviceGenerationControlLastTransactionDraftDepth];
            const int last_transaction_emitted_token_count =
                control[
                    kDeviceGenerationControlLastTransactionEmittedTokenCount];
            const int current_batch_llep_movement_layers =
                control[
                    kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount];
            const int current_batch_llep_non_owner_assignment_layers =
                control[
                    kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount];
            const int final_draft_depth =
                control[kDeviceGenerationControlCurrentDraftDepth];
            const int minimum_draft_depth =
                control[kDeviceGenerationControlMinimumDraftDepth];
            const int maximum_draft_depth =
                control[kDeviceGenerationControlMaximumDraftDepth];
            const int depth_policy_mode =
                control[kDeviceGenerationControlDepthPolicyMode];
            const int depth_window_size =
                control[kDeviceGenerationControlDepthWindowSize];
            const int depth_minimum_samples =
                control[kDeviceGenerationControlDepthMinimumSamples];
            const int depth_window_verifier_runs =
                control[kDeviceGenerationControlDepthWindowVerifierRuns];
            const int depth_window_attempted_tokens =
                control[kDeviceGenerationControlDepthWindowAttemptedTokens];
            const int depth_evaluated_windows =
                control[kDeviceGenerationControlDepthEvaluatedWindows];
            const int depth_updates =
                control[kDeviceGenerationControlDepthUpdates];
            const int depth_promotions =
                control[kDeviceGenerationControlDepthPromotions];
            const int depth_demotions =
                control[kDeviceGenerationControlDepthDemotions];
            const int error_code =
                control[kDeviceGenerationControlErrorCode];

            // One shared contract owns both algorithms. Authentication uses
            // the original admission, not policy bytes copied from this result.
            const DeviceGenerationTerminalError terminal_error =
                validateDeviceGenerationTerminal(
                    std::span<const int, kDeviceGenerationControlCount>(
                        control, kDeviceGenerationControlCount),
                    *active_device_generation_admission_,
                    device_generation_storage_.response_token_stride,
                    request_index,
                    llep_evidence_source.valid() ? llep_evidence_source.layer_count : 0);
            if (terminal_error != DeviceGenerationTerminalError::None)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Terminal device-generation controller is invalid"
                          << " device=" << state_.device_id.toString()
                          << " request=" << request_index
                          << " terminal_contract_error=" << static_cast<int>(terminal_error)
                          << " ok="
                          << control[kDeviceGenerationControlOk]
                          << " complete=" << request_complete
                          << " error=" << error_code
                          << " response_count=" << response_count
                          << " remaining=" << remaining_count
                          << " model_stopped=" << model_stopped
                          << " transactions=" << transaction_count
                          << " next_leading=" << next_leading_count
                          << " transaction_budget=" << transaction_budget
                          << " accepted=" << accepted_count
                          << " rejected=" << rejected_count
                          << " consumed_rows=" << consumed_rows
                          << " state_commits=" << published_state_commits
                          << " attempted_drafts=" << attempted_draft_tokens
                          << " verifier_tokens=" << verifier_tokens
                          << " last_transaction_draft_depth="
                          << last_transaction_draft_depth
                          << " last_transaction_emitted_token_count="
                          << last_transaction_emitted_token_count
                          << " llep_movement_layers="
                          << current_batch_llep_movement_layers
                          << " llep_non_owner_assignment_layers="
                          << current_batch_llep_non_owner_assignment_layers
                          << " final_depth=" << final_draft_depth
                          << " depth_range=[" << minimum_draft_depth << ','
                          << maximum_draft_depth << ']'
                          << " depth_policy=" << depth_policy_mode
                          << " depth_window_size=" << depth_window_size
                          << " depth_minimum_samples="
                          << depth_minimum_samples
                          << " pending_depth_window_runs="
                          << depth_window_verifier_runs
                          << " pending_depth_window_tokens="
                          << depth_window_attempted_tokens
                          << " evaluated_windows="
                          << depth_evaluated_windows
                          << " depth_updates=" << depth_updates
                          << " promotions=" << depth_promotions
                          << " demotions=" << depth_demotions);
                return false;
            }

            if (!observeMTPDepthTransactionTraceTerminal(
                    std::span<const int, sampling_math::kDeviceGenerationControlCount>(control, sampling_math::kDeviceGenerationControlCount),
                    result_stream)) return false;

            qwen4exp::QwenSparseCounterTotals qsa_actual;
            if(!qwen4exp::readQwenSparseTerminalCounters(control,kDeviceGenerationControlCount,qsa_actual))return false;
            qsa_terminal_totals.push_back(qsa_actual);
            // The three native source policies exclusively own the same
            // immutable observation interval. Pipeline generation projects
            // its actual original participants at their common terminal.
            if (lookup_observation != LookupObservation::FixedHeadPipeline) {
                qwen4exp::QwenFixedHeadNativeSendTotals fixed_native_sends;
                if (!qwen4exp::readQwenFixedHeadNativeSendCounters(control,kDeviceGenerationControlCount,fixed_native_sends))
                    return false;
                qwen4exp::QwenFixedHeadCounterTotals fixed_actual;
                if (fixed_head_lookup) {
                    if (!qwen4exp::projectQwenFixedHeadTerminalContributions(fixed_native_sends,
                        static_cast<int>(fixed_participants),fixed_row_bytes,fixed_actual) || fixed_actual.failures) return false;
                } else if (fixed_native_sends != qwen4exp::QwenFixedHeadNativeSendTotals{}) return false;
                fixed_head_terminal_totals.push_back(fixed_actual);
                qwen4exp::QwenAdaptiveCounterTotals adaptive_actual;
                if (!qwen4exp::readQwenAdaptiveCounters(control,kDeviceGenerationControlCount,adaptive_actual)) return false;
                if (adaptive_lookup) {
                    if (!qwen4exp::validQwenAdaptiveCounters(adaptive_actual, adaptive_row_bytes,
                            adaptive_maximum_ids, adaptive_query) || adaptive_actual.failed_frames) return false;
                } else if (adaptive_actual != qwen4exp::QwenAdaptiveCounterTotals{}) return false;
                adaptive_terminal_totals.push_back(adaptive_actual);
            }

            DeviceGenerationTerminalRequestResult request_result;
            request_result.remaining_token_count = remaining_count;
            request_result.model_stopped = model_stopped != 0;
            request_result.next_leading_row_disposition =
                static_cast<
                    sampling_math::DeviceGenerationLeadingRowDisposition>(
                    next_leading_count);
            request_result.transaction_count = transaction_count;
            request_result.accepted_speculative_token_count =
                accepted_count;
            request_result.rejected_transaction_count = rejected_count;
            request_result.consumed_verifier_row_count = consumed_rows;
            request_result.published_state_commit_count =
                published_state_commits;
            request_result.attempted_draft_token_count =
                attempted_draft_tokens;
            request_result.verifier_token_count = verifier_tokens;
            request_result.last_transaction_draft_depth =
                last_transaction_draft_depth;
            request_result.last_transaction_emitted_token_count =
                last_transaction_emitted_token_count;
            request_result.final_draft_depth = final_draft_depth;
            request_result.depth_evaluated_window_count =
                depth_evaluated_windows;
            request_result.depth_update_count = depth_updates;
            request_result.depth_promotion_count = depth_promotions;
            request_result.depth_demotion_count = depth_demotions;
            request_result.depth_utility_probe_count = control[kDeviceGenerationControlDepthUtilityProbes];
            request_result.depth_utility_keep_count = control[kDeviceGenerationControlDepthUtilityKeeps];
            request_result.depth_utility_return_count = control[kDeviceGenerationControlDepthUtilityReturns];
            request_result.tokens.reserve(
                static_cast<size_t>(response_count));
            const int32_t *const response_row =
                host_response_tokens +
                static_cast<size_t>(request_index) *
                    static_cast<size_t>(
                        device_generation_storage_.response_token_stride);
            for (int token_index = 0;
                 token_index < response_count;
                 ++token_index)
            {
                const int32_t token = response_row[token_index];
                if (token < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Terminal device-generation response contains an invalid token"
                              << " request=" << request_index
                              << " token_index=" << token_index
                              << " token=" << token);
                    return false;
                }
                request_result.tokens.push_back(token);
            }
            total_response_tokens += response_count;
            total_transactions += transaction_count;
            total_accepted_speculative_tokens += accepted_count;
            total_rejected_transactions += rejected_count;
            total_consumed_verifier_rows += consumed_rows;
            total_published_state_commits += published_state_commits;
            total_attempted_draft_tokens += attempted_draft_tokens;
            total_verifier_tokens += verifier_tokens;
            total_current_batch_llep_movement_layers +=
                current_batch_llep_movement_layers;
            total_current_batch_llep_non_owner_assignment_layers +=
                current_batch_llep_non_owner_assignment_layers;
            total_depth_evaluated_windows += depth_evaluated_windows;
            total_depth_updates += depth_updates;
            total_depth_promotions += depth_promotions;
            total_depth_demotions += depth_demotions;
            /*
             * The terminal bridge already reads this compact controller row
             * to return the public response.  Publish the immutable policy and
             * its residual window beside that existing readback so a parity or
             * production diagnosis can distinguish a clipped observation from
             * a stale/default policy without adding another device transfer.
             * Residual accepted/rejected/full/zero/rollback counts describe
             * eligible, non-budget-clipped outcomes; physical draft-work totals
             * remain separate and must not be used as their denominator.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_terminal_depth_policy_state",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"request", std::to_string(request_index)},
                 {"mode", std::to_string(depth_policy_mode)},
                 {"minimum_depth", std::to_string(minimum_draft_depth)},
                 {"maximum_depth", std::to_string(maximum_draft_depth)},
                 {"final_depth", std::to_string(final_draft_depth)},
                 {"window_size", std::to_string(depth_window_size)},
                 {"minimum_samples", std::to_string(depth_minimum_samples)},
                 {"pending_window_runs",
                  std::to_string(depth_window_verifier_runs)},
                 {"pending_window_tokens",
                  std::to_string(depth_window_attempted_tokens)},
                 {"pending_window_accepted_tokens", std::to_string(
                      control[kDeviceGenerationControlDepthWindowAcceptedTokens])},
                 {"pending_window_rejected_tokens", std::to_string(
                      control[kDeviceGenerationControlDepthWindowRejectedTokens])},
                 {"pending_window_full_accepts", std::to_string(
                      control[kDeviceGenerationControlDepthWindowFullAccepts])},
                 {"pending_window_zero_accepts", std::to_string(
                      control[kDeviceGenerationControlDepthWindowZeroAccepts])},
                 {"pending_window_rollbacks", std::to_string(
                      control[kDeviceGenerationControlDepthWindowRollbacks])},
                 {"depth_policy_utility_probes", std::to_string(control[kDeviceGenerationControlDepthUtilityProbes])},
                 {"depth_policy_utility_keeps", std::to_string(control[kDeviceGenerationControlDepthUtilityKeeps])},
                 {"depth_policy_utility_returns", std::to_string(control[kDeviceGenerationControlDepthUtilityReturns])},
                 {"evaluated_windows",
                  std::to_string(depth_evaluated_windows)},
                 {"learned_matched_windows", std::to_string(
                      control[kDeviceGenerationControlLearnedDepthMatchedWindows])}});
            parsed.requests.push_back(std::move(request_result));
        }

        if (!parsed.valid())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Parsed terminal device-generation result is incomplete on "
                      << state_.device_id.toString());
            return false;
        }

        // The terminal already authenticated these consumed rows. A composed
        // parent bypasses host forward dispatch, so report its actual child
        // executions here without submitting work or inspecting more GPU state.
        if (active_device_generation_admission_->depth_policy.isOrdinary() &&
            !observeOrdinaryGenerationForwardCompletion(total_published_state_commits))
            return false;

        // Ordinary budget-one continuations can execute an MTP-bound main
        // graph too. Its exact retained signature, rather than enabled policy
        // or emitted tokens, authenticates that the canonical copy was present.
        const auto &completed_loop = deviceGenerationLoopGraph();
        const bool canonical_main_producer =
            !active_device_generation_admission_->depth_policy.isOrdinary() ||
            (completed_loop.ordinary_forward_identity &&
             completed_loop.ordinary_forward_identity->mtp_main_terminal_logits_capture_identity != 0);
        if (request_count == 1 && canonical_main_producer &&
            !publishCanonicalMTPMainLogits(result_stream, "completed_main_parent_terminal"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Completed parent lost its canonical terminal-logit publication");
            return false;
        }

        /*
         * Validation above is the sole release gate. The device-generation loop
         * has already committed every accepted KV/recurrent/logical-state row
         * and published the final mailbox on its exact stream. The terminal
         * bridge is an observer only: it must neither advance the host epoch nor
         * reset captured replay state. Keeping the mailbox live permits a later
         * device-resident continuation to consume that final publication; an
         * explicit request/session reset owns retirement and epoch mutation.
         *
         * Clearing this host-side controller-admission marker therefore does
         * not transfer state authority to the CPU. It merely releases the
         * persistent response ledger after its terminal contents were proven.
         */
        if ((!hosted_device_generation_cursor_.inactive() &&
             !hosted_device_generation_cursor_.terminalSubmitted()) ||
            !deviceGenerationLoopGraph().retireCompletedLaunch())
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal device-generation response could not retire one complete loop lifecycle on "
                      << state_.device_id.toString());
            return false;
        }
        if (!hosted_device_generation_cursor_.retireCompleted())
        {
            /* Preconditions above make this an internal transition defect. */
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal response bridge lost hosted scheduler lifecycle ownership");
            std::terminate();
        }
        if (!device_generation_state_ready_
                 .handoff.retireAfterHostCompletion(request_count))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Terminal response bridge could not retire its completed controller borrow");
            return false;
        }
        device_generation_storage_.active_request_count = 0;
        active_device_generation_admission_.reset();
        last_device_generation_dispatch_ticket_.reset();
        *out_result = std::move(parsed);
        if(localTerminalBytes)
            publishLocalHotTerminalObservation(device_generation_terminal_host_scratch_->meta,localTerminalBytes,request_count);
        // These words arrive in the existing immutable terminal controller copy.
        // Emit only after every request and complete-parent retirement succeeded.
        const char* qsa_metric_names[]{"verifier_prior_header_bytes","verifier_prior_payload_bytes",
            "verifier_candidate_payload_bytes","verifier_accepted_payload_bytes"};
        for(std::size_t request=0;request<qsa_terminal_totals.size();++request) {
            const auto& total=qsa_terminal_totals[request];
            const std::uint64_t values[]{total.prior_header,total.prior_payload,total.candidate_payload,total.accepted_payload};
            for(int metric=0;metric<4;++metric)
                PerfStatsCollector::addCounterWithSequence("qsa",qsa_metric_names[metric],static_cast<double>(values[metric]),
                    {values[metric]},"decode",state_.device_id.toString(),{{"request",std::to_string(request)},
                    {"authority","terminal_device_generation_control"}});
        }
        // The same complete terminal copy contains these observations. Every
        // pair remains device-owned until all request/parent validation and
        // retirement above succeeded; no per-stage/per-token readback occurs.
        if(fixed_head_lookup) {
            constexpr const char *names[]{"fixed_head_lookup_calls","fixed_head_live_ids","fixed_head_native_bytes",
                "fixed_head_failures","fixed_head_header_native_bytes","fixed_head_payload_native_bytes"};
            for(std::size_t request=0;request<fixed_head_terminal_totals.size();++request) {
                const auto &total=fixed_head_terminal_totals[request];
                const std::uint64_t values[]{total.calls,total.live_ids,total.native_bytes,total.failures,total.header_bytes,total.payload_bytes};
                for(int metric=0;metric<6;++metric)
                    PerfStatsCollector::addCounterWithSequence("immutable_lookup",names[metric],static_cast<double>(values[metric]),
                        {values[metric]},"generation_terminal",state_.device_id.toString(),{{"request",std::to_string(request)},
                            {"storage","vram"},{"transport","native_fixed_head"},{"authority","terminal_device_generation_control"}});
            }
        }
        // Publish only the actual owner's terminal words after the same complete
        // parent validation/retirement; source-worker statistics never infer GPU hits.
        if (adaptive_lookup)
        {
            constexpr const char *names[]{
                "adaptive_lookup_calls",    "adaptive_completed_live_ids", "adaptive_cache_hit_rows",
                "adaptive_cache_miss_rows", "adaptive_source_rows",        "adaptive_source_native_bytes",
                "adaptive_promotions",      "adaptive_evictions",          "adaptive_failed_frames"};
            for (std::size_t request = 0; request < adaptive_terminal_totals.size(); ++request)
            {
                const auto values = qwen4exp::adaptiveCounterValues(adaptive_terminal_totals[request]);
                std::map<std::string,std::string> tags{{"request",std::to_string(request)},
                    {"storage","vram"},{"transport","pooled_subordinate"},
                    {"authority","terminal_device_generation_control"}};
                // A compact input count is shared-pool work, not original model
                // rows. Preserve that unit in the immutable terminal evidence.
                if(compact_lookup)tags.emplace("query","compact_native_misses");
                for (unsigned metric = 0; metric < 9; ++metric)
                    PerfStatsCollector::addCounterWithSequence(
                        "immutable_lookup", names[metric], static_cast<double>(values[metric]),
                        {values[metric]}, "generation_terminal", state_.device_id.toString(),
                        tags);
            }
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_response_bridges",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"d2h_operations", "2"},
             {"event_waits", "1"},
             {"stream_synchronizations", "0"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_response_tokens",
            static_cast<double>(total_response_tokens),
            "decode",
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)}});
        const PerfStatsCollector::Tags terminal_multiplier_tags{
            {"requests", std::to_string(request_count)},
            {"accounting_role", "captured_graph_replay_multiplier"},
            {"authority", "device_generation_controller"}};
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_transactions",
            static_cast<double>(total_transactions),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        /*
         * A device-resident transaction is committed only after
         * append_speculative_outcome_to_device_generation() consumes one valid
         * compact reducer row. Publish that causal identity explicitly. Native
         * conditional replay and HIP ticket-scheduled replay differ only in
         * outer graph submission; the terminal device ledger is authoritative
         * for both paths.
         */
        PerfStatsCollector::Tags compact_reducer_tags =
            terminal_multiplier_tags;
        compact_reducer_tags.emplace(
            "source",
            deviceGenerationLoopGraph().sampling_mode ==
                    DeviceGenerationSamplingMode::Stochastic
                ? "captured_stochastic_compact_outcome"
                : "captured_greedy_compact_outcome");
        compact_reducer_tags.emplace(
            "execution",
            deviceGenerationLoopGraph().execution_kind ==
                    MTPDeviceGenerationLoopGraphCache::ExecutionKind::
                        NativeConditionalParent
                ? "native_conditional_graph"
                : "host_scheduled_captured_transactions");
        // Ordinary sampling shares the terminal ledger but executes no MTP
        // verifier/reducer. Do not manufacture speculative path evidence from
        // its ordinary transaction count.
        if (!deviceGenerationLoopGraph().ordinary_sampling_identity)
            PerfStatsCollector::addCounter(
                "mtp",
                "device_generation_terminal_compact_outcome_reductions",
                static_cast<double>(total_transactions),
                "decode",
                state_.device_id.toString(),
                compact_reducer_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_accepted_speculative_tokens",
            static_cast<double>(total_accepted_speculative_tokens),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_rejected_transactions",
            static_cast<double>(total_rejected_transactions),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_consumed_verifier_rows",
            static_cast<double>(total_consumed_verifier_rows),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_published_state_commits",
            static_cast<double>(total_published_state_commits),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_attempted_draft_tokens",
            static_cast<double>(total_attempted_draft_tokens),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_verifier_tokens",
            static_cast<double>(total_verifier_tokens),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        if (llep_evidence_source.valid())
        {
            const PerfStatsCollector::Tags llep_evidence_tags{
                {"source", "device_generation_terminal_control"},
                {"layer_count",
                 std::to_string(llep_evidence_source.layer_count)},
                {"expert_payload_slot_bytes",
                 std::to_string(
                     llep_evidence_source.expert_payload_slot_bytes)},
                {"ordering", "device_generation_state_ready_event"}};
            const auto request_movement =
                deviceMoERebalanceRequestMovementEvidence(
                    /*waves=*/{},
                    static_cast<uint32_t>(
                        total_current_batch_llep_movement_layers),
                    llep_evidence_source.expert_payload_slot_bytes);
            recordCompletedDeviceMoEMovement(request_movement, "prefill", state_.device_id.toString(), "device_generation_terminal_control");
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_prefill_current_batch_movement_layers",
                static_cast<double>(
                    total_current_batch_llep_movement_layers),
                "prefill",
                state_.device_id.toString(),
                llep_evidence_tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_prefill_current_batch_non_owner_assignment_layers",
                static_cast<double>(
                    total_current_batch_llep_non_owner_assignment_layers),
                "prefill",
                state_.device_id.toString(),
                llep_evidence_tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_request_copied_payload_lower_bound",
                static_cast<double>(
                    request_movement.copied_payload_lower_bound),
                "prefill",
                state_.device_id.toString(),
                llep_evidence_tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_request_applied_payload_lower_bound",
                static_cast<double>(
                    request_movement.applied_payload_lower_bound),
                "prefill",
                state_.device_id.toString(),
                llep_evidence_tags);
            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "device_rebalance_request_useful_payload_bytes_lower_bound",
                static_cast<double>(
                    request_movement.useful_payload_bytes_lower_bound),
                "prefill",
                state_.device_id.toString(),
                llep_evidence_tags);
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_depth_evaluated_windows",
            static_cast<double>(total_depth_evaluated_windows),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_depth_updates",
            static_cast<double>(total_depth_updates),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_depth_promotions",
            static_cast<double>(total_depth_promotions),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        PerfStatsCollector::addCounter(
            "mtp",
            "device_generation_terminal_depth_demotions",
            static_cast<double>(total_depth_demotions),
            "decode",
            state_.device_id.toString(),
            terminal_multiplier_tags);
        return true;
    }

    bool DeviceGraphOrchestrator::joinPriorDeviceWorkForRequestStateReset(
        void *reset_stream,
        const char *consumer_name,RetainedStorageTerminalPhase phase)
    {
        if (!state_.device_id.is_gpu())
            return true;

        /*
         * Server initialization and request admission may establish adjacent
         * reset boundaries before any forward graph consumes the first
         * publication.  Treat the newer reset as the explicit consumer of the
         * prior reset generation.  This queues the dependency onto the exact
         * reset stream and retires the old publication before the same
         * lifecycle-owned event is recorded for the new generation.
         */
        if (!reset_stream)
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "request_state_reset",
                "ExplicitResetStream",
                consumer_name);
        }
        if (!waitForPendingRequestStateReset(
                reset_stream,
                DeviceTimelineRole::RequestStateReset,
                consumer_name))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "request_state_reset",
                "RequestStateResetReady",
                consumer_name);
        }
        if (!waitForPendingGraphBuildDeviceStateReady(
                main_graph_build_device_state_ready_,
                reset_stream,
                DeviceTimelineRole::RequestStateReset,
                consumer_name) ||
            !waitForPendingGraphBuildDeviceStateReady(
                mtp_graph_build_device_state_ready_,
                reset_stream,
                DeviceTimelineRole::RequestStateReset,
                consumer_name))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "request_state_reset",
                "GraphBuildDeviceStateReady",
                consumer_name);
        }
        if (!waitForLiveInferenceStateReadyForObservation(
                reset_stream,
                consumer_name,
                DeviceTimelineRole::RequestStateReset))
        {
            return reportDeviceTimelineJoinFailure(
                state_.device_id,
                "request_state_reset",
                "LiveInferenceStateReady",
                consumer_name);
        }
        if (device_generation_storage_.active_request_count > 0)
        {
            if (!retireDeviceGenerationStateForRequestReset(
                    reset_stream,
                    device_generation_storage_.active_request_count,
                    consumer_name))
            {
                return reportDeviceTimelineJoinFailure(
                    state_.device_id,
                    "request_state_reset",
                    "DeviceGenerationStateReady",
                    consumer_name);
            }
            device_generation_storage_.active_request_count = 0;
            active_device_generation_admission_.reset();
            last_device_generation_dispatch_ticket_.reset();
            hosted_device_generation_cursor_.reset();
            deviceGenerationLoopGraph().hosted_advance.reset();
        }
        // Standalone prefill can own live immutable I/O without ever launching
        // a complete generation parent. Retire that interval after every old
        // producer joined reset, before clearing caches or advancing the epoch.
        try
        {
            retireRetainedStorageForRequestReset(reset_stream,phase);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Immutable storage request terminal retirement failed: " << error.what());
            return false;
        }
        PerfStatsCollector::addCounter(
            "request_reset",
            "device_prior_work_joins",
            1.0,
            "request_reset",
            state_.device_id.toString(),
            {{"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"},
             {"ordering", "event_wait"}});
        return true;
    }

    void DeviceGraphOrchestrator::publishRequestStateResetReady(
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return;

        auto &ready = request_state_reset_ready_;
        if (!ready.event || ready.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-state reset publication has invalid lifecycle ownership"
                      << " producer="
                      << (producer_name && producer_name[0] != '\0'
                              ? producer_name
                              : "unknown")
                      << " event=" << static_cast<bool>(ready.event)
                      << " already_valid=" << ready.valid);
            std::terminate();
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!producer_stream || !backend ||
            !DeviceEventEdge::at(DeviceTimelinePoint::RequestStateResetReady)
                 .from(DeviceTimelineRole::RequestStateReset)
                 .publish(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     producer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish request-state reset readiness");
            std::terminate();
        }

        ready.producer_stream = producer_stream;
        ready.valid = true;
        PerfStatsCollector::addCounter(
            "request_reset",
            "device_state_ready_events",
            1.0,
            "request_reset",
            state_.device_id.toString(),
            {{"producer",
              producer_name && producer_name[0] != '\0'
                  ? producer_name
                  : "unknown"},
             {"ordering", "event_wait"}});
    }

    bool DeviceGraphOrchestrator::waitForPendingRequestStateReset(
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu())
            return true;

        auto &ready = request_state_reset_ready_;
        if (!ready.valid)
            return true;
        if (!consumer_stream ||
            !ready.event ||
            !ready.producer_stream ||
            (consumer_role != DeviceTimelineRole::RequestStateReset &&
             consumer_role != DeviceTimelineRole::RequestAdmissionTransfer &&
             consumer_role != DeviceTimelineRole::MainForwardGraph &&
             consumer_role != DeviceTimelineRole::MTPSidecarGraph &&
             consumer_role != DeviceTimelineRole::PrefixRestoreMutation))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-state reset dependency has incomplete or invalid ownership"
                      << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !DeviceEventEdge::at(DeviceTimelinePoint::RequestStateResetReady)
                 .from(DeviceTimelineRole::RequestStateReset)
                 .to(consumer_role)
                 .enqueueWait(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     ready.producer_stream,
                     consumer_stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to order graph execution after request-state reset"
                      << " consumer="
                      << (consumer_name && consumer_name[0] != '\0'
                              ? consumer_name
                              : "unknown"));
            return false;
        }

        /*
         * Observe cache-owned recurrent banks only after the consumer stream
         * has inherited RequestStateResetReady. This diagnostic distinguishes
         * an incomplete reset from a captured graph that later addresses the
         * wrong persistent bank. It is opt-in and synchronizes only this
         * consumer stream; production request reset remains event-only.
         */
        const auto diagnostic_gdn_layer =
            mtpGraphReuseDiagnosticGDNLayer();
        if (consumer_role == DeviceTimelineRole::RequestAdmissionTransfer &&
            diagnostic_gdn_layer &&
            mtpGraphReuseDiagnosticsEnabledForContext(
                "request_state_reset_gdn"))
        {
            auto *hybrid_cache =
                dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
            const HybridGDNLayerState *gdn_state =
                hybrid_cache
                    ? hybrid_cache->getGDNState(*diagnostic_gdn_layer)
                    : nullptr;
            if (!gdn_state)
            {
                LOG_ERROR("[MTPGraphReuseDiagnostics] selected reset diagnostic layer is not GDN"
                          << " layer=" << *diagnostic_gdn_layer
                          << " device=" << state_.device_id.toString());
                return false;
            }

            std::vector<MTPGraphReuseDiagnosticSource> sources;
            const auto append_binding =
                [&](const char *owner,
                    const GDNDeviceStateBinding &binding)
            {
                const std::string prefix =
                    std::string{"layer"} +
                    std::to_string(*diagnostic_gdn_layer) +
                    "_" + owner;
                if (binding.primary_state &&
                    binding.primary_state_floats > 0)
                {
                    sources.push_back({
                        prefix + "_primary",
                        binding.primary_state,
                        static_cast<size_t>(
                            binding.primary_state_floats) *
                            sizeof(float),
                    });
                }
                if (binding.secondary_state &&
                    binding.secondary_state_floats > 0)
                {
                    sources.push_back({
                        prefix + "_secondary",
                        binding.secondary_state,
                        static_cast<size_t>(
                            binding.secondary_state_floats) *
                            sizeof(float),
                    });
                }
                if (binding.request_state_bank &&
                    binding.request_state_bank_floats > 0)
                {
                    sources.push_back({
                        prefix + "_requests",
                        binding.request_state_bank,
                        binding.request_state_bank_floats *
                            sizeof(float),
                    });
                }
            };
            append_binding("conv", gdn_state->conv_device_state);
            append_binding(
                "recurrence",
                gdn_state->recurrence_device_state);
            if (sources.empty() ||
                !logMTPGraphReuseBoundaryDiagnostics(
                    "after_reset_wait",
                    "request_state_reset_gdn",
                    session_epoch_,
                    /*graph_rebuilt=*/false,
                    /*graph_replayed=*/false,
                    *backend,
                    state_.device_id,
                    consumer_stream,
                    sources))
            {
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "request_reset",
            "device_state_ready_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"},
             {"same_stream",
              boolTag(consumer_stream == ready.producer_stream)}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        return true;
    }

    void DeviceGraphOrchestrator::publishGraphBuildDeviceStateReady(
        PendingGraphBuildDeviceStateReadyState &ready,
        void *producer_stream,
        const char *producer_name)
    {
        if (!state_.device_id.is_gpu())
            return;
        if (!ready.event || ready.valid)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Graph-build device-state "
                "publication has invalid lifecycle ownership"
                << " producer="
                << (producer_name && producer_name[0] != '\0'
                        ? producer_name
                        : "unknown")
                << " event=" << static_cast<bool>(ready.event)
                << " already_valid=" << ready.valid);
            std::terminate();
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!producer_stream ||
            !backend ||
            !DeviceEventEdge::at(
                 DeviceTimelinePoint::GraphBuildDeviceStateReady)
                 .from(
                     DeviceTimelineRole::
                         GraphBuildDeviceStatePublication)
                 .publish(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     producer_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to publish graph-build "
                "device-state readiness");
            std::terminate();
        }

        ready.producer_stream = producer_stream;
        ready.valid = true;
        PerfStatsCollector::addCounter(
            "graph",
            "device_state_publication_events",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"producer",
              producer_name && producer_name[0] != '\0'
                  ? producer_name
                  : "unknown"},
             {"ordering", "event_wait"}});
    }

    bool DeviceGraphOrchestrator::waitForPendingGraphBuildDeviceStateReady(
        PendingGraphBuildDeviceStateReadyState &ready,
        void *consumer_stream,
        DeviceTimelineRole consumer_role,
        const char *consumer_name)
    {
        if (!state_.device_id.is_gpu() || !ready.valid)
            return true;
        if (!consumer_stream ||
            !ready.event ||
            !ready.producer_stream ||
            (consumer_role != DeviceTimelineRole::MainForwardGraph &&
             consumer_role != DeviceTimelineRole::MTPSidecarGraph &&
             consumer_role != DeviceTimelineRole::RequestStateReset))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Graph-build device-state dependency "
                "has incomplete or invalid ownership"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown"));
            return false;
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend ||
            !DeviceEventEdge::at(
                 DeviceTimelinePoint::GraphBuildDeviceStateReady)
                 .from(
                     DeviceTimelineRole::
                         GraphBuildDeviceStatePublication)
                 .to(consumer_role)
                 .enqueueWait(
                     *backend,
                     state_.device_id,
                     ready.event.get(),
                     ready.producer_stream,
                     consumer_stream))
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to order graph execution "
                "after graph-build device-state publication"
                << " consumer="
                << (consumer_name && consumer_name[0] != '\0'
                        ? consumer_name
                        : "unknown"));
            return false;
        }

        PerfStatsCollector::addCounter(
            "graph",
            "device_state_publication_waits",
            1.0,
            perfPhaseName(),
            state_.device_id.toString(),
            {{"consumer",
              consumer_name && consumer_name[0] != '\0'
                  ? consumer_name
                  : "unknown"},
             {"same_stream",
              boolTag(consumer_stream == ready.producer_stream)}});
        ready.valid = false;
        ready.producer_stream = nullptr;
        return true;
    }

    bool DeviceGraphOrchestrator::admitRequestInputsOnDevice(
        const int *tokens,
        const int *position_ids,
        std::span<const int> request_real_lengths,
        int total_tokens,
        int request_count,
        int padded_seq_len,
        std::optional<DeviceDecodePositionBinding> decode_position)
    {
        static_assert(
            sizeof(int) == sizeof(int32_t),
            "GPU request admission requires 32-bit host ints");

        if (!state_.device_id.is_gpu() ||
            !tokens ||
            total_tokens <= 0 ||
            request_count <= 0 ||
            padded_seq_len <= 0 ||
            std::int64_t(total_tokens) != std::int64_t(request_count) * padded_seq_len ||
            total_tokens > request_input_row_capacity_ ||
            request_count > request_sequence_lengths_capacity_ ||
            request_real_lengths.size() !=
                static_cast<size_t>(request_count) ||
            !request_batch_geometry_layout_.valid() ||
            request_batch_geometry_host_.size() !=
                request_batch_geometry_layout_.scalarCount() ||
            !request_batch_geometry_dev_ ||
            !request_token_ids_dev_ ||
            !request_position_ids_dev_ ||
            !request_sequence_lengths_dev_ ||
            !request_row_stride_dev_ ||
            !request_input_admission_ready_.event ||
            !request_input_reuse_ready_.event)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-input admission received invalid geometry or ownership"
                      << " device=" << state_.device_id.toString()
                      << " tokens=" << total_tokens
                      << " requests=" << request_count
                      << " padded_seq_len=" << padded_seq_len
                      << " token_capacity=" << request_input_row_capacity_
                      << " request_capacity=" << request_sequence_lengths_capacity_);
            return false;
        }
        // Chunked ordinary diagnostic callers share the same pre-enqueue guard.
        requireActiveNativeTerminalPublicationCapacity(request_count, padded_seq_len);
        IBackend *backend=getBackendFor(state_.device_id);
        std::optional<RequestInputPositionAdmission> position_admission;
        try {
            position_admission=RequestInputPositionAdmission::bind(position_ids,decode_position,backend,
                state_.kv_cache ? state_.kv_cache->deviceSequenceCachedTokenCountPtr(0) : nullptr,
                static_cast<int32_t *>(request_position_ids_dev_),total_tokens,request_count,padded_seq_len);
        } catch(const std::invalid_argument &error) {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-input position authority rejected: "<<error.what());
            return false;
        }
        const char *const admission_phase=decode_position ? "decode" : "prefill";
        if (request_input_admission_ready_.valid)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] A new request attempted to overwrite unconsumed GPU admission state");
            return false;
        }
        std::fill(
            request_batch_geometry_host_.begin(),
            request_batch_geometry_host_.end(),
            int32_t{0});
        for (int request = 0; request < request_count; ++request)
        {
            const int real_tokens =
                request_real_lengths[static_cast<size_t>(request)];
            if (real_tokens <= 0 || real_tokens > padded_seq_len)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Request-input admission received an invalid real row count"
                          << " request=" << request
                          << " real_tokens=" << real_tokens
                          << " padded_seq_len=" << padded_seq_len);
                return false;
            }
            request_batch_geometry_host_[static_cast<size_t>(request)] =
                static_cast<int32_t>(real_tokens);
        }
        request_batch_geometry_host_[
            request_batch_geometry_layout_.rowStrideIndex()] =
                static_cast<int32_t>(padded_seq_len);

        void *stream =
            explicitGPUStreamForOperation("admitRequestInputsOnDevice");
        if (!stream || !backend)
            return false;
        if (!waitForPendingRequestStateReset(
                stream,
                DeviceTimelineRole::RequestAdmissionTransfer,
                "request_input_admission"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-input admission could not consume request-state reset");
            return false;
        }

        auto &reuse = request_input_reuse_ready_;
        if (reuse.consumers_started)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Request-input admission reached an input bank whose prior readers never published reuse readiness");
            std::terminate();
        }
        if (reuse.valid)
        {
            if (!reuse.event || !reuse.producer_stream ||
                !DeviceEventEdge::at(
                     DeviceTimelinePoint::RequestInputReuseReady)
                     .from(DeviceTimelineRole::MainForwardGraph)
                     .to(DeviceTimelineRole::RequestAdmissionTransfer)
                     .enqueueWait(
                         *backend,
                         state_.device_id,
                         reuse.event.get(),
                         reuse.producer_stream,
                         stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to order request-input admission after prior consumers");
                std::terminate();
            }

            PerfStatsCollector::addCounter(
                "request_admission",
                "device_input_reuse_waits",
                1.0,
                admission_phase,
                state_.device_id.toString(),
                {{"same_stream", boolTag(reuse.producer_stream == stream)},
                 {"ordering", "consumer_release_event"}});
            reuse.valid = false;
            reuse.producer_stream = nullptr;
        }

        const size_t row_bytes =
            sizeof(int32_t) * static_cast<size_t>(total_tokens);
        const size_t unused_request_rows =
            static_cast<size_t>(request_input_row_capacity_ - total_tokens);
        const size_t padding_tokens = position_admission->allowsBucketPadding() ? std::min(
            unused_request_rows,request_padding_token_ids_host_.size()) : 0;
        const size_t padding_bytes =
            sizeof(int32_t) * padding_tokens;
        const size_t geometry_bytes =
            sizeof(int32_t) *
            request_batch_geometry_layout_.scalarCount();

        /*
         * The forward engine may widen this real row to a larger captured graph
         * bucket after admission. Materialize one maximum activation-width tail
         * now so a monolithic bucket can never expose allocator residue. The
         * complete long-request bank does not need padding through max context:
         * its captured chunk materializer writes every visible output row and
         * pads the final window itself. The immutable source vector was
         * allocated during initialization and remains alive until the exact
         * request-reuse event retires.
         */
        const auto enqueue_admission_member =
            [&](const char *member,
                void *destination,
                const void *source,
                size_t bytes)
        {
            if (backend->hostToDeviceOnStream(
                    destination,
                    source,
                    bytes,
                    state_.device_id.gpu_ordinal(),
                    stream))
            {
                return true;
            }
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Failed to enqueue GPU request-input member"
                << " member=" << member
                << " bytes=" << bytes
                << " destination=" << destination
                << " source=" << source
                << " stream=" << stream);
            return false;
        };
        if ((padding_tokens > 0 &&
             !enqueue_admission_member(
                 "token_padding",
                 static_cast<int32_t *>(request_token_ids_dev_) + total_tokens,
                 request_padding_token_ids_host_.data(),
                 padding_bytes)) ||
            !enqueue_admission_member(
                "tokens",
                request_token_ids_dev_,
                tokens,
                row_bytes) ||
            (position_admission->uploadsPositions() && !enqueue_admission_member(
                "positions",request_position_ids_dev_,position_ids,row_bytes)) ||
            !enqueue_admission_member(
                "batch_geometry",
                request_batch_geometry_dev_,
                request_batch_geometry_host_.data(),
                geometry_bytes))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to enqueue the complete GPU request-input admission");
            return false;
        }
        /*
         * The raw transfer backend owns bytes, while BufferArena owns graph
         * coherence. Publish only the members admission actually authored on
         * the exact transfer stream before exposing its readiness event. A
         * scalar decode root owns its position output and publishes it later. A
         * consumer may never infer authority merely from a non-null GPU pointer.
         */
        if (!publishPreparedArenaGraphInput(
                BufferId::REQUEST_TOKEN_IDS,
                request_token_ids_dev_,
                stream,
                state_.device_id,
                "request_input_admission") ||
            (position_admission->uploadsPositions() && !publishPreparedArenaGraphInput(
                BufferId::REQUEST_POSITION_IDS,request_position_ids_dev_,stream,state_.device_id,
                "request_input_admission")) ||
            !publishPreparedArenaGraphInput(
                BufferId::REQUEST_BATCH_GEOMETRY,
                request_batch_geometry_dev_,
                stream,
                state_.device_id,
                "request_input_admission"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU request-input admission could not publish its complete arena transaction");
            return false;
        }
        if (!publishMTPRequestStopTokensOnDevice(
                stream,
                "request_input_admission"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish device-owned MTP stop controls during request admission");
            return false;
        }
        if (!publishMTPRequestPenaltyPolicyOnDevice(
                stream,
                "request_input_admission"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish device-owned MTP penalty controls during request admission");
            return false;
        }
        if (!DeviceEventEdge::at(DeviceTimelinePoint::RequestInputAdmission)
                 .from(DeviceTimelineRole::RequestAdmissionTransfer)
                 .publish(
                     *backend,
                     state_.device_id,
                     request_input_admission_ready_.event.get(),
                     stream))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Failed to publish GPU request-input admission readiness");
            std::terminate();
        }

        request_input_admission_ready_.producer_stream = stream;
        request_input_admission_ready_.valid = true;
        request_input_admission_ready_.token_count = total_tokens;
        request_input_admission_ready_.request_count = request_count;
        request_sequence_lengths_active_count_ = request_count;
        for(const auto &[name,bytes]:std::array<std::pair<const char *,size_t>,3>{{
                {"device_input_token_bytes",row_bytes},
                {"device_input_position_bytes",position_admission->uploadsPositions()?row_bytes:0},
                {"device_input_padding_bytes",padding_bytes}}})
            PerfStatsCollector::addCounter("request_admission",name,static_cast<double>(bytes),
                admission_phase,state_.device_id.toString(),{{"owner","arena_request_inputs"}});
        PerfStatsCollector::addCounter(
            "request_admission",
            "device_owned_input_rows",
            static_cast<double>(total_tokens),
            admission_phase,
            state_.device_id.toString(),
            {{"requests", std::to_string(request_count)},
             {"owner", "arena_request_inputs"},
             {"ordering", "event_wait"}});
        if (request_count > 1 &&
            graph_builder_ &&
            graph_builder_->config().mtp.enabled)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "request_batch_prefill_device_position_admissions",
                static_cast<double>(request_count),
                admission_phase,
                state_.device_id.toString(),
                {{"owner", "request_length_admission"},
                 {"ordering", "event_wait"}});
        }
        return true;
    }

    bool DeviceGraphOrchestrator::publishMTPRequestStopTokensOnDevice(
        void *producer_stream,
        const char *producer)
    {
        using namespace sampling_math;

        if (!state_.device_id.is_gpu())
        {
            return true;
        }
        if (mtp_request_stop_tokens_published_session_epoch_ &&
            *mtp_request_stop_tokens_published_session_epoch_ ==
                session_epoch_)
        {
            return true;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        constexpr size_t kControlBytes =
            sizeof(int32_t) * kSpeculativeBatchMaxStopTokens;
        if (!producer_stream ||
            !backend ||
            !mtp_verifier_stop_tokens_dev_ ||
            !backend->hostToDeviceOnStream(
                mtp_verifier_stop_tokens_dev_,
                mtp_request_stop_tokens_.data(),
                kControlBytes,
                state_.device_id.gpu_ordinal(),
                producer_stream) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_VERIFIER_STOP_TOKENS,
                mtp_verifier_stop_tokens_dev_,
                producer_stream,
                state_.device_id,
                producer))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not publish request-owned MTP stop controls"
                      << " producer="
                      << (producer && producer[0] != '\0'
                              ? producer
                              : "unknown")
                      << " stream=" << producer_stream
                      << " buffer=" << mtp_verifier_stop_tokens_dev_);
            return false;
        }

        mtp_request_stop_tokens_published_session_epoch_ = session_epoch_;
        PerfStatsCollector::addCounter(
            "mtp",
            "request_stop_token_control_publications",
            1.0,
            "request_admission",
            state_.device_id.toString(),
            {{"bytes", std::to_string(kControlBytes)},
             {"count", std::to_string(mtp_request_stop_token_count_)},
             {"producer",
              producer && producer[0] != '\0'
                  ? producer
                  : "unknown"},
             {"ordering", "event_wait"}});
        return true;
    }

    bool DeviceGraphOrchestrator::publishMTPRequestPenaltyPolicyOnDevice(
        void *producer_stream,
        const char *producer)
    {
        if (!state_.device_id.is_gpu())
        {
            return true;
        }
        if (mtp_request_penalty_policy_published_session_epoch_ &&
            *mtp_request_penalty_policy_published_session_epoch_ ==
                session_epoch_)
        {
            return true;
        }

        IBackend *const backend = getBackendFor(state_.device_id);
        if (!producer_stream || !backend ||
            !mtp_greedy_penalty_policy_dev_ ||
            !backend->enqueueConfigureMTPGreedyPenaltyPolicyDevice(
                mtp_greedy_penalty_policy_dev_,
                mtp_request_penalty_policy_.presence_penalty,
                mtp_request_penalty_policy_.frequency_penalty,
                /*first_token_already_in_history=*/false,
                state_.device_id.gpu_ordinal(),
                producer_stream, mtp_request_penalty_policy_.repetition_penalty) ||
            !publishPreparedArenaGraphInput(
                BufferId::MTP_GREEDY_PENALTY_POLICY,
                mtp_greedy_penalty_policy_dev_,
                producer_stream,
                state_.device_id,
                producer))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Could not publish request-owned MTP penalty controls"
                      << " producer="
                      << (producer && producer[0] != '\0'
                              ? producer
                              : "unknown")
                      << " stream=" << producer_stream
                      << " buffer=" << mtp_greedy_penalty_policy_dev_);
            return false;
        }

        mtp_published_request_penalty_policy_ =
            mtp_request_penalty_policy_;
        mtp_request_penalty_policy_published_session_epoch_ = session_epoch_;
        PerfStatsCollector::addCounter(
            "mtp",
            "request_penalty_policy_publications",
            1.0,
            "request_admission",
            state_.device_id.toString(),
            {{"presence_penalty",
              std::to_string(
                  mtp_published_request_penalty_policy_.presence_penalty)},
             {"frequency_penalty",
              std::to_string(
                  mtp_published_request_penalty_policy_.frequency_penalty)},
             {"repetition_penalty",
              std::to_string(mtp_published_request_penalty_policy_.repetition_penalty)},
             {"enabled",
              mtp_published_request_penalty_policy_.enabled()
                  ? "true"
                  : "false"},
             {"producer",
              producer && producer[0] != '\0'
                  ? producer
                  : "unknown"},
             {"ordering", "event_wait"}});
        return true;
    }
} // namespace llaminar2
