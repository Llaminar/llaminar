/**
 * @file DeviceGraphOrchestratorPrefixRestore.cpp
 * @brief DeviceGraphOrchestrator method shard (46668..51041).
 */
#include "DeviceGraphOrchestratorInternal.h"

namespace llaminar2
{

    PrefixDeviceHotLeaseResult
    DeviceGraphOrchestrator::promotePrefixBlockToDeviceHotForRestore(
        const PrefixBlockHandle &lower_tier_handle,
        void *producer_stream,
        PrefixBlockHandle *promoted_handle)
    {
        auto fail = [&](const std::string &reason,
                        PrefixBlockHandle *prepared = nullptr)
            -> PrefixDeviceHotLeaseResult
        {
            if (prepared && prepared->valid() && prefix_device_hot_backend_)
            {
                /*
                 * release() retires backend budget accounting. The handle's
                 * shared allocation owners remain alive until their queued
                 * readiness event is complete, so exceptional cleanup cannot
                 * free storage still referenced by an asynchronous transfer.
                 */
                (void)prefix_device_hot_backend_->release(*prepared);
            }
            LOG_ERROR("[DeviceGraphOrchestrator] Device-hot prefix promotion failed: "
                      << reason
                      << " key=" << lower_tier_handle.key.toHex()
                      << " bytes=" << lower_tier_handle.total_bytes);
            return PrefixDeviceHotLeaseResult::Error;
        };

        if (!promoted_handle ||
            !state_.device_id.is_gpu() ||
            !producer_stream ||
            !prefix_cache_ ||
            !prefix_device_hot_backend_)
        {
            return fail("promotion dependencies are unavailable");
        }
        if (!lower_tier_handle.valid() ||
            lower_tier_handle.tier != PrefixStorageTier::Ram ||
            !prefix_cache_->deviceHotCapacityEligible(
                lower_tier_handle.total_bytes))
        {
            return fail("lower-tier handle is not eligible for the hot tier");
        }
        if (!lower_tier_handle.waitForPayloadOnStream(producer_stream))
        {
            return fail("lower-tier payload readiness wait failed");
        }

        PrefixBlockHandle prepared;
        const PrefixDeviceHotLeaseResult lease_result =
            prefix_cache_->prepareDeviceHotCopy(
                lower_tier_handle, producer_stream, &prepared);
        if (lease_result == PrefixDeviceHotLeaseResult::Busy)
            return lease_result;
        if (lease_result != PrefixDeviceHotLeaseResult::Acquired)
        {
            return fail(
                "hot-tier arena rejected an eligible promotion lease");
        }

        IBackend *backend = getBackendFor(state_.device_id);
        if (!backend)
        {
            return fail("GPU backend is unavailable", &prepared);
        }
        PrefixArchivePublicationScope promotion_completion([&](PrefixArchiveCompletion) {
            if (!prepared.payload_readiness || !prepared.payload_readiness->prepared())
                return true;
            const bool result = recordPrefixArchivePayloadReady(
                &prepared, producer_stream, "restore_prefix_device_hot_promotion");
            // Publication precedes LRU installation, which may itself fail.
            // Keep both DMA endpoints through the same existing event on every exit.
            PendingPrefixPayloadUse pending;
            pending.completion_event = std::shared_ptr<void>(
                prepared.payload_readiness, prepared.payload_readiness->event());
            pending.retained_payload_sources.push_back(
                PrefixPayloadReadLease::wholeArchive(lower_tier_handle));
            pending.retained_payload_sources.push_back(
                PrefixPayloadReadLease::wholeArchive(prepared));
            pending_prefix_payload_uses_.push_back(std::move(pending));
            return result;
        });
        if (!preparePrefixArchivePayloadReady(
                &prepared,
                producer_stream,
                "restore_prefix_device_hot_promotion"))
        {
            return fail(
                "promotion readiness event preflight failed",
                &prepared);
        }
        const int ordinal = state_.device_id.gpu_ordinal();
        const auto upload_section =
            [&](const char *section,
                void *destination,
                const void *source,
                size_t bytes) -> bool
        {
            if (bytes == 0)
                return true;
            if (!destination || !source)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-hot promotion section "
                          << section << " has a null "
                          << (!destination ? "destination" : "source"));
                return false;
            }
            if (!backend->hostToDeviceOnStream(
                    destination,
                    source,
                    bytes,
                    ordinal,
                    producer_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Device-hot promotion H2D failed "
                          "for section=" << section
                          << " bytes=" << bytes);
                return false;
            }
            return true;
        };

        /*
         * The RAM archive stores K followed by V in one allocation, and shifted
         * MTP K followed by V in another. Uploading each contiguous allocation
         * once avoids the old per-layer staging loop and leaves the promoted
         * handle immediately usable by every logical import operation.
         */
        bool uploaded =
            upload_section(
                "kv",
                prepared.deviceKVKData(),
                lower_tier_handle.kv_payload,
                lower_tier_handle.kvBytes()) &&
            upload_section(
                "mtp",
                prepared.deviceMTPKData(),
                lower_tier_handle.mtp_payload,
                lower_tier_handle.layout.mtpKVBytes());

        if(uploaded && lower_tier_handle.layout.modelAuxiliaryBytes())
            uploaded=upload_section("model_auxiliary",prepared.deviceModelAuxiliaryData(),
                lower_tier_handle.model_auxiliary_payload,lower_tier_handle.layout.modelAuxiliaryBytes());

        if (uploaded && lower_tier_handle.has_hybrid_state)
        {
            uploaded = upload_section(
                "hybrid",
                prepared.deviceHybridData(),
                lower_tier_handle.hybrid_payload,
                lower_tier_handle.hybridBytes());
        }
        if (uploaded && lower_tier_handle.has_terminal_hidden)
        {
            uploaded = upload_section(
                "terminal_hidden",
                prepared.deviceTerminalHiddenData(),
                lower_tier_handle.terminal_hidden,
                lower_tier_handle.terminalHiddenBytes());
        }
        if (uploaded && lower_tier_handle.has_terminal_logits)
        {
            uploaded = upload_section(
                "terminal_logits",
                prepared.deviceTerminalLogitsData(),
                lower_tier_handle.terminal_logits,
                lower_tier_handle.terminalLogitsBytes());
        }
        if (!uploaded)
        {
            return fail("one or more archive sections could not be uploaded",
                        &prepared);
        }

        promotion_completion.commit();
        if (!prefix_cache_->installDeviceHotCopy(
                prepared,
                /*repromotion=*/true))
        {
            return fail("completed replica could not enter the hot LRU",
                        &prepared);
        }

        PerfStatsCollector::addCounterWithSequence(
            "prefix_cache",
            "device_hot_repromotions",
            1.0,
            {static_cast<uint64_t>(lower_tier_handle.key.block_index)},
            "promotion",
            state_.device_id.toString(),
            {{"tokens", std::to_string(lower_tier_handle.key.token_count)},
             {"bytes", std::to_string(lower_tier_handle.total_bytes)},
             {"source", "ram_or_disk_hydration"}});
        *promoted_handle = std::move(prepared);
        return PrefixDeviceHotLeaseResult::Acquired;
    }

    bool DeviceGraphOrchestrator::publishPrefixFingerprintTransition(
        const PrefixCacheFingerprintResult &next_identity,
        PrefixCacheMoEPolicy policy,
        const char *operation)
    {
        const uint64_t next_fingerprint = next_identity.key;
        if (next_identity.bypass || next_fingerprint == 0 || prefix_identity_.key == 0 ||
            !prefix_cache_)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Prefix fingerprint transition has "
                "incomplete old/new identity or cache ownership"
                << " operation=" << (operation ? operation : "<unknown>")
                << " previous=" << prefixHashHex(prefix_identity_.key)
                << " next=" << prefixHashHex(next_fingerprint)
                << " cache=" << (prefix_cache_ ? "ready" : "missing"));
            return false;
        }
        if (next_fingerprint == prefix_identity_.key)
        {
            // A placement-compatible policy can retain the same key while
            // still publishing the newly observed admission epoch.
            prefix_identity_ = next_identity;
            return true;
        }

        const uint64_t previous_fingerprint = prefix_identity_.key;
        std::optional<PrefixFingerprintRebase> rebase;
        if (policy == PrefixCacheMoEPolicy::InvalidateOnRebalance)
        {
            rebase = prefix_cache_->rebaseFingerprint(
                previous_fingerprint,
                next_fingerprint);
            if (!rebase)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Prefix fingerprint rebase was "
                    "rejected by a busy or invalid cache lifecycle"
                    << " operation=" << (operation ? operation : "<unknown>")
                    << " previous=" << prefixHashHex(previous_fingerprint)
                    << " next=" << prefixHashHex(next_fingerprint));
                return false;
            }

            PerfStatsCollector::addCounter(
                "prefix_cache",
                "fingerprint_rebases",
                1.0,
                "request_boundary",
                state_.device_id.toString(),
                {
                    {"operation", operation ? operation : "unknown"},
                    {"ram_entries", std::to_string(rebase->invalidated_ram_entries)},
                    {"device_entries", std::to_string(rebase->invalidated_device_entries)},
                    {"disk_entries", std::to_string(rebase->unindexed_disk_entries)},
                    {"ram_bytes", std::to_string(rebase->released_ram_bytes)},
                    {"device_bytes", std::to_string(rebase->released_device_bytes)},
                });
        }

        prefix_identity_ = next_identity;
        LOG_INFO(
            "[DeviceGraphOrchestrator] Prefix cache fingerprint published"
            << " operation=" << (operation ? operation : "<unknown>")
            << " policy=" << prefixCacheMoEPolicyToString(policy)
            << " previous=" << prefixHashHex(previous_fingerprint)
            << " next=" << prefixHashHex(next_fingerprint)
            << " volatile_rebase=" << (rebase ? "yes" : "no")
            << " invalidated_ram_entries="
            << (rebase ? rebase->invalidated_ram_entries : 0u)
            << " invalidated_device_entries="
            << (rebase ? rebase->invalidated_device_entries : 0u)
            << " unindexed_disk_entries="
            << (rebase ? rebase->unindexed_disk_entries : 0u));
        return true;
    }

    bool DeviceGraphOrchestrator::ensurePrefixCacheReady()
    {
        if (!graph_builder_)
        {
            disablePrefixCacheForRunner("graph builder unavailable");
            return false;
        }

        const auto &config = graph_builder_->config();
        const auto &prefix_config = config.prefix_cache;
        if (prefix_cache_bypassed_)
        {
            return false;
        }
        if (prefix_cache_)
        {
            /*
             * The compatibility material is model-lifetime state except for
             * the authoritative MoE movement epoch. Rebuilding it for every
             * request allocates and sorts tens of thousands of placement
             * strings for large MoE models. The published identity already
             * binds its key to the exact epoch observation, so an unchanged
             * epoch proves that the existing value is current. Movement (or a
             * live hybrid-layout refresh, which publishes explicitly below)
             * still takes the full rebuild-and-transition path.
             */
            const uint64_t live_movement_epoch = moeRuntimeMovementEpoch();
            if (prefix_identity_.key != 0 &&
                !prefix_identity_.bypass &&
                prefix_identity_.placement_epoch == live_movement_epoch)
            {
                return true;
            }
            auto fingerprint = buildCurrentPrefixFingerprint(prefix_config);
            if (fingerprint.bypass || fingerprint.key == 0)
            {
                disablePrefixCacheForRunner(
                    fingerprint.bypass_reason.empty() ? "fingerprint refresh requested bypass"
                                                      : fingerprint.bypass_reason);
                return false;
            }
            return publishPrefixFingerprintTransition(
                fingerprint,
                prefix_config.moe_policy,
                "ensure_prefix_cache_ready");
        }
        if (!prefix_config.enabled ||
            prefix_config.storage_mode == PrefixCacheStorageMode::Disabled)
        {
            disablePrefixCacheForRunner("feature disabled");
            return false;
        }
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Device)
        {
            disablePrefixCacheForRunner("device-only prefix cache tier is not implemented yet");
            return false;
        }
        if (!state_.isInitialized() || !state_.kv_cache)
        {
            disablePrefixCacheForRunner("inference state or KV cache unavailable");
            return false;
        }
        const int block_size = prefix_config.block_size > 0 ? prefix_config.block_size : 64;
        const bool terminal_state_enabled =
            prefix_config.terminal_state != PrefixCacheTerminalStateMode::Off;
        const bool owns_terminal_state =
            !pp_stage_config_.has_value() || pp_stage_config_->has_lm_head;
        const bool use_local_terminal_logits = activeMainLogitsAreColumnParallel();
        const auto *terminal_logits_tensor =
            use_local_terminal_logits ? state_.logits_local.get() : state_.logits.get();
        const size_t terminal_hidden_bytes =
            owns_terminal_state && config.mtp.enabled && terminal_state_enabled && state_.d_model > 0
                ? static_cast<size_t>(state_.hidden->shape().back()) * sizeof(float)
                : 0;
        const size_t terminal_logits_bytes =
            !terminal_state_enabled || !owns_terminal_state
                ? 0
                : fp32LogitsRowBytes(terminal_logits_tensor);

        prefix_layout_ = buildDensePrefixPayloadLayout(
            *state_.kv_cache,
            state_.device_id,
            block_size,
            terminal_hidden_bytes,
            terminal_logits_bytes);

        // Use the same participant role as cache construction. Pipeline
        // followers verify/restore their main shard, but own no predictor.
        if (resolveMTPRuntimeTransactionRole(config.mtp.enabled, owns_terminal_state) ==
            MTPRuntimeTransactionRole::PredictorOwner)
        {
            if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
            {
                disablePrefixCacheForRunner("MTP prefix cache requested without an initialized MTP KV cache");
                return false;
            }
            if (!attachMTPPayloadLayout(prefix_layout_, *state_.mtp_kv_caches[0]))
            {
                disablePrefixCacheForRunner("MTP KV cache does not expose a logical payload layout");
                return false;
            }
        }

        applyPrefixAuxiliaryLayout(prefix_layout_,*state_.kv_cache,
            state_.mtp_kv_caches.empty()?nullptr:state_.mtp_kv_caches[0].get(),0,0,block_size,block_size,true);

        if (!prefix_layout_.hasRestorableMainState())
        {
            disablePrefixCacheForRunner("main cache does not expose a complete attention/recurrent payload layout");
            return false;
        }
        if (!ensurePrefixArchiveDeviceStaging())
        {
            disablePrefixCacheForRunner("GPU prefix archive staging allocation failed");
            return false;
        }
        const size_t block_bytes = prefixPayloadCapacityLayout(prefix_layout_, *state_.kv_cache,
            state_.mtp_kv_caches.empty() ? nullptr : state_.mtp_kv_caches[0].get()).totalBytes();
        if (block_bytes == 0 || prefix_config.ram_budget_bytes < block_bytes)
        {
            disablePrefixCacheForRunner("RAM budget cannot hold one complete prefix block");
            return false;
        }

        auto fingerprint = buildCurrentPrefixFingerprint(prefix_config);
        if (fingerprint.bypass || fingerprint.key == 0)
        {
            disablePrefixCacheForRunner(
                fingerprint.bypass_reason.empty() ? "fingerprint build requested bypass"
                                                  : fingerprint.bypass_reason);
            return false;
        }

        prefix_identity_ = fingerprint;
        if (!physical_memory_authority_ &&
            memory_authority_mode_ ==
                MemoryAuthorityMode::ProductionRequired)
        {
            disablePrefixCacheForRunner(
                "RAM prefix tier has no rank-local physical-memory authority");
            return false;
        }
        std::string ram_tier_error;
        prefix_ram_backend_ = physical_memory_authority_
            ? RamPrefixStorageBackend::create(
                  state_.device_id,
                  prefix_config.ram_budget_bytes,
                  physical_memory_authority_,
                  &ram_tier_error)
            : std::make_shared<RamPrefixStorageBackend>(
                  state_.device_id,
                  prefix_config.ram_budget_bytes);
        if (!prefix_ram_backend_)
        {
            disablePrefixCacheForRunner(
                "RAM prefix tier admission failed: " + ram_tier_error);
            return false;
        }
        prefix_device_hot_backend_.reset();
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Tiered &&
            state_.device_id.is_gpu() &&
            prefix_config.device_budget_bytes >= block_bytes)
        {
            if (!physical_memory_authority_)
            {
                disablePrefixCacheForRunner(
                    "device-hot prefix arena has no rank-local physical-memory authority");
                return false;
            }
            std::string device_hot_error;
            prefix_device_hot_backend_ = DeviceHotPrefixStorageBackend::create(
                state_.device_id,
                prefix_config.device_budget_bytes,
                block_bytes,
                physical_memory_authority_,
                &device_hot_error);
            if (!prefix_device_hot_backend_)
            {
                disablePrefixCacheForRunner(
                    "device-hot prefix arena admission failed: " +
                    device_hot_error);
                return false;
            }
        }
        prefix_disk_backend_.reset();
        if (prefix_config.storage_mode == PrefixCacheStorageMode::Tiered &&
            prefix_config.disk_budget_bytes > 0)
        {
            if (prefix_config.disk_dir.empty())
            {
                disablePrefixCacheForRunner(
                    "disk prefix cache requested but HOME/default disk directory is unavailable");
                return false;
            }
            if (!injected_model_ctx_ || injected_model_ctx_->path().empty())
            {
                disablePrefixCacheForRunner(
                    "disk prefix cache requires a file-backed model context");
                return false;
            }

            std::vector<std::filesystem::path> model_artifact_paths;
            for (const auto &path : injected_model_ctx_->artifactPaths())
                model_artifact_paths.emplace_back(path);
            std::string identity_error;
            const auto model_artifact_identity =
                sha256FileSetIdentityHex(
                    model_artifact_paths, &identity_error);
            if (!model_artifact_identity)
            {
                disablePrefixCacheForRunner(
                    "failed to identify the loaded model artifact for disk prefix cache: " +
                    identity_error);
                return false;
            }

            const std::filesystem::path archive_path =
                std::filesystem::path(prefix_config.disk_dir) /
                (*model_artifact_identity + ".kvcache");
            std::string archive_error;
            prefix_disk_backend_ = DiskPrefixStorageBackend::openShared(
                archive_path,
                prefix_config.disk_budget_bytes,
                *model_artifact_identity,
                physical_memory_authority_,
                &archive_error);
            if (!prefix_disk_backend_)
            {
                disablePrefixCacheForRunner(
                    "failed to open disk prefix archive: " + archive_error);
                return false;
            }
        }
        prefix_cache_ = std::make_shared<PrefixStateCache>(
            prefix_config.ram_budget_bytes,
            prefix_ram_backend_,
            prefix_disk_backend_,
            prefix_device_hot_backend_,
            this);
        if (prefix_disk_backend_)
        {
            std::string discovery_error;
            if (!prefix_cache_->installDiscoveredDiskEntries(
                    prefix_identity_.key,
                    prefix_layout_,
                    &discovery_error))
            {
                disablePrefixCacheForRunner(
                    "failed to index disk prefix archive: " +
                    discovery_error);
                return false;
            }
        }
        prefix_telemetry_->bind(prefix_ram_backend_->telemetry(),
            prefix_disk_backend_ ? prefix_disk_backend_->telemetry() : nullptr);
        prefix_cache_bypassed_ = false;
        prefix_cache_bypass_reason_.clear();
        LOG_INFO("[DeviceGraphOrchestrator] Prefix cache enabled: block_size="
                 << prefix_layout_.block_size
                 << " block_bytes=" << block_bytes
                 << " ram_budget_bytes=" << prefix_config.ram_budget_bytes
                 << " device_hot_budget_bytes=" << (prefix_device_hot_backend_ ? prefix_device_hot_backend_->budgetBytes() : 0)
                 << " device_hot_reserved_bytes=" << (prefix_device_hot_backend_ ? prefix_device_hot_backend_->reservedBytes() : 0)
                 << " device_hot_slots=" << (prefix_device_hot_backend_ ? prefix_device_hot_backend_->slotCount() : 0)
                 << " disk_budget_bytes=" << (prefix_disk_backend_ ? prefix_config.disk_budget_bytes : 0)
                 << " disk_archive="
                 << (prefix_disk_backend_
                         ? prefix_disk_backend_->archivePath().string()
                         : ""));
        return true;
    }

    bool DeviceGraphOrchestrator::refreshPrefixPayloadLayoutForLiveHybridState(const char *operation)
    {
        if (!prefix_cache_ || !state_.kv_cache)
        {
            return false;
        }
        if (!dynamic_cast<const IHybridKVCache *>(state_.kv_cache.get()))
        {
            return true;
        }

        const auto &graph_config = graph_builder_->config();
        PrefixPayloadLayout live_layout = buildDensePrefixPayloadLayout(
            *state_.kv_cache,
            state_.device_id,
            prefix_layout_.block_size,
            prefix_layout_.terminal_hidden_bytes,
            prefix_layout_.terminal_logits_bytes);
        copyMTPPayloadLayoutFields(prefix_layout_, live_layout);
        applyPrefixAuxiliaryLayout(live_layout,*state_.kv_cache,
            state_.mtp_kv_caches.empty()?nullptr:state_.mtp_kv_caches[0].get(),0,0,
            live_layout.block_size,live_layout.block_size,true);

        if (samePrefixPayloadShape(live_layout, prefix_layout_))
        {
            return true;
        }
        if (!live_layout.hasRestorableMainState())
        {
            disablePrefixCacheForRunner("live hybrid prefix layout does not expose complete restorable state");
            return false;
        }

        const auto &prefix_config = graph_config.prefix_cache;
        const size_t live_block_bytes = live_layout.totalBytes();
        if (live_block_bytes == 0 || prefix_config.ram_budget_bytes < live_block_bytes)
        {
            disablePrefixCacheForRunner("RAM budget cannot hold one live hybrid prefix block");
            return false;
        }

        const auto stats = prefix_cache_->stats();
        const bool has_cached_entries =
            prefix_cache_->size() > 0 ||
            stats.device_hot_bytes > 0 ||
            stats.disk_bytes > 0;
        if (has_cached_entries && !prefix_cache_->clear())
        {
            disablePrefixCacheForRunner("live hybrid prefix layout changed while entries were retained");
            return false;
        }

        LOG_INFO("[DeviceGraphOrchestrator] Prefix cache live hybrid layout refreshed"
                 << " operation=" << (operation ? operation : "<unknown>")
                 << " old_hybrid_host_bytes=" << prefix_layout_.hybrid_host_state_bytes
                 << " old_hybrid_device_bytes=" << prefix_layout_.hybrid_device_state_bytes
                 << " new_hybrid_host_bytes=" << live_layout.hybrid_host_state_bytes
                 << " new_hybrid_device_bytes=" << live_layout.hybrid_device_state_bytes
                 << " block_bytes=" << live_block_bytes
                 << " invalidated_entries=" << (has_cached_entries ? "yes" : "no"));
        prefix_layout_ = live_layout;
        if (!ensurePrefixArchiveDeviceStaging())
        {
            disablePrefixCacheForRunner("live hybrid GPU prefix archive staging resize failed");
            return false;
        }
        auto fingerprint = buildCurrentPrefixFingerprint(prefix_config);
        if (fingerprint.bypass || fingerprint.key == 0)
        {
            disablePrefixCacheForRunner(
                fingerprint.bypass_reason.empty() ? "live hybrid fingerprint refresh requested bypass"
                                                  : fingerprint.bypass_reason);
            return false;
        }
        return publishPrefixFingerprintTransition(
            fingerprint,
            prefix_config.moe_policy,
            "refresh_live_hybrid_layout");
    }

    PrefixCacheKey DeviceGraphOrchestrator::makePrefixKeyForBlock(
        uint64_t request_fingerprint,
        const std::vector<int32_t> &tokens,
        int block_index,
        uint64_t parent_hash) const
    {
        const int start = block_index * prefix_layout_.block_size;
        const int end = std::min<int>(start + prefix_layout_.block_size, tokens.size());
        std::vector<int32_t> block_tokens(tokens.begin() + start, tokens.begin() + end);
        return makePrefixCacheKey(
            request_fingerprint,
            parent_hash,
            block_index,
            start,
            block_tokens);
    }

    PrefixLookupResult DeviceGraphOrchestrator::lookupPrefix(const std::vector<int32_t> &tokens)
    {
        PrefixLookupResult result;
        result.cache_enabled = graph_builder_ && graph_builder_->config().prefix_cache.enabled;

        if (!ensurePrefixCacheReady())
        {
            result.bypass_reason = prefix_cache_bypass_reason_;
            return result;
        }

        result.supported = true;
        result.block_size = prefix_layout_.block_size;
        result.fingerprint_key = prefix_identity_.key;
        result.checkpoint_policy = prefix_layout_.includes_hybrid_state
            ? PrefixCheckpointPolicy::ReusableBoundary
            : PrefixCheckpointPolicy::TerminalOnly;
        result.placement_epochs = PrefixPlacementEpochSpan::at(
            prefix_identity_.placement_epoch);
        result.requires_terminal_logits = prefix_layout_.includes_terminal_logits;
        result.requires_terminal_hidden = prefix_layout_.includes_terminal_hidden;
        if (tokens.empty() || prefix_layout_.block_size <= 0)
        {
            return result;
        }

        const int complete_blocks =
            (static_cast<int>(tokens.size()) + prefix_layout_.block_size - 1) /
            prefix_layout_.block_size;
        const auto organization = prefix_layout_.organization();
        result.payload_plan = prefix_cache_->beginLookup();
        uint64_t parent_hash = 0;
        for (int block = 0; block < complete_blocks; ++block)
        {
            const PrefixCacheKey requested_key =
                makePrefixKeyForBlock(
                    result.fingerprint_key,
                    tokens,
                    block,
                    parent_hash);
            const int block_start = block * prefix_layout_.block_size;
            const int block_end = std::min<int>(
                block_start + prefix_layout_.block_size,
                tokens.size());
            const std::vector<int32_t> requested_block_tokens(
                tokens.begin() + block_start,
                tokens.begin() + block_end);
            auto handle = result.payload_plan->selectLongest(
                result.fingerprint_key,
                parent_hash,
                block,
                block_start,
                requested_block_tokens);
            const PrefixCacheKey key = handle ? handle->key : requested_key;
            if (!handle || !handle->layout.compatiblePayloadShape(prefix_layout_))
            {
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "block_misses",
                    1.0,
                    {static_cast<uint64_t>(block), static_cast<uint64_t>(tokens.size())},
                    "lookup",
                    state_.device_id.toString(),
                    {{"handle_present", handle ? "true" : "false"},
                     {"shape_compatible", (handle && handle->layout.compatiblePayloadShape(prefix_layout_)) ? "true" : "false"}});
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] lookup miss device=" << state_.device_id.toString()
                                                                  << " block=" << block
                                                                  << " key=" << key.toHex()
                                                                  << " fingerprint="
                                                                  << prefixHashHex(result.fingerprint_key)
                                                                  << " handle="
                                                                  << (handle ? "present" : "absent")
                                                                  << " shape_compatible="
                                                                  << (handle && handle->layout.compatiblePayloadShape(prefix_layout_) ? "yes" : "no"));
                }
                if (organization == PrefixPayloadOrganization::RecurrentCheckpoint)
                {
                    // The prompt itself authenticates every ancestor hash.
                    // No K/V payload is needed before a complete recurrent
                    // checkpoint, so a missing earlier record is not a gap.
                    parent_hash = requested_key.stableHash();
                    continue;
                }
                break;
            }

            const bool terminal_prefix_match =
                handle->key.token_count < requested_key.token_count;

            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache",
                "block_hits",
                1.0,
                {static_cast<uint64_t>(block), static_cast<uint64_t>(tokens.size())},
                "lookup",
                state_.device_id.toString(),
                {{"token_count", std::to_string(handle->key.token_count)},
                 {"terminal_logits", handle->has_terminal_logits ? "true" : "false"},
                 {"terminal_hidden", handle->has_terminal_hidden ? "true" : "false"},
                 {"includes_mtp_state", handle->layout.includes_mtp_state ? "true" : "false"},
                 {"model_runtime_state", handle->has_model_runtime_state ? "true" : "false"},
                 {"match_kind", terminal_prefix_match ? "terminal_prefix" : "exact_block"}});
            if (terminal_prefix_match)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "terminal_partial_block_prefix_hits",
                    1.0,
                    {static_cast<uint64_t>(block), static_cast<uint64_t>(handle->key.token_start + handle->key.token_count)},
                    "lookup",
                    state_.device_id.toString(),
                    {{"cached_block_tokens",
                      std::to_string(handle->key.token_count)},
                     {"requested_block_tokens",
                      std::to_string(requested_key.token_count)},
                     {"includes_hybrid_state",
                      handle->has_hybrid_state ? "true" : "false"}});
            }
            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] lookup hit device=" << state_.device_id.toString()
                                                             << " block=" << block
                                                             << " key=" << key.toHex()
                                                             << " token_count=" << handle->key.token_count
                                                             << " terminal_logits="
                                                             << (handle->has_terminal_logits ? "yes" : "no")
                                                             << " terminal_hidden="
                                                             << (handle->has_terminal_hidden ? "yes" : "no")
                                                             << " mtp_state="
                                                             << (handle->layout.includes_mtp_state ? "yes" : "no")
                                                             << " model_runtime_state="
                                                             << (handle->has_model_runtime_state ? "yes" : "no")
                                                             << " model_runtime_state_bytes="
                                                             << (handle->model_runtime_state_storage
                                                                     ? handle->model_runtime_state_storage->size()
                                                                     : 0u));
            }
            result.blocks.push_back(*handle);
            result.cached_tokens = handle->key.token_start + handle->key.token_count;
            result.has_terminal_hidden = handle->has_terminal_hidden;
            result.has_terminal_logits = handle->has_terminal_logits;
            parent_hash = organization == PrefixPayloadOrganization::RecurrentCheckpoint
                ? requested_key.stableHash() : key.stableHash();
            if (organization == PrefixPayloadOrganization::RecurrentCheckpoint)
            {
                // Retain complete candidate frontiers for coordinated clamp.
                // A longer matching checkpoint may exist past this partial
                // hash chunk; only attention requires a contiguous chain.
                continue;
            }
            if (terminal_prefix_match)
            {
                /*
                 * Tokens after the matched terminal prefix belong to suffix
                 * prefill. They cannot contribute to a child block key until
                 * the restored recurrent state has consumed them.
                 */
                break;
            }
        }

        result = result.clampedTo(result.payload_plan->boundedTokenCount(result.blocks));
        while (!result.blocks.empty() &&
               prefix_layout_.includes_hybrid_state &&
               !result.blocks.back().has_hybrid_state)
        {
            result.blocks.pop_back();
            result.cached_tokens = result.blocks.empty()
                                       ? 0
                                       : result.blocks.back().key.token_start +
                                             result.blocks.back().key.token_count;
            result.has_terminal_hidden =
                !result.blocks.empty() && result.blocks.back().has_terminal_hidden;
            result.has_terminal_logits =
                !result.blocks.empty() && result.blocks.back().has_terminal_logits;
        }

        prefix_cache_->recordRequestLookup(
            static_cast<int>(tokens.size()),
            result.cached_tokens,
            static_cast<int>(result.blocks.size()));
        PerfStatsCollector::addCounterWithSequence(
            "prefix_cache",
            "lookup_results",
            1.0,
            {static_cast<uint64_t>(tokens.size()), static_cast<uint64_t>(result.cached_tokens), static_cast<uint64_t>(result.blocks.size())},
            "lookup",
            state_.device_id.toString(),
            {{"hit_type", result.cached_tokens <= 0
                                 ? "miss"
                                 : (result.cached_tokens >= static_cast<int>(tokens.size()) ? "full" : "partial")},
             {"terminal_logits", result.has_terminal_logits ? "true" : "false"},
             {"terminal_hidden", result.has_terminal_hidden ? "true" : "false"}});

        if (prefixCacheTraceEnabled())
        {
            LOG_INFO("[PREFIX_TRACE] lookup result device=" << state_.device_id.toString()
                                                            << " cached_tokens=" << result.cached_tokens
                                                            << " blocks=" << result.blocks.size()
                                                            << " terminal_logits="
                                                            << (result.has_terminal_logits ? "yes" : "no")
                                                            << " terminal_hidden="
                                                            << (result.has_terminal_hidden ? "yes" : "no")
                                                            << " fingerprint="
                                                            << prefixHashHex(result.fingerprint_key));
        }

        return result;
    }

    bool DeviceGraphOrchestrator::populatePrefix(const PrefixLookupResult &hit, int seq_idx)
    {
        (void)hit.restoreBlocks();
        auto fail = [](const std::string &reason) -> bool
        {
            LOG_ERROR("[DeviceGraphOrchestrator] populatePrefix failed: " << reason);
            return false;
        };

        if (hit.cached_tokens <= 0)
        {
            return true;
        }
        if (!ensurePrefixCacheReady() || !state_.kv_cache)
        {
            return fail("prefix cache or KV cache is not ready");
        }
        if (seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return fail("sequence index out of range");
        }
        if (hit.blocks.empty())
        {
            return fail("prefix hit has no blocks");
        }

        // Coordination has chosen the one restore endpoint. Historical blocks
        // now need only sequence rows; acquire no other recurrent snapshots.
        std::vector<PrefixBlockHandle> restore_blocks = hit.materializeRestoreBlocks();

        void *stream = explicitGPUStreamForOperation("populatePrefix");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing explicit GPU stream");
        }
        if (!joinPublishedLiveStateHandoffs(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "populate_prefix_previous_mutation"))
        {
            return fail("pending live prefix mutation wait failed");
        }
        if (!waitForPendingLiveGraphProducersBeforePrefixMutation(
                stream,
                "populate_prefix"))
        {
            return fail("pending live graph producer wait failed");
        }

        const auto terminal_block = restore_blocks.back();
        const bool restore_model_runtime_state =
            hit.restore_model_runtime_state &&
            terminal_block.has_model_runtime_state &&
            terminal_block.model_runtime_state_storage &&
            !terminal_block.model_runtime_state_storage->empty();
        /*
         * A prefix restore is not an overlay on top of the current request; it is
         * a complete replacement of live sequence state.  Decode may have warmed
         * graph replay, kernel dynamic buffers, token mailboxes, sampled-logit
         * streams, or model-local scratch that remains valid for ordinary
         * request resets but is stale for a restored prefix followed by suffix
         * prefill.  Reset inference-owned state after ordering earlier
         * producers, then import the snapshot into a clean live-state
         * transaction.  If a model-runtime snapshot is present, the typed
         * reset deliberately leaves the graph-owned model-runtime owner for
         * the restore call below; otherwise it resets that owner here so a
         * missing snapshot cannot silently reuse stale MoE placement state.
         */
        try
        {
            resetInferenceState(
                InferenceStateResetRequest::prefixRestoreBoundary(
                    "populate_prefix_cache",
                    /*reset_model_runtime_owner=*/!restore_model_runtime_state));
        }
        catch (const std::exception &ex)
        {
            return fail(std::string("prefix restore state reset failed: ") + ex.what());
        }
        stream = explicitGPUStreamForOperation("populatePrefix");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing explicit GPU stream after prefix restore reset");
        }
        if (!waitForPendingRequestStateReset(
                stream,
                DeviceTimelineRole::PrefixRestoreMutation,
                "populate_prefix_request_state_reset"))
        {
            return fail("prefix restore could not consume request-state reset");
        }
        if (!publishMTPRequestStopTokensOnDevice(
                stream,
                "prefix_restore"))
        {
            return fail(
                "prefix restore could not publish device-owned MTP request "
                "stop controls");
        }
        if (!publishMTPRequestPenaltyPolicyOnDevice(
                stream,
                "prefix_restore"))
        {
            return fail(
                "prefix restore could not publish device-owned MTP request "
                "penalty controls");
        }
        IBackend *prefix_backend =
            state_.device_id.is_gpu() ? getBackendFor(state_.device_id) : nullptr;
        if (state_.device_id.is_gpu() &&
            (!prefix_backend || !prefix_archive_device_staging_))
        {
            return fail("GPU prefix archive staging is unavailable");
        }
        if (!prepareLivePrefixMutationReadyEvent(
                stream,
                "populate_prefix_cache"))
        {
            return fail("could not preflight live-state readiness event");
        }

        /*
         * A lookup result owns the tier that was resident when lookup ran. Cold
         * GPU hits are promoted before live-state import when a preallocated
         * slot is reusable. Blocks larger than the tier, and eligible blocks
         * contending with a hot handle retained by this same immutable lookup,
         * restore directly from pinned RAM. Busy is therefore a declared tier
         * choice; event/copy/arena lifecycle errors remain fatal.
         */
        std::vector<PrefixPayloadReadLease> retained_promotion_sources;
        if (state_.device_id.is_gpu())
        {
            auto original = hit.blocks.begin();
            for (PrefixBlockHandle &handle : restore_blocks)
            {
                while (original != hit.blocks.end() && original->key != handle.key)
                    ++original;
                if (original == hit.blocks.end())
                    return fail("restore source lacks its selected archive identity");
                // A row-only read is a consumer lease, never a replacement
                // for the richer archive stored under this same key.
                if (handle.total_bytes != original->total_bytes)
                    continue;
                if (handle.tier == PrefixStorageTier::DeviceHot ||
                    !prefix_cache_->deviceHotCapacityEligible(
                        handle.total_bytes))
                {
                    continue;
                }

                PrefixBlockHandle promoted;
                const PrefixDeviceHotLeaseResult promotion =
                    promotePrefixBlockToDeviceHotForRestore(
                        handle, stream, &promoted);
                if (promotion == PrefixDeviceHotLeaseResult::Busy)
                {
                    /*
                     * Another block in this immutable lookup result still
                     * owns the bounded arena slot. Preserve that hot source
                     * and import this block from its configured pinned-RAM
                     * tier; waiting or allocating here would block inference.
                     */
                    PerfStatsCollector::addCounterWithSequence(
                        "prefix_cache",
                        "device_hot_promotion_busy_ram_restores",
                        1.0,
                        {static_cast<uint64_t>(handle.key.block_index)},
                        "restore",
                        state_.device_id.toString(),
                        {{"bytes", std::to_string(handle.total_bytes)},
                         {"policy", "non_blocking_bounded_arena"}});
                    continue;
                }
                if (promotion != PrefixDeviceHotLeaseResult::Acquired)
                {
                    return fail("eligible lower-tier block could not be promoted");
                }
                // Promotion reads the complete lower-tier archive. Keep that
                // distinct read even when the following row import needs less.
                retained_promotion_sources.push_back(PrefixPayloadReadLease::wholeArchive(handle));
                handle = std::move(promoted);
            }
        }

        PrefixArchivePublicationScope restore_completion([&](PrefixArchiveCompletion) {
            if (!state_.device_id.is_gpu())
                return true;
            std::vector<PrefixPayloadReadLease> retained_restore_sources =
                std::move(retained_promotion_sources);
            for (size_t index = 0; index < restore_blocks.size(); ++index)
            {
                const size_t archived_bytes = restore_blocks[index].total_bytes;
                retained_restore_sources.push_back(index + 1u == restore_blocks.size()
                    ? PrefixPayloadReadLease::wholeArchive(std::move(restore_blocks[index]))
                    : PrefixPayloadReadLease::sequenceRows(std::move(restore_blocks[index])));
                if (index + 1u != restore_blocks.size())
                {
                    PerfStatsCollector::addCounter("prefix_cache", "restore_sequence_read_leases",
                        1.0, "restore", state_.device_id.toString());
                    PerfStatsCollector::addCounter("prefix_cache", "restore_unread_section_owner_bytes_excluded",
                        static_cast<double>(archived_bytes - retained_restore_sources.back().retainedBytes()),
                        "restore", state_.device_id.toString());
                }
            }
            return recordLivePrefixMutationReady(
                stream, "populate_prefix_cache", std::move(retained_restore_sources));
        });

        if (restore_model_runtime_state)
        {
            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] restore model-runtime-state device="
                         << state_.device_id.toString()
                         << " cached_tokens=" << hit.cached_tokens
                         << " bytes=" << terminal_block.model_runtime_state_storage->size());
            }
            if (!graph_builder_)
            {
                return fail("model runtime state restore has no graph builder");
            }
            const PrefixCacheRuntimeRestoreResult restore_result =
                graph_builder_->restorePrefixCacheRuntimeState(
                    terminal_block.model_runtime_state_storage->bytes(),
                    stream);
            if (!restore_result)
            {
                return fail("model runtime state restore failed");
            }
            const bool device_rehydration_pending =
                restore_result.device_rehydration_required;
            const bool placement_changed =
                restore_result.placement_effect ==
                PrefixCacheRuntimePlacementEffect::Changed;
            LOG_INFO("[DeviceGraphOrchestrator] Adopted prefix-runtime restore "
                     "transaction"
                     << " device=" << state_.device_id.toString()
                     << " cached_tokens=" << hit.cached_tokens
                     << " placement_effect="
                     << (placement_changed ? "changed" : "unchanged")
                     << " device_payload_rehydration="
                     << (device_rehydration_pending
                             ? "pending"
                             : "not_required"));
            if (placement_changed)
                ++moe_runtime_movement_epoch_;
            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache",
                "model_runtime_state_restores",
                1.0,
                {static_cast<uint64_t>(hit.cached_tokens)},
                "prefix_cache",
                state_.device_id.toString(),
                {{"placement_effect",
                  placement_changed ? "changed" : "unchanged"},
                 {"device_payload_rehydration",
                  device_rehydration_pending ? "required" : "not_required"}});
        }

        for (const auto &handle : restore_blocks)
        {
            const bool device_hot =
                handle.tier == PrefixStorageTier::DeviceHot;
            if (!handle.valid() || !handle.layout.compatiblePayloadShape(prefix_layout_) ||
                (prefix_layout_.fa_layers > 0 && (device_hot
                     ? (!handle.deviceKVKData() ||
                        !handle.deviceKVVData())
                     : (!handle.kvKData() || !handle.kvVData()))))
            {
                return fail("prefix block handle is invalid or layout-incompatible");
            }
            if (state_.device_id.is_gpu() &&
                !handle.waitForPayloadOnStream(stream))
            {
                return fail("prefix block archive readiness wait failed");
            }

            for (int local_layer = 0; local_layer < prefix_layout_.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("prefix full-attention layer index is invalid");
                }
                const auto block_payload_layout =
                    state_.kv_cache->logicalBlockLayout(global_layer, handle.key.token_count);
                if (block_payload_layout.k_bytes > prefix_layout_.bytes_per_fa_layer_k ||
                    block_payload_layout.v_bytes > prefix_layout_.bytes_per_fa_layer_v)
                {
                    return fail("prefix block payload layout exceeds per-layer storage stride");
                }
                const uint8_t *k_src =
                    (device_hot ? handle.deviceKVKData()
                                : handle.kvKData()) +
                    static_cast<size_t>(local_layer) *
                        prefix_layout_.bytes_per_fa_layer_k;
                const uint8_t *v_src =
                    (device_hot ? handle.deviceKVVData()
                                : handle.kvVData()) +
                    static_cast<size_t>(local_layer) *
                        prefix_layout_.bytes_per_fa_layer_v;
                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = handle.key.token_start;
                desc.token_count = handle.key.token_count;
                desc.stream = stream;
                const bool trace_payload_hash =
                    shouldTracePrefixPayloadHash(handle.key, global_layer);
                if (trace_payload_hash && !device_hot)
                {
                    logPrefixPayloadHashTrace(
                        "populate_import_payload",
                        state_.device_id,
                        handle.key,
                        global_layer,
                        local_layer,
                        k_src,
                        block_payload_layout.k_bytes,
                        v_src,
                        block_payload_layout.v_bytes);
                }
                bool imported = false;
                if (state_.device_id.is_gpu())
                {
                    desc.payload_domain =
                        IKVCache::KVCacheLogicalBlockPayloadDomain::Device;
                    if (device_hot)
                    {
                        imported =
                            state_.kv_cache->importLogicalBlock(
                                desc,
                                k_src,
                                v_src);
                    }
                    else
                    {
                        void *k_staging =
                            prefix_archive_device_staging_->k.devicePtr();
                        void *v_staging =
                            prefix_archive_device_staging_->v.devicePtr();
                        if (!k_staging || !v_staging ||
                            !prefix_backend->hostToDeviceOnStream(
                                k_staging,
                                k_src,
                                block_payload_layout.k_bytes,
                                state_.device_id.gpu_ordinal(),
                                stream) ||
                            !prefix_backend->hostToDeviceOnStream(
                                v_staging,
                                v_src,
                                block_payload_layout.v_bytes,
                                state_.device_id.gpu_ordinal(),
                                stream))
                        {
                            return fail("main KV logical block H2D staging failed");
                        }
                        imported =
                            state_.kv_cache->importLogicalBlock(
                                desc,
                                k_staging,
                                v_staging);
                    }
                }
                else
                {
                    imported =
                        state_.kv_cache->importLogicalBlock(desc, k_src, v_src);
                }
                if (!imported)
                {
                    return fail("main KV logical block import failed");
                }
                if (trace_payload_hash && !device_hot)
                {
                    std::vector<uint8_t> live_k(block_payload_layout.k_bytes, 0);
                    std::vector<uint8_t> live_v(block_payload_layout.v_bytes, 0);
                    auto trace_desc = desc;
                    trace_desc.payload_domain =
                        IKVCache::KVCacheLogicalBlockPayloadDomain::Host;
                    if (!state_.kv_cache->exportLogicalBlock(
                            trace_desc,
                            live_k.empty() ? nullptr : live_k.data(),
                            live_v.empty() ? nullptr : live_v.data()))
                    {
                        return fail("main KV logical block post-import export failed");
                    }
                    logPrefixPayloadHashTrace(
                        "populate_import_live",
                        state_.device_id,
                        handle.key,
                        global_layer,
                        local_layer,
                        live_k.empty() ? nullptr : live_k.data(),
                        live_k.size(),
                        live_v.empty() ? nullptr : live_v.data(),
                        live_v.size());
                }
            }

            importPrefixAuxiliaryState(*state_.kv_cache,seq_idx,PrefixAuxiliaryLane::Main,handle,stream);
            if(handle.layout.includes_mtp_state && !state_.mtp_kv_caches.empty())
                importPrefixAuxiliaryState(*state_.mtp_kv_caches[0],seq_idx,PrefixAuxiliaryLane::ShiftedMTP,handle,stream);

            if (device_hot)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "device_hot_direct_kv_restores",
                    1.0,
                    {static_cast<uint64_t>(handle.key.block_index)},
                    "restore",
                    state_.device_id.toString(),
                    {{"tokens", std::to_string(handle.key.token_count)},
                     {"layers", std::to_string(prefix_layout_.fa_layers)}});
            }

            if (prefix_layout_.includes_mtp_state)
            {
                if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
                {
                    return fail("prefix hit contains MTP state but runner has no MTP KV cache");
                }
                if (!importMTPPrefixPayload(
                        *state_.mtp_kv_caches[0],
                        seq_idx,
                        handle,
                        stream,
                        state_.device_id.is_gpu()
                            ? prefix_archive_device_staging_->mtp_k.devicePtr()
                            : nullptr,
                        state_.device_id.is_gpu()
                            ? prefix_archive_device_staging_->mtp_v.devicePtr()
                            : nullptr,
                        prefix_backend,
                        state_.device_id))
                {
                    return fail("MTP prefix payload import failed");
                }
                if (device_hot)
                {
                    PerfStatsCollector::addCounterWithSequence(
                        "prefix_cache",
                        "device_hot_direct_mtp_restores",
                        1.0,
                        {static_cast<uint64_t>(handle.key.block_index)},
                        "restore",
                        state_.device_id.toString(),
                        {{"tokens", std::to_string(handle.key.token_count)}});
                }
            }
        }

        bool hybrid_imported = true;
        if (prefix_layout_.includes_hybrid_state)
        {
            const PrefixBlockHandle &terminal_handle = restore_blocks.back();
            if (state_.device_id.is_gpu())
            {
                auto *hybrid =
                    dynamic_cast<IHybridKVCache *>(state_.kv_cache.get());
                const bool terminal_device_hot =
                    terminal_handle.tier == PrefixStorageTier::DeviceHot;
                void *hybrid_staging =
                    terminal_device_hot
                        ? const_cast<void *>(
                              terminal_handle.deviceHybridData())
                        : prefix_archive_device_staging_->hybrid.devicePtr();
                // Authenticate the source before queuing even the archive staging copy.
                HybridPrefixStateDescriptor source_format;
                source_format.include_host_state = false;
                source_format.include_device_state = true;
                source_format.host_recurrence_layout =
                    terminal_handle.layout.hybrid_host_recurrence_layout;
                source_format.device_recurrence_layout =
                    terminal_handle.layout.hybrid_device_recurrence_layout;
                if (!hybrid ||
                    !hybridPrefixStateLayoutsMatch(hybrid->hybridPrefixStateMetadata(),
                                                   source_format) ||
                    !terminal_handle.has_hybrid_state ||
                    terminal_handle.layout.hybrid_host_state_bytes != 0 ||
                    terminal_handle.layout.hybrid_device_state_bytes == 0 || !hybrid_staging ||
                    (!terminal_device_hot && (!terminal_handle.hybrid_payload ||
                                              !prefix_backend->hostToDeviceOnStream(
                                                  hybrid_staging,
                                                  terminal_handle.hybrid_payload,
                                                  terminal_handle.layout.hybrid_device_state_bytes,
                                                  state_.device_id.gpu_ordinal(),
                                                  stream))))
                {
                    hybrid_imported = false;
                }
                else
                {
                    HybridPrefixStateDescriptor desc;
                    desc.host_recurrence_layout =
                        terminal_handle.layout.hybrid_host_recurrence_layout;
                    desc.device_recurrence_layout =
                        terminal_handle.layout.hybrid_device_recurrence_layout;
                    desc.seq_idx = seq_idx;
                    desc.logical_token_count =
                        terminal_handle.key.token_start +
                        terminal_handle.key.token_count;
                    desc.stream = stream;
                    desc.synchronize = false;
                    desc.include_host_state = false;
                    desc.include_device_state = true;
                    hybrid_imported = hybrid->importHybridPrefixState(
                        desc,
                        nullptr,
                        hybrid_staging);
                }
            }
            else
            {
                hybrid_imported = importHybridPrefixPayload(
                    *state_.kv_cache,
                    terminal_handle,
                    seq_idx,
                    hit.restore_hybrid_state_for_suffix_prefill,
                    /*synchronize=*/true,
                    stream);
            }
        }
        if (!hybrid_imported)
        {
            return fail("hybrid prefix payload import failed");
        }
        if (prefix_layout_.includes_hybrid_state &&
            restore_blocks.back().tier == PrefixStorageTier::DeviceHot)
        {
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "device_hot_direct_hybrid_restores",
                1.0,
                "restore",
                state_.device_id.toString(),
                {{"bytes", std::to_string(
                               restore_blocks.back().layout.hybrid_device_state_bytes)}});
        }

        if (graph_builder_ && graph_builder_->config().mtp.enabled &&
            hit.has_terminal_hidden &&
            restore_blocks.back().has_terminal_hidden &&
            (restore_blocks.back().terminal_hidden ||
             restore_blocks.back().deviceTerminalHiddenData()))
        {
            if (!ensureMTPTerminalHiddenBuffer())
                return fail("terminal hidden buffer allocation failed");
            if (!state_.prefix_terminal_hidden ||
                prefix_layout_.terminal_hidden_bytes == 0)
                return fail("terminal hidden destination is unavailable");
            if (state_.device_id.is_gpu())
            {
                if (!state_.prefix_terminal_hidden->gpu_data_ptr() &&
                    !state_.prefix_terminal_hidden->allocateOnDevice(
                        state_.device_id,
                        stream))
                {
                    return fail("terminal hidden device allocation failed");
                }
                void *hidden_device =
                    state_.prefix_terminal_hidden->gpu_data_ptr();
                const bool terminal_device_hot =
                    restore_blocks.back().tier ==
                    PrefixStorageTier::DeviceHot;
                const void *hidden_source =
                    terminal_device_hot
                        ? restore_blocks.back().deviceTerminalHiddenData()
                        : restore_blocks.back().terminal_hidden;
                if (!hidden_device || !hidden_source ||
                    !(terminal_device_hot
                          ? prefix_backend->deviceCopyAsync(
                                hidden_device,
                                hidden_source,
                                prefix_layout_.terminal_hidden_bytes,
                                state_.device_id.gpu_ordinal(),
                                stream)
                          : prefix_backend->hostToDeviceOnStream(
                                hidden_device,
                                hidden_source,
                                prefix_layout_.terminal_hidden_bytes,
                                state_.device_id.gpu_ordinal(),
                                stream)))
                {
                    return fail("terminal hidden device restore failed");
                }
                TransferEngine::publishDeviceWrite(
                    state_.prefix_terminal_hidden.get(),
                    state_.device_id,
                    stream);
            }
            else
            {
                void *hidden_host =
                    state_.prefix_terminal_hidden->raw_mutable_data();
                if (!hidden_host)
                    return fail("terminal hidden host destination is unavailable");
                std::memcpy(
                    hidden_host,
                    restore_blocks.back().terminal_hidden,
                    prefix_layout_.terminal_hidden_bytes);
                TransferEngine::publishHostWrite(
                    state_.prefix_terminal_hidden.get());
            }
            state_.mtp_terminal_hidden_publication.publishPrefixRestore();
            if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                arena_->markWrittenFlagsOnly(
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
            }
            if (restore_blocks.back().tier == PrefixStorageTier::DeviceHot)
            {
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "device_hot_direct_terminal_hidden_restores",
                    1.0,
                    "restore",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(
                                   prefix_layout_.terminal_hidden_bytes)}});
            }
        }

        state_.positions[seq_idx] = hit.cached_tokens;
        state_.sequence_lengths[seq_idx] = hit.cached_tokens;
        handleLivePrefixReplayStateAfterMutation(
            LivePrefixMutationReason::PrefixRestore,
            "populate_prefix_cache");

        /*
         * Row imports read KV/MTP sections of earlier checkpoints and the
         * complete final checkpoint. Promotions have their own whole-archive
         * sources. Retain exactly those storage owners through the mutation
         * event so unused recurrent images can retire under RAM pressure.
         */
        restore_completion.commit();
        PerfStatsCollector::addCounterWithSequence(
            "prefix_cache",
            "populate_restores",
            1.0,
            {static_cast<uint64_t>(hit.cached_tokens), static_cast<uint64_t>(hit.blocks.size())},
            "restore",
            state_.device_id.toString(),
            {{"includes_hybrid_state", prefix_layout_.includes_hybrid_state ? "true" : "false"},
             {"includes_mtp_state", prefix_layout_.includes_mtp_state ? "true" : "false"},
             {"terminal_logits", hit.has_terminal_logits ? "true" : "false"},
             {"terminal_hidden", hit.has_terminal_hidden ? "true" : "false"},
             {"model_runtime_state", restore_model_runtime_state ? "true" : "false"}});
        return true;
    }

    bool DeviceGraphOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        void *terminal_restore_stream = nullptr;
        TensorBase *terminal_logits_tensor =
            activeMainLogitsAreColumnParallel() ? state_.logits_local.get() : state_.logits.get();
        TensorBase *published_terminal_logits_tensor = terminal_logits_tensor;
        const BufferId terminal_logits_buffer_id =
            terminal_logits_tensor == state_.logits_local.get() ? BufferId::LOGITS_LOCAL : BufferId::LOGITS;

        if (!prefix_layout_.includes_terminal_logits ||
            prefix_layout_.terminal_logits_bytes == 0)
        {
            return true;
        }
        if (!hit.has_terminal_logits || hit.blocks.empty() || !terminal_logits_tensor)
        {
            return false;
        }

        PrefixBlockHandle terminal = hit.payload_plan
            ? hit.payload_plan->terminal(hit.blocks.back().key) : hit.blocks.back();
        if (state_.device_id.is_gpu() &&
            prefix_cache_ &&
            terminal.tier != PrefixStorageTier::DeviceHot)
        {
            /*
             * populatePrefix() may have promoted this lower-tier block after
             * the immutable lookup result was assembled. Resolve the installed
             * hot owner so terminal logits and hidden state reuse the same VRAM
             * archive instead of issuing a second H2D transfer.
             */
            if (auto promoted =
                    prefix_cache_->deviceHotCopy(terminal.key))
            {
                terminal = std::move(*promoted);
            }
        }
        if (!terminal.has_terminal_logits ||
            (!terminal.terminal_logits &&
             !terminal.deviceTerminalLogitsData()))
        {
            return false;
        }

        if (fp32LogitsRowBytes(terminal_logits_tensor) < prefix_layout_.terminal_logits_bytes)
        {
            return false;
        }

        if (state_.device_id.is_gpu())
        {
            terminal_restore_stream =
                explicitGPUStreamForOperation("restorePrefixTerminalState");
            if (!terminal_restore_stream)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal logits: "
                          "missing explicit GPU stream");
                return false;
            }
            /*
             * Full-hit restore is two ordered mutations: populatePrefix()
             * restores KV/GDN/MTP payloads, then this method restores the
             * terminal logits and hidden row.  Consume the first publication
             * through the typed timeline before reusing the mutation event for
             * the second transaction.  The join is a stream event wait even
             * when both operations currently share a context stream; no host
             * synchronization or implicit same-stream assumption is allowed.
             */
            if (!joinPublishedLiveStateHandoffs(
                    terminal_restore_stream,
                    DeviceTimelineRole::PrefixRestoreMutation,
                    "restore_prefix_terminal_state_previous_mutation"))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to join bulk prefix "
                    "restore before terminal-state restoration");
                return false;
            }
            if (!prepareLivePrefixMutationReadyEvent(
                    terminal_restore_stream,
                    "restore_prefix_terminal_state"))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to preflight restored "
                    "prefix terminal-state completion event");
                return false;
            }
            if (!terminal.waitForPayloadOnStream(terminal_restore_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal logits: "
                          "archive readiness wait failed");
                return false;
            }
            if (!terminal_logits_tensor->gpu_data_ptr() &&
                !terminal_logits_tensor->allocateOnDevice(
                    state_.device_id,
                    terminal_restore_stream))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to allocate restored prefix terminal logits");
                return false;
            }
            IBackend *backend = getBackendFor(state_.device_id);
            const bool device_hot =
                terminal.tier == PrefixStorageTier::DeviceHot;
            const void *logits_source =
                device_hot
                    ? terminal.deviceTerminalLogitsData()
                    : terminal.terminal_logits;
            if (!backend || !logits_source ||
                !(device_hot
                      ? backend->deviceCopyAsync(
                            terminal_logits_tensor->gpu_data_ptr(),
                            logits_source,
                            prefix_layout_.terminal_logits_bytes,
                            state_.device_id.gpu_ordinal(),
                            terminal_restore_stream)
                      : backend->hostToDeviceOnStream(
                            terminal_logits_tensor->gpu_data_ptr(),
                            logits_source,
                            prefix_layout_.terminal_logits_bytes,
                            state_.device_id.gpu_ordinal(),
                            terminal_restore_stream)))
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Failed to submit restored prefix terminal logits device copy");
                return false;
            }
            TransferEngine::publishDeviceWrite(
                terminal_logits_tensor,
                state_.device_id,
                terminal_restore_stream);
            if (device_hot)
            {
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "device_hot_direct_terminal_logits_restores",
                    1.0,
                    "restore",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(
                                   prefix_layout_.terminal_logits_bytes)}});
            }
        }
        else
        {
            void *dst = terminal_logits_tensor->raw_mutable_data();
            if (!dst)
            {
                return false;
            }
            std::memcpy(
                dst,
                terminal.terminal_logits,
                prefix_layout_.terminal_logits_bytes);
            TransferEngine::publishHostWrite(terminal_logits_tensor);
        }
        if (arena_ && arena_->isRegistered(terminal_logits_buffer_id))
        {
            arena_->markWrittenFlagsOnly(
                terminal_logits_buffer_id,
                state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
        }

        if (terminal_logits_tensor == state_.logits_local.get() &&
            state_.device_id.is_cpu() &&
            state_.logits &&
            mpi_ctx_ &&
            mpi_ctx_->world_size() > 1)
        {
            /*
             * Global/NodeLocal TP stores the authoritative terminal prefix row
             * as one column-parallel LOGITS_LOCAL shard per MPI participant.
             * The normal forward graph immediately follows LM_HEAD_LOCAL with
             * `lm_head_allgather`, so CPU stochastic sampling reads the complete
             * row from LOGITS. A prefix restore must reproduce that same graph
             * boundary before advertising terminal logits as ready.
             *
             * Without this compact one-row collective, LOGITS retains the last
             * decode row from the previous request. The restored KV, hidden,
             * and local logit shard are all correct, but the first MTP target
             * sample is taken from stale full-vocabulary bytes. Execute the
             * existing AllGatherStage contract for exactly one row; this is the
             * same grouped collective used by production forward, not serial
             * row replay and not an ad hoc host reconstruction.
             *
             * GPU restores intentionally do not enter this host-visible path.
             * GPU LocalTP uses mirrored device-owned heads, while supported GPU
             * GlobalTP sampling coordinates resident shard candidates.
             */
            AllGatherStage::Params gather_params;
            gather_params.device_id = DeviceId::cpu();
            gather_params.mpi_ctx = mpi_ctx_.get();
            gather_params.local_input = state_.logits_local.get();
            gather_params.full_output = state_.logits.get();
            gather_params.actual_seq_len = 1;
            gather_params.input_buffer_id = BufferId::LOGITS_LOCAL;
            gather_params.output_buffer_id = BufferId::LOGITS;

            AllGatherStage gather_stage(std::move(gather_params));
            if (!gather_stage.execute(/*ctx=*/nullptr))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to rebuild the CPU "
                    "GlobalTP full-vocabulary terminal row after prefix restore");
                return false;
            }
            TransferEngine::publishHostWrite(state_.logits.get());
            published_terminal_logits_tensor = state_.logits.get();
            if (arena_ && arena_->isRegistered(BufferId::LOGITS))
            {
                arena_->markWrittenFlagsOnly(
                    BufferId::LOGITS,
                    DeviceId::cpu());
            }
            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache",
                "cpu_global_tp_terminal_logits_allgathers",
                1.0,
                {static_cast<uint64_t>(hit.cached_tokens)},
                "restore",
                state_.device_id.toString(),
                {{"rows", "1"},
                 {"participants", std::to_string(mpi_ctx_->world_size())},
                 {"source", "restored_logits_local"},
                 {"destination", "logits"}});
        }
        if (prefix_cache_)
        {
            prefix_cache_->recordTerminalStateHit();
        }

        if (graph_builder_ && graph_builder_->config().mtp.enabled &&
            terminal.has_terminal_hidden &&
            (terminal.terminal_hidden ||
             terminal.deviceTerminalHiddenData()))
        {
            if (!ensureMTPTerminalHiddenBuffer())
                return false;
            if (!state_.prefix_terminal_hidden ||
                prefix_layout_.terminal_hidden_bytes == 0)
                return false;
            if (state_.device_id.is_gpu())
            {
                if (!terminal_restore_stream)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal hidden: "
                              "missing explicit GPU stream");
                    return false;
                }
                if (!terminal.waitForPayloadOnStream(terminal_restore_stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to upload restored prefix terminal hidden: "
                              "archive readiness wait failed");
                    return false;
                }
                if (!state_.prefix_terminal_hidden->gpu_data_ptr() &&
                    !state_.prefix_terminal_hidden->allocateOnDevice(
                        state_.device_id,
                        terminal_restore_stream))
                {
                    return false;
                }
                IBackend *backend = getBackendFor(state_.device_id);
                const bool device_hot =
                    terminal.tier == PrefixStorageTier::DeviceHot;
                const void *hidden_source =
                    device_hot
                        ? terminal.deviceTerminalHiddenData()
                        : terminal.terminal_hidden;
                if (!backend || !hidden_source ||
                    !(device_hot
                          ? backend->deviceCopyAsync(
                                state_.prefix_terminal_hidden->gpu_data_ptr(),
                                hidden_source,
                                prefix_layout_.terminal_hidden_bytes,
                                state_.device_id.gpu_ordinal(),
                                terminal_restore_stream)
                          : backend->hostToDeviceOnStream(
                                state_.prefix_terminal_hidden->gpu_data_ptr(),
                                hidden_source,
                                prefix_layout_.terminal_hidden_bytes,
                                state_.device_id.gpu_ordinal(),
                                terminal_restore_stream)))
                {
                    return false;
                }
                TransferEngine::publishDeviceWrite(
                    state_.prefix_terminal_hidden.get(),
                    state_.device_id,
                    terminal_restore_stream);
            }
            else
            {
                void *hidden_dst =
                    state_.prefix_terminal_hidden->raw_mutable_data();
                if (!hidden_dst || !terminal.terminal_hidden)
                    return false;
                std::memcpy(
                    hidden_dst,
                    terminal.terminal_hidden,
                    prefix_layout_.terminal_hidden_bytes);
                TransferEngine::publishHostWrite(
                    state_.prefix_terminal_hidden.get());
            }
            state_.mtp_terminal_hidden_publication.publishPrefixRestore();
            if (arena_ && arena_->isRegistered(BufferId::PREFIX_TERMINAL_HIDDEN))
            {
                arena_->markWrittenFlagsOnly(
                    BufferId::PREFIX_TERMINAL_HIDDEN,
                    state_.device_id.is_gpu() ? state_.device_id : DeviceId::cpu());
            }
        }

        if (state_.device_id.is_gpu())
        {
            std::vector<PrefixPayloadReadLease> retained_terminal_sources;
            retained_terminal_sources.push_back(PrefixPayloadReadLease::wholeArchive(std::move(terminal)));
            if (!recordLivePrefixMutationReady(
                    terminal_restore_stream,
                    "restore_prefix_terminal_state",
                    std::move(retained_terminal_sources)))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Failed to publish terminal "
                    "prefix restore source ownership");
                return false;
            }
        }

        const ForwardLogitsPublicationDescriptor restored_logits{
            .execution_role = ForwardExecutionRole::MainInference,
            .logical_all_position_logits = false,
            .storage_surface = classifyForwardLogitsSurface(
                published_terminal_logits_tensor),
        };
        if (!publishCurrentMainLogits(
                published_terminal_logits_tensor,
                restored_logits,
                MainLogitsPublicationSource::PrefixTerminalRestore,
                "restore_prefix_terminal_state"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix terminal restore could not publish current main-logits ownership");
            return false;
        }

        return true;
    }

    bool DeviceGraphOrchestrator::preparePrefixHarvest(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        const PrefixHarvestSchedule &schedule)
    {
        if (!admission.supported || !admission.cache_enabled ||
            !prefix_cache_ || admission.fingerprint_key == 0u)
            return false;
        // As at harvest, a moved namespace cannot be relabeled. Preparing its
        // obsolete payloads would merely create avoidable background traffic.
        if (admission.fingerprint_key != prefix_identity_.key)
            return true;
        const auto preparation = prefix_cache_->prepareHarvest(
            admission, tokens, schedule, prefix_layout_,
            graph_builder_ ? graph_builder_->prefixCacheRuntimeStateCapacity() : 0u);
        if (preparation == PrefixRamInsertPreparation::Error)
            return false;
        PerfStatsCollector::addCounterWithSequence(
            "prefix_cache",
            "prefill_archive_preparations",
            1.0,
            {static_cast<uint64_t>(schedule.promptTokens()),
             static_cast<uint64_t>(schedule.reusableCheckpoints().size())},
            "admission",
            state_.device_id.toString(),
            {{"state", preparation == PrefixRamInsertPreparation::Busy ? "pending" : "prepared"}});
        for (const int checkpoint : schedule.reusableCheckpoints())
            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache", "reusable_checkpoints_planned", 1.0,
                {static_cast<uint64_t>(checkpoint)}, "admission",
                state_.device_id.toString());
        return true;
    }

    bool DeviceGraphOrchestrator::harvestPrefix(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        if (!admission.cache_enabled ||
            !admission.supported ||
            admission.fingerprint_key == 0)
        {
            LOG_ERROR(
                "[DeviceGraphOrchestrator] Prefix harvest rejected an invalid "
                "request admission identity"
                << " cache_enabled=" << (admission.cache_enabled ? "yes" : "no")
                << " supported=" << (admission.supported ? "yes" : "no")
                << " fingerprint="
                << prefixHashHex(admission.fingerprint_key));
            return false;
        }
        if (prompt_token_count <= 0 || prompt_token_count > static_cast<int>(tokens.size()))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest rejected invalid prompt_token_count="
                      << prompt_token_count << " token_count=" << tokens.size());
            return false;
        }
        if (!ensurePrefixCacheReady() || !state_.kv_cache || !prefix_ram_backend_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest rejected unavailable cache state: cache_ready="
                      << (prefix_cache_ ? "yes" : "no")
                      << " kv_cache=" << (state_.kv_cache ? "yes" : "no")
                      << " ram_backend=" << (prefix_ram_backend_ ? "yes" : "no")
                      << " bypass_reason=" << prefix_cache_bypass_reason_);
            return false;
        }

        void *stream = explicitGPUStreamForOperation("harvestPrefix");
        if (state_.device_id.is_gpu() && !stream)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest has no explicit GPU stream");
            return false;
        }
        if (!waitForLiveInferenceStateReadyForObservation(
                stream,
                "harvest_prefix",
                DeviceTimelineRole::PrefixCheckpointArchive))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest could not order its observation after live inference");
            return false;
        }

        if (!refreshPrefixPayloadLayoutForLiveHybridState("harvestPrefix"))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest could not refresh its live payload layout");
            return false;
        }

        // Completed restore reads no longer need their source aliases. Poll
        // their exact events before admitting new archives, without waiting on
        // a stream or mistaking retired cache metadata for released PMA bytes.
        if (state_.device_id.is_gpu())
            retirePendingPrefixPayloadUses(false);

        const uint64_t request_fingerprint = admission.fingerprint_key;
        if (request_fingerprint != prefix_identity_.key)
        {
            /*
             * InvalidateOnRebalance deliberately keys cached state by the
             * ExpertOverlay movement epoch.  The request may retain its old
             * RCU placement lease while background maintenance publishes a
             * newer epoch.  Its output remains a valid inference result, but
             * archiving that output under the new live fingerprint would make
             * stale request state appear current.  Discard the archive write;
             * the next request will look up and harvest under the new epoch.
             */
            PerfStatsCollector::addCounter(
                "prefix_cache",
                "stale_request_harvest_discards",
                1.0,
                "harvest",
                state_.device_id.toString(),
                {
                    {"admission_fingerprint", prefixHashHex(request_fingerprint)},
                    {"live_fingerprint", prefixHashHex(prefix_identity_.key)},
                    {"admission_movement_epoch", std::to_string(admission.placement_epochs.earliest())},
                    {"live_movement_epoch", std::to_string(moeRuntimeMovementEpoch())},
                });
            LOG_DEBUG(
                "[DeviceGraphOrchestrator] Prefix harvest discarded after "
                "request-scoped fingerprint became stale"
                << " admission_fingerprint="
                << prefixHashHex(request_fingerprint)
                << " live_fingerprint="
                << prefixHashHex(prefix_identity_.key)
                << " admission_movement_epoch="
                << admission.placement_epochs.earliest()
                << " live_movement_epoch="
                << moeRuntimeMovementEpoch());
            return true;
        }
        IBackend *prefix_backend =
            state_.device_id.is_gpu() ? getBackendFor(state_.device_id) : nullptr;
        if (state_.device_id.is_gpu() &&
            (!prefix_backend || !prefix_archive_device_staging_))
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest GPU staging is unavailable");
            return false;
        }

        const int complete_blocks =
            (prompt_token_count + prefix_layout_.block_size - 1) /
            prefix_layout_.block_size;

        uint64_t parent_hash = 0;
        for (int block = 0; block < complete_blocks; ++block)
        {
            PrefixCacheKey key = makePrefixKeyForBlock(
                request_fingerprint,
                tokens,
                block,
                parent_hash);
            parent_hash = key.stableHash();
            const bool terminal_block =
                (key.token_start + key.token_count) == prompt_token_count;
            if (!terminal_block && prefix_layout_.organization() ==
                    PrefixPayloadOrganization::RecurrentCheckpoint)
            {
                // Recurrent state summarizes the entire prompt, not each
                // attention chunk. Store its one real checkpoint and keep
                // ancestry in the authenticated key rather than allocating
                // empty/fake K/V blocks or copying the same state repeatedly.
                continue;
            }
            const bool key_already_installed = prefix_cache_->contains(key);
            const bool reuse_admitted_terminal =
                terminal_block && key_already_installed &&
                admission.terminalHarvestDisposition(
                    key,
                    prompt_token_count) ==
                    PrefixTerminalHarvestDisposition::ReuseAdmittedArchive;
            if (key_already_installed &&
                (!terminal_block || reuse_admitted_terminal))
            {
                if (reuse_admitted_terminal)
                {
                    /*
                     * The full-hit restore consumed this exact immutable
                     * terminal archive. Replacing it would duplicate every
                     * pinned section while the restore event still retains
                     * the admitted owner. Under a full RAM tier that duplicate
                     * violates the canonical physical-memory reservation even
                     * though logical LRU accounting has retired the old key.
                     * Preserve the archive and avoid all redundant D2H work.
                     */
                    PerfStatsCollector::addCounterWithSequence(
                        "prefix_cache",
                        "exact_terminal_archive_reuses",
                        1.0,
                        {static_cast<uint64_t>(block)},
                        "harvest",
                        state_.device_id.toString(),
                        {{"tokens", std::to_string(key.token_count)}});
                }
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] harvest skip-existing device=" << state_.device_id.toString()
                                                                            << " block=" << block
                                                                            << " key=" << key.toHex()
                                                                            << " disposition="
                                                                            << (reuse_admitted_terminal
                                                                                    ? "reuse_exact_terminal"
                                                                                    : "reuse_nonterminal"));
                }
                continue;
            }

            PrefixPayloadLayout block_layout = prefix_layout_;
            if (!terminal_block)
            {
                block_layout.includes_hybrid_state = false;
                block_layout.includes_terminal_hidden = false;
                block_layout.includes_terminal_logits = false;
            }

            applyPrefixAuxiliaryLayout(block_layout,*state_.kv_cache,
                state_.mtp_kv_caches.empty()?nullptr:state_.mtp_kv_caches[0].get(),0,
                key.token_start,key.token_count,key.token_start+key.token_count,terminal_block);

            applyHybridPrefixPayloadExtent(block_layout, *state_.kv_cache, 0);

            std::shared_ptr<std::vector<uint8_t>> model_runtime_state;
            if (terminal_block && graph_builder_)
            {
                auto captured = std::make_shared<std::vector<uint8_t>>();
                if (!graph_builder_->capturePrefixCacheRuntimeState(*captured, stream))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: model runtime state capture failed");
                    return false;
                }
                if (captured->size() > graph_builder_->prefixCacheRuntimeStateCapacity())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Model prefix serializer exceeded its declared capacity");
                    return false;
                }
                if (!captured->empty())
                    model_runtime_state = std::move(captured);
            }

            const size_t model_runtime_state_bytes =
                model_runtime_state ? model_runtime_state->size() : 0u;
            const size_t total_block_bytes =
                block_layout.totalBytes() + model_runtime_state_bytes;

            // Early request preparation overlaps writes with prefill, but a
            // short tail can finish first. This required storage boundary joins
            // only the receipt needed to admit its original physical payload;
            // write/fsync stay on the background worker. Busy never
            // means permission to drop the terminal state or overcommit PMA.
            prefix_cache_->completeInsertPreparation(key,
                PrefixPayloadAllocationPlan::archive(block_layout, model_runtime_state_bytes));

            std::string archive_allocation_error;
            PrefixBlockHandle handle = prefix_ram_backend_->allocateWithDiagnostics(
                key, block_layout, &archive_allocation_error);
            if (!handle.valid())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: RAM allocate rejected key="
                          << key.toHex()
                          << " block_bytes=" << block_layout.totalBytes()
                          << " reason=" << archive_allocation_error);
                return false;
            }
            if (model_runtime_state &&
                !prefix_ram_backend_->attachModelRuntimeState(
                    &handle, std::move(model_runtime_state)))
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Prefix harvest failed: "
                    "model runtime state exceeded RAM tier authority key="
                    << key.toHex()
                    << " block_bytes=" << total_block_bytes);
                prefix_ram_backend_->release(handle);
                return false;
            }
            if (handle.total_bytes != total_block_bytes)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Prefix harvest produced a RAM "
                    "accounting mismatch key="
                    << key.toHex() << " expected=" << total_block_bytes
                    << " actual=" << handle.total_bytes);
                prefix_ram_backend_->release(handle);
                return false;
            }

            PrefixBlockHandle device_hot_handle;
            const bool prepare_device_hot =
                prefix_cache_->deviceHotCapacityEligible(
                    total_block_bytes);
            const PrefixDeviceHotLeaseResult hot_lease =
                prepare_device_hot
                    ? prefix_cache_->prepareDeviceHotCopy(
                          handle, stream, &device_hot_handle)
                    : PrefixDeviceHotLeaseResult::Ineligible;
            if (hot_lease == PrefixDeviceHotLeaseResult::Error)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: "
                          "device-hot arena lifecycle rejected key="
                          << key.toHex()
                          << " block_bytes=" << total_block_bytes);
                prefix_ram_backend_->release(handle);
                return false;
            }
            if (hot_lease == PrefixDeviceHotLeaseResult::Busy)
            {
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "device_hot_harvest_busy_ram_only_stores",
                    1.0,
                    {static_cast<uint64_t>(key.block_index)},
                    "harvest",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(total_block_bytes)},
                     {"policy", "non_blocking_bounded_arena"}});
            }

            PrefixArchivePublicationScope ram_completion([&](PrefixArchiveCompletion) {
                return !state_.device_id.is_gpu() || !handle.payload_readiness || !handle.payload_readiness->prepared() ||
                    recordPrefixArchivePayloadReady(&handle,stream,"harvest_prefix_block");
            });
            PrefixArchivePublicationScope hot_completion([&](PrefixArchiveCompletion) {
                return !state_.device_id.is_gpu() || !device_hot_handle.payload_readiness || !device_hot_handle.payload_readiness->prepared() ||
                    recordPrefixArchivePayloadReady(&device_hot_handle,stream,"harvest_prefix_device_hot_block");
            });
            /*
             * Event allocation must precede the first asynchronous payload
             * operation. Once a D2H or D2D copy is accepted, both handles own
             * in-flight storage and completion-record failure is fatal.
             */
            if (state_.device_id.is_gpu() &&
                (!preparePrefixArchivePayloadReady(
                     &handle,
                     stream,
                     "harvest_prefix_block") ||
                 (device_hot_handle.valid() &&
                  !preparePrefixArchivePayloadReady(
                      &device_hot_handle,
                      stream,
                      "harvest_prefix_device_hot_block"))))
            {
                if (device_hot_handle.valid() && prefix_device_hot_backend_)
                    (void)prefix_device_hot_backend_->release(device_hot_handle);
                (void)prefix_ram_backend_->release(handle);
                return false;
            }

            bool ok = true;
            bool terminal_hidden_archival_refresh = false;
            for (int local_layer = 0; local_layer < prefix_layout_.fa_layers && ok; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: no FA layer for local index "
                              << local_layer);
                    ok = false;
                    break;
                }
                const auto block_payload_layout =
                    state_.kv_cache->logicalBlockLayout(global_layer, key.token_count);
                if (block_payload_layout.k_bytes > block_layout.bytes_per_fa_layer_k ||
                    block_payload_layout.v_bytes > block_layout.bytes_per_fa_layer_v)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: logical block payload layout "
                              "exceeds per-layer storage stride for layer="
                              << global_layer
                              << " token_start=" << key.token_start
                              << " token_count=" << key.token_count
                              << " k_bytes=" << block_payload_layout.k_bytes
                              << " k_stride=" << block_layout.bytes_per_fa_layer_k
                              << " v_bytes=" << block_payload_layout.v_bytes
                              << " v_stride=" << block_layout.bytes_per_fa_layer_v);
                    ok = false;
                    break;
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * prefix_layout_.bytes_per_fa_layer_v;
                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = 0;
                desc.logical_token_start = key.token_start;
                desc.token_count = key.token_count;
                desc.stream = stream;
                if (state_.device_id.is_gpu())
                {
                    void *k_staging =
                        prefix_archive_device_staging_->k.devicePtr();
                    void *v_staging =
                        prefix_archive_device_staging_->v.devicePtr();
                    desc.payload_domain =
                        IKVCache::KVCacheLogicalBlockPayloadDomain::Device;
                    ok = k_staging && v_staging &&
                         state_.kv_cache->exportLogicalBlock(
                             desc,
                             k_staging,
                             v_staging) &&
                         prefix_backend->deviceToHostOnStream(
                             k_dst,
                             k_staging,
                             block_payload_layout.k_bytes,
                             state_.device_id.gpu_ordinal(),
                             stream) &&
                         prefix_backend->deviceToHostOnStream(
                             v_dst,
                             v_staging,
                             block_payload_layout.v_bytes,
                             state_.device_id.gpu_ordinal(),
                             stream) &&
                         (!device_hot_handle.valid() ||
                          (device_hot_handle.deviceKVKData() &&
                           device_hot_handle.deviceKVVData() &&
                           prefix_backend->deviceCopyAsync(
                               device_hot_handle.deviceKVKData() +
                                   static_cast<size_t>(local_layer) *
                                       prefix_layout_.bytes_per_fa_layer_k,
                               k_staging,
                               block_payload_layout.k_bytes,
                               state_.device_id.gpu_ordinal(),
                               stream) &&
                           prefix_backend->deviceCopyAsync(
                               device_hot_handle.deviceKVVData() +
                                   static_cast<size_t>(local_layer) *
                                       prefix_layout_.bytes_per_fa_layer_v,
                               v_staging,
                               block_payload_layout.v_bytes,
                               state_.device_id.gpu_ordinal(),
                               stream)));
                }
                else
                {
                    ok = state_.kv_cache->exportLogicalBlock(
                        desc,
                        k_dst,
                        v_dst);
                }
                if (ok && shouldTracePrefixPayloadHash(key, global_layer))
                {
                    /*
                     * GPU archive bytes are still being filled asynchronously.
                     * The deferred block-ready event remains the authority; a
                     * host hash here would race DMA or force a synchronization.
                     */
                    if (!state_.device_id.is_gpu())
                    {
                        logPrefixPayloadHashTrace(
                            "harvest_export",
                            state_.device_id,
                            key,
                            global_layer,
                            local_layer,
                            k_dst,
                            block_payload_layout.k_bytes,
                            v_dst,
                            block_payload_layout.v_bytes);
                    }
                }
                if (!ok)
                {
                    const auto seq_state = state_.kv_cache->sequenceState(global_layer, desc.seq_idx);
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: exportLogicalBlock layer="
                              << global_layer
                              << " token_start=" << desc.logical_token_start
                              << " token_count=" << desc.token_count
                              << " cached_tokens=" << seq_state.cached_tokens
                              << " first_layer_index=" << state_.kv_cache->first_layer_index()
                              << " total_layers=" << state_.kv_cache->n_layers());
                }
            }

            if (ok && prefix_layout_.includes_mtp_state)
            {
                if (state_.mtp_kv_caches.empty() || !state_.mtp_kv_caches[0])
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: missing MTP KV cache");
                    ok = false;
                }
                else
                {
                    ok = exportMTPPrefixPayload(
                        *state_.mtp_kv_caches[0],
                        0,
                        key,
                        &handle,
                        device_hot_handle.valid()
                            ? &device_hot_handle
                            : nullptr,
                        stream,
                        state_.device_id.is_gpu()
                            ? prefix_archive_device_staging_->mtp_k.devicePtr()
                            : nullptr,
                        state_.device_id.is_gpu()
                            ? prefix_archive_device_staging_->mtp_v.devicePtr()
                            : nullptr,
                        prefix_backend,
                        state_.device_id);
                    if (!ok)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: MTP payload export failed");
                    }
                }
            }

            if(ok) {
                exportPrefixAuxiliaryState(*state_.kv_cache,0,PrefixAuxiliaryLane::Main,handle,stream);
                if(device_hot_handle.valid())
                    exportPrefixAuxiliaryState(*state_.kv_cache,0,PrefixAuxiliaryLane::Main,device_hot_handle,stream);
                if(block_layout.includes_mtp_state && !state_.mtp_kv_caches.empty()) {
                    exportPrefixAuxiliaryState(*state_.mtp_kv_caches[0],0,PrefixAuxiliaryLane::ShiftedMTP,handle,stream);
                    if(device_hot_handle.valid())
                        exportPrefixAuxiliaryState(*state_.mtp_kv_caches[0],0,PrefixAuxiliaryLane::ShiftedMTP,device_hot_handle,stream);
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_hybrid_state)
            {
                if (state_.device_id.is_gpu())
                {
                    const auto *hybrid =
                        dynamic_cast<const IHybridKVCache *>(state_.kv_cache.get());
                    void *hybrid_staging =
                        prefix_archive_device_staging_->hybrid.devicePtr();
                    if (!hybrid ||
                        block_layout.hybrid_host_state_bytes != 0 ||
                        block_layout.hybrid_device_state_bytes == 0 ||
                        !handle.hybrid_payload ||
                        !hybrid_staging)
                    {
                        ok = false;
                    }
                    else
                    {
                        HybridPrefixStateDescriptor desc;
                        desc.host_recurrence_layout = block_layout.hybrid_host_recurrence_layout;
                        desc.device_recurrence_layout =
                            block_layout.hybrid_device_recurrence_layout;
                        desc.seq_idx = 0;
                        desc.logical_token_count =
                            key.token_start + key.token_count;
                        desc.stream = stream;
                        desc.synchronize = false;
                        desc.include_host_state = false;
                        desc.include_device_state = true;
                        ok = hybrid->exportHybridPrefixState(
                                 desc,
                                 nullptr,
                                 hybrid_staging) &&
                             prefix_backend->deviceToHostOnStream(
                                 handle.hybrid_payload,
                                 hybrid_staging,
                                 block_layout.hybrid_device_state_bytes,
                                 state_.device_id.gpu_ordinal(),
                                 stream) &&
                             (!device_hot_handle.valid() ||
                              (device_hot_handle.deviceHybridData() &&
                               prefix_backend->deviceCopyAsync(
                                   device_hot_handle.deviceHybridData(),
                                   hybrid_staging,
                                   block_layout.hybrid_device_state_bytes,
                                   state_.device_id.gpu_ordinal(),
                                   stream)));
                        handle.has_hybrid_state = ok;
                        device_hot_handle.has_hybrid_state =
                            device_hot_handle.valid() && ok;
                    }
                }
                else
                {
                    ok = exportHybridPrefixPayload(
                        *state_.kv_cache,
                        0,
                        key.token_start + key.token_count,
                        &handle,
                        /*synchronize=*/true,
                        stream);
                }
                if (!ok)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: hybrid payload export failed");
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_terminal_hidden &&
                !state_.mtp_terminal_hidden_publication.current())
            {
                /*
                 * Prefix full hits with MTP need the terminal hidden row from
                 * the prompt prefill before the first draft sidecar can run.
                 * MTP sidecar execution lazily refreshes this row too, but
                 * prefix harvest happens immediately after prefill and before
                 * decode.  Refresh it here so a stored terminal block never
                 * advertises MTP state while silently lacking terminal hidden.
                 *
                 * This refresh is an archival read for the prefix-cache block,
                 * not a decode-state publication.  Leave the real sidecar
                 * input contract untouched after the block copies the row so
                 * the following MTP decode step refreshes terminal hidden at
                 * the same transaction boundary as the no-prefix path.
                 */
                ok = refreshMTPTerminalHiddenState(prompt_token_count, 1);
                if (!ok)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: terminal hidden refresh failed");
                }
                else
                {
                    terminal_hidden_archival_refresh = true;
                    PerfStatsCollector::addCounter(
                        "prefix_cache",
                        "harvest_terminal_hidden_archival_refreshes",
                        1.0,
                        "harvest",
                        state_.device_id.toString(),
                        {{"prompt_token_count", std::to_string(prompt_token_count)}});
                }
            }

            const auto terminal_logits = currentPrefixArchiveLogits();
            if (ok && terminal_block && prefix_layout_.includes_terminal_logits &&
                !terminal_logits)
            {
                LOG_ERROR(
                    "[DeviceGraphOrchestrator] Prefix harvest has no current "
                    "scalar main-logits publication");
                ok = false;
            }
            if (ok && terminal_block && prefix_layout_.includes_terminal_logits &&
                terminal_logits && handle.terminal_logits)
            {
                // Slice the actual producer, not a dormant local allocation.
                // This is the same checked interval used by prefix diagnostics.
                TensorBase *terminal_logits_tensor = terminal_logits->tensor;
                const size_t logits_offset = terminal_logits->slice.byteOffset();
                const size_t logits_bytes = terminal_logits->slice.byteCount();
                if (state_.device_id.is_gpu())
                {
                    const auto *logits_base = static_cast<const uint8_t *>(
                        terminal_logits_tensor->gpu_data_ptr());
                    const void *logits_device = logits_base ? logits_base + logits_offset : nullptr;
                    if (!logits_device ||
                        !prefix_backend->deviceToHostOnStream(
                            handle.terminal_logits,
                            logits_device,
                            logits_bytes,
                            state_.device_id.gpu_ordinal(),
                            stream) ||
                        (device_hot_handle.valid() &&
                         (!device_hot_handle.deviceTerminalLogitsData() ||
                          !prefix_backend->deviceCopyAsync(
                              device_hot_handle.deviceTerminalLogitsData(),
                              logits_device,
                              logits_bytes,
                              state_.device_id.gpu_ordinal(),
                              stream))))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to submit prefix terminal logits D2H");
                        ok = false;
                    }
                    handle.has_terminal_logits = ok;
                    device_hot_handle.has_terminal_logits =
                        device_hot_handle.valid() && ok;
                }
                else
                {
                    const auto *logits_base = static_cast<const uint8_t *>(
                        terminal_logits_tensor->raw_data());
                    const void *logits = logits_base ? logits_base + logits_offset : nullptr;
                    if (ok && logits)
                    {
                        std::memcpy(
                            handle.terminal_logits,
                            logits,
                            logits_bytes);
                        handle.has_terminal_logits = true;
                    }
                    else if (ok)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: terminal logits host pointer unavailable");
                        ok = false;
                    }
                }
            }

            if (ok && terminal_block && prefix_layout_.includes_terminal_hidden &&
                state_.prefix_terminal_hidden &&
                state_.mtp_terminal_hidden_publication.current() &&
                handle.terminal_hidden)
            {
                if (state_.device_id.is_gpu())
                {
                    const void *hidden_device =
                        state_.prefix_terminal_hidden->gpu_data_ptr();
                    if (!hidden_device ||
                        !prefix_backend->deviceToHostOnStream(
                            handle.terminal_hidden,
                            hidden_device,
                            prefix_layout_.terminal_hidden_bytes,
                            state_.device_id.gpu_ordinal(),
                            stream) ||
                        (device_hot_handle.valid() &&
                         (!device_hot_handle.deviceTerminalHiddenData() ||
                          !prefix_backend->deviceCopyAsync(
                              device_hot_handle.deviceTerminalHiddenData(),
                              hidden_device,
                              prefix_layout_.terminal_hidden_bytes,
                              state_.device_id.gpu_ordinal(),
                              stream))))
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Failed to submit prefix terminal hidden D2H");
                        ok = false;
                    }
                    handle.has_terminal_hidden = ok;
                    device_hot_handle.has_terminal_hidden =
                        device_hot_handle.valid() && ok;
                }
                else
                {
                    const void *hidden =
                        ok ? state_.prefix_terminal_hidden->raw_data() : nullptr;
                    if (ok && hidden)
                    {
                        std::memcpy(
                            handle.terminal_hidden,
                            hidden,
                            prefix_layout_.terminal_hidden_bytes);
                        handle.has_terminal_hidden = true;
                    }
                    else if (ok)
                    {
                        LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: terminal hidden host pointer unavailable");
                        ok = false;
                    }
                }
            }

            if (terminal_hidden_archival_refresh &&
                terminal_block &&
                prefix_layout_.includes_terminal_hidden &&
                handle.has_terminal_hidden)
            {
                /*
                 * `PREFIX_TERMINAL_HIDDEN` now has a durable copy in the
                 * prefix-cache handle.  If harvest was the operation that made
                 * the sidecar input look current, clear only the live-current
                 * bit.  The buffer contents remain valid for the archive, but
                 * MTP decode must not skip its own terminal-hidden refresh
                 * because a cache-miss observer happened to run first.
                 */
                state_.mtp_terminal_hidden_publication.invalidate();
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "harvest_terminal_hidden_live_current_clears",
                    1.0,
                    "harvest",
                    state_.device_id.toString(),
                    {{"prompt_token_count", std::to_string(prompt_token_count)}});
            }

            ram_completion.commit();
            hot_completion.commit();

            if (!ok ||
                !prefix_cache_->insert(
                    handle,
                    device_hot_handle))
            {
                if (ok)
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Prefix harvest failed: cache insert rejected key="
                              << key.toHex());
                }
                if (prefixCacheTraceEnabled())
                {
                    LOG_INFO("[PREFIX_TRACE] harvest failed device=" << state_.device_id.toString()
                                                                     << " block=" << block
                                                                     << " key=" << key.toHex()
                                                                     << " ok_before_insert="
                                                                     << (ok ? "yes" : "no")
                                                                     << " terminal_block="
                                                                     << (terminal_block ? "yes" : "no")
                                                                     << " terminal_logits="
                                                                     << (handle.has_terminal_logits ? "yes" : "no")
                                                                     << " terminal_hidden="
                                                                     << (handle.has_terminal_hidden ? "yes" : "no")
                                                                     << " mtp_state="
                                                                     << (handle.layout.includes_mtp_state ? "yes" : "no")
                                                                     << " model_runtime_state="
                                                                     << (handle.has_model_runtime_state ? "yes" : "no")
                                                                     << " model_runtime_state_bytes="
                                                                     << (handle.model_runtime_state_storage
                                                                             ? handle.model_runtime_state_storage->size()
                                                                             : 0u)
                                                                     << " total_bytes=" << handle.total_bytes);
                }
                if (device_hot_handle.valid() &&
                    prefix_device_hot_backend_)
                {
                    prefix_device_hot_backend_->release(device_hot_handle);
                }
                prefix_ram_backend_->release(handle);
                return false;
            }

            if (device_hot_handle.valid())
            {
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "harvest_device_hot_d2d_publications",
                    1.0,
                    {static_cast<uint64_t>(block)},
                    "harvest",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(device_hot_handle.total_bytes)}});
            }

            PerfStatsCollector::addCounterWithSequence(
                "prefix_cache",
                "harvest_inserts",
                1.0,
                {static_cast<uint64_t>(block), static_cast<uint64_t>(key.token_start)},
                "harvest",
                state_.device_id.toString(),
                {{"token_count", std::to_string(key.token_count)},
                 {"terminal_block", terminal_block ? "true" : "false"},
                 {"terminal_logits", handle.has_terminal_logits ? "true" : "false"},
                 {"terminal_hidden", handle.has_terminal_hidden ? "true" : "false"},
                 {"includes_mtp_state", handle.layout.includes_mtp_state ? "true" : "false"},
                 {"model_runtime_state", handle.has_model_runtime_state ? "true" : "false"},
                 {"total_bytes", std::to_string(handle.total_bytes)}});

            if (prefixCacheTraceEnabled())
            {
                LOG_INFO("[PREFIX_TRACE] harvest insert device=" << state_.device_id.toString()
                                                                 << " block=" << block
                                                                 << " key=" << key.toHex()
                                                                 << " token_count=" << key.token_count
                                                                 << " terminal_logits="
                                                                 << (handle.has_terminal_logits ? "yes" : "no")
                                                                 << " terminal_hidden="
                                                                 << (handle.has_terminal_hidden ? "yes" : "no")
                                                                 << " mtp_state="
                                                                 << (handle.layout.includes_mtp_state ? "yes" : "no")
                                                                 << " model_runtime_state="
                                                                 << (handle.has_model_runtime_state ? "yes" : "no")
                                                                 << " model_runtime_state_bytes="
                                                                 << (handle.model_runtime_state_storage
                                                                         ? handle.model_runtime_state_storage->size()
                                                                         : 0u)
                                                                 << " total_bytes=" << handle.total_bytes);
            }
        }

        return true;
    }

    PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        PrefixStateSnapshot snapshot;
        bool submitted_async_device_payload = false;
        auto fail = [&](const std::string &reason) -> PrefixStateSnapshot
        {
            if (state_.device_id.is_gpu() && submitted_async_device_payload)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Full snapshot failed after asynchronous payload submission: " << reason);
                std::terminate();
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_payload_capture_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"reason", reason},
                 {"seq_idx", std::to_string(seq_idx)}});
            LOG_WARN("[DeviceGraphOrchestrator] captureLivePrefixState failed on "
                     << state_.device_id.toString()
                     << " seq_idx=" << seq_idx
                     << " reason=" << reason);
            return {};
        };

        if (!state_.kv_cache || seq_idx < 0 || seq_idx >= state_.batch_size)
        {
            return fail("missing_kv_cache_or_sequence_index_out_of_range");
        }

        void *stream = explicitGPUStreamForOperation("captureLivePrefixState");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing_explicit_gpu_stream");
        }
        if (!waitForLiveInferenceStateReadyForObservation(
                stream,
                "capture_live_prefix_state",
                DeviceTimelineRole::PrefixCheckpointArchive))
        {
            return fail("live_state_observation_wait_failed");
        }

        if (state_.device_id.is_gpu()) retirePendingPrefixPayloadUses(false);

        // Full snapshots are diagnostic archive transactions, unlike small
        // logical rollback. Borrow exactly the compact auxiliary section from
        // the canonical RAM tier and retain its physical owner in the snapshot.
        auto allocate_auxiliary_archive = [&](const IKVCache &cache, PrefixBlockHandle &handle) -> bool
        {
            applyPrefixAuxiliaryLayout(handle.layout, cache, nullptr, seq_idx,
                handle.key.token_start, handle.key.token_count,
                handle.key.token_start + handle.key.token_count, true);
            handle.total_bytes = handle.layout.totalBytes();
            if (handle.layout.modelAuxiliaryBytes() == 0) return true;
            if (!prefix_ram_backend_)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Full model auxiliary snapshot requires an admitted prefix RAM archive authority");
                return false;
            }
            PrefixPayloadLayout section;
            section.fa_layers = handle.layout.fa_layers;
            section.main_auxiliary = handle.layout.main_auxiliary;
            std::string error;
            auto archive = prefix_ram_backend_->allocateWithDiagnostics(handle.key, section, &error);
            if (!archive.valid())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Full model auxiliary snapshot archive rejected: " << error);
                return false;
            }
            // Retire the transient index entry immediately; the retained
            // section owner keeps PMA bytes/pinned arena ranges unavailable
            // until all snapshot aliases have released them.
            if (!prefix_ram_backend_->release(archive))
                throw std::logic_error("full auxiliary archive lost its canonical RAM allocation identity");
            handle.payload_readiness = std::move(archive.payload_readiness);
            handle.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::ModelAuxiliary)] =
                std::move(archive.ram_section_memory_leases[static_cast<size_t>(PrefixPayloadSection::ModelAuxiliary)]);
            handle.model_auxiliary_storage = std::move(archive.model_auxiliary_storage);
            handle.pinned_model_auxiliary_storage = std::move(archive.pinned_model_auxiliary_storage);
            handle.model_auxiliary_payload = archive.model_auxiliary_payload;
            return true;
        };

        // Auxiliary-only shifted caches have no recurrent slot event. Diagnostic
        // full archives prepare that event before copies through the same archive
        // authority; ordinary retained logical checkpoints remain unchanged.
        auto prepare_handle = [&](PrefixBlockHandle &handle, const char *producer) {
            if (!state_.device_id.is_gpu() || handle.layout.modelAuxiliaryBytes() == 0)
                return true;
            auto *backend = getBackendFor(state_.device_id);
            return backend && preparePrefixArchiveCheckpointReadiness(
                handle, state_.device_id, stream, *backend, [&] {
                    return preparePrefixArchivePayloadReady(&handle, stream, producer);
                });
        };

        auto complete_handle = [&](PrefixBlockHandle &handle, PrefixArchiveCompletion) {
            if (!state_.device_id.is_gpu())
                return true;
            const auto event = handle.live_checkpoint_ready_event;
            if (!event)
            {
                // Existing synchronous KV-only exports have no queued auxiliary/GDN payload.
                return handle.layout.modelAuxiliaryBytes() == 0 &&
                    !handle.device_hybrid_allocation && !handle.device_terminal_hidden_allocation;
            }
            if (handle.payload_readiness && handle.payload_readiness->prepared())
            {
                if (!recordPrefixArchivePayloadReady(
                        &handle, stream, "capture_live_prefix_state_section"))
                    return false;
            }
            else
            {
                auto *backend = getBackendFor(state_.device_id);
                if (!backend || !backend->recordEvent(
                        event.get(), state_.device_id.ordinal, stream))
                    return false;
            }
            // Every completed handle remains event-owned if a later handle
            // throws before aggregate snapshot publication. Reap through the
            // existing pending-use authority, never through a second ledger.
            PendingPrefixPayloadUse pending;
            pending.completion_event = event;
            pending.retained_payload_sources.push_back(
                PrefixPayloadReadLease::wholeArchive(handle));
            pending_prefix_payload_uses_.push_back(std::move(pending));
            return true;
        };

        int cached_tokens = -1;
        if (state_.device_id.is_gpu())
        {
            const std::optional<int> resident_tokens =
                deviceResidentLogicalTokenCountForObservation(
                    seq_idx,
                    stream,
                    "capture_live_prefix_state_logical_tokens");
            if (!resident_tokens)
            {
                return fail(
                    "canonical_device_cached_token_observation_failed");
            }
            cached_tokens = *resident_tokens;
        }
        else
        {
            cached_tokens =
                restorablePrefixCachedTokens(*state_.kv_cache, seq_idx);
        }
        if (cached_tokens < 0 || cached_tokens > state_.kv_cache->max_seq_len())
        {
            return fail("cached_token_count_out_of_range");
        }

        snapshot.valid = true;
        snapshot.provenance = PrefixStateProvenance::PayloadCheckpoint;
        snapshot.cached_tokens = cached_tokens;
        snapshot.mtp_shifted_origin=graph_builder_->config().mtpShiftedCacheOrigin();
        bool queued_async_device_checkpoint_payload = false;
        if (cached_tokens > 0)
        {
            size_t live_terminal_hidden_bytes = 0;
            if (graph_builder_ &&
                graph_builder_->config().mtp.enabled &&
                state_.prefix_terminal_hidden &&
                state_.mtp_terminal_hidden_publication.current() &&
                state_.d_model > 0)
            {
                live_terminal_hidden_bytes =
                    ResidualRowGeometry(graph_builder_->config().residualWidth()).rowBytes();
            }

            PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                cached_tokens,
                live_terminal_hidden_bytes);
            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_identity_.key != 0 ? prefix_identity_.key : 1;
            handle.key.block_index = 0;
            handle.key.token_start = 0;
            handle.key.token_count = cached_tokens;
            handle.layout = layout;
            applyLocalHybridCheckpointExtent(handle.layout, *state_.kv_cache);
            if (state_.device_id.is_gpu() &&
                handle.layout.hybrid_device_state_bytes > 0)
            {
                handle.layout.hybrid_host_state_bytes = 0;
                handle.layout.hybrid_state_bytes =
                    handle.layout.hybrid_device_state_bytes;
            }
            handle.total_bytes = handle.layout.totalBytes();
            if (!allocate_auxiliary_archive(*state_.kv_cache, handle))
                return fail("main_auxiliary_archive_authority_unavailable");

            const size_t kv_bytes = layout.faKVBytes();
            if (kv_bytes == 0)
            {
                return fail("main_kv_layout_empty");
            }
            handle.kv_storage = std::make_shared<std::vector<uint8_t>>(kv_bytes, 0);
            handle.kv_payload = handle.kv_storage->data();
            if (!ensureLiveCheckpointStorage(handle))
            {
                return fail("main_checkpoint_storage_unavailable");
            }

            if (!prepare_handle(handle, "capture_live_prefix_state_main"))
                return fail("full_auxiliary_archive_requires_exact_ready_authority");
            PrefixArchivePublicationScope main_completion([&](PrefixArchiveCompletion reason) {
                return complete_handle(handle, reason);
            });

            for (int local_layer = 0; local_layer < layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*state_.kv_cache, local_layer);
                if (global_layer < 0)
                {
                    return fail("main_kv_global_layer_unavailable");
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = 0;
                desc.token_count = cached_tokens;
                desc.stream = stream;
                if (!state_.kv_cache->exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return fail(
                        "main_kv_export_failed_layer_" +
                        std::to_string(global_layer) +
                        "_tokens_" + std::to_string(cached_tokens));
                }
            }

            if (!exportHybridPrefixPayload(
                    *state_.kv_cache,
                    seq_idx,
                    cached_tokens,
                    &handle,
                    /*synchronize=*/true,
                    stream, HybridPrefixStatePurpose::LocalCheckpoint))
            {
                return fail("main_hybrid_payload_export_failed");
            }

            if (live_terminal_hidden_bytes > 0)
            {
                if (state_.device_id.is_gpu())
                {
                    IBackend *backend = getBackendFor(state_.device_id);
                    const void *source =
                        state_.prefix_terminal_hidden
                            ? static_cast<const std::byte *>(state_.prefix_terminal_hidden->gpu_data_ptr()) +
                                  ResidualRowGeometry(graph_builder_->config().residualWidth()).elementOffset(
                                      state_.prefix_terminal_hidden->shape(), seq_idx) * sizeof(float)
                            : nullptr;
                    void *destination =
                        handle.deviceTerminalHiddenData();
                    submitted_async_device_payload = true;
                    if (!backend ||
                        !source ||
                        !destination ||
                        !backend->deviceCopyAsync(
                            destination,
                            source,
                            live_terminal_hidden_bytes,
                            state_.device_id.gpu_ordinal(),
                            stream))
                    {
                        return fail(
                            "terminal_hidden_device_copy_failed");
                    }
                    queued_async_device_checkpoint_payload = true;
                }
                else
                {
                    const void *source =
                        static_cast<const std::byte *>(state_.prefix_terminal_hidden->raw_data()) +
                        ResidualRowGeometry(graph_builder_->config().residualWidth()).elementOffset(
                            state_.prefix_terminal_hidden->shape(), seq_idx) * sizeof(float);
                    if (!source || !handle.terminal_hidden)
                    {
                        return fail(
                            "terminal_hidden_host_storage_unavailable");
                    }
                    std::memcpy(
                        handle.terminal_hidden,
                        source,
                        live_terminal_hidden_bytes);
                }
                handle.has_terminal_hidden = true;
            }

            submitted_async_device_payload |= state_.device_id.is_gpu() && handle.layout.modelAuxiliaryBytes() > 0;
            exportPrefixAuxiliaryState(*state_.kv_cache, seq_idx, PrefixAuxiliaryLane::Main, handle, stream);
            queued_async_device_checkpoint_payload |= state_.device_id.is_gpu() && handle.layout.modelAuxiliaryBytes() > 0;
            main_completion.commit();
            snapshot.blocks.push_back(std::move(handle));
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
            {
                continue;
            }

            int mtp_cached_tokens = restorablePrefixCachedTokens(*cache, seq_idx);
            if (auto resident_mtp_tokens =
                    deviceResidentShiftedMTPKVTokenCountForObservation(
                        static_cast<int>(depth),
                        seq_idx,
                        stream,
                        "capture_live_prefix_state_shifted_mtp"))
            {
                mtp_cached_tokens = *resident_mtp_tokens;
            }
            if (mtp_cached_tokens < 0 || mtp_cached_tokens > cache->max_seq_len())
            {
                return fail(
                    "mtp_cached_token_count_out_of_range_depth_" +
                    std::to_string(depth));
            }
            if (mtp_cached_tokens == 0)
            {
                continue;
            }

            PrefixPayloadLayout layout = buildDensePrefixPayloadLayout(
                *cache,
                state_.device_id,
                mtp_cached_tokens);

            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_identity_.key != 0 ? prefix_identity_.key : 1;
            handle.key.block_index = static_cast<int>(depth);
            handle.key.token_start = 0;
            handle.key.token_count = mtp_cached_tokens;
            handle.layout = layout;
            handle.total_bytes = layout.totalBytes();
            if (!allocate_auxiliary_archive(*cache, handle))
                return fail("shifted_auxiliary_archive_authority_unavailable_depth_" + std::to_string(depth));

            const size_t kv_bytes = layout.faKVBytes();
            if (kv_bytes == 0)
            {
                return fail(
                    "mtp_kv_layout_empty_depth_" +
                    std::to_string(depth));
            }
            handle.kv_storage = std::make_shared<std::vector<uint8_t>>(kv_bytes, 0);
            handle.kv_payload = handle.kv_storage->data();

            if (!ensureLiveCheckpointStorage(handle))
                return fail("shifted_checkpoint_storage_unavailable_depth_" + std::to_string(depth));
            if (!prepare_handle(handle, "capture_live_prefix_state_shifted"))
                return fail("shifted_auxiliary_archive_requires_exact_ready_authority");
            PrefixArchivePublicationScope mtp_completion([&](PrefixArchiveCompletion reason) {
                return complete_handle(handle, reason);
            });

            for (int local_layer = 0; local_layer < layout.fa_layers; ++local_layer)
            {
                const int global_layer = prefixFALayerForIndex(*cache, local_layer);
                if (global_layer < 0)
                {
                    return fail(
                        "mtp_kv_global_layer_unavailable_depth_" +
                        std::to_string(depth));
                }
                uint8_t *k_dst = handle.kvKData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_k;
                uint8_t *v_dst = handle.kvVData() +
                                 static_cast<size_t>(local_layer) * layout.bytes_per_fa_layer_v;

                IKVCache::KVCacheLogicalBlockDescriptor desc;
                desc.layer = global_layer;
                desc.seq_idx = seq_idx;
                desc.logical_token_start = 0;
                desc.token_count = mtp_cached_tokens;
                desc.stream = stream;
                if (!cache->exportLogicalBlock(desc, k_dst, v_dst))
                {
                    return fail(
                        "mtp_kv_export_failed_depth_" +
                        std::to_string(depth) +
                        "_layer_" + std::to_string(global_layer) +
                        "_tokens_" + std::to_string(mtp_cached_tokens));
                }
            }

            submitted_async_device_payload |= state_.device_id.is_gpu() && handle.layout.modelAuxiliaryBytes() > 0;
            exportPrefixAuxiliaryState(*cache, seq_idx, PrefixAuxiliaryLane::Main, handle, stream);
            queued_async_device_checkpoint_payload |= state_.device_id.is_gpu() && handle.layout.modelAuxiliaryBytes() > 0;
            mtp_completion.commit();
            snapshot.mtp_blocks.push_back(std::move(handle));
        }

        if (queued_async_device_checkpoint_payload &&
            !recordLivePrefixCheckpointReady(
                &snapshot,
                stream,
                "capture_live_prefix_state"))
        {
            return fail("checkpoint_ready_event_record_failed");
        }
        return snapshot;
    }

    bool DeviceGraphOrchestrator::ensureLiveCheckpointStorage(PrefixBlockHandle &handle) const
    {
        return acquireLiveCheckpointStorage(handle);
    }

    bool DeviceGraphOrchestrator::claimPhysicalAllocation(
        DeviceId device,
        PhysicalMemoryOwner owner,
        size_t bytes,
        const char *allocation_name,
        std::shared_ptr<void> *lease) const
    {
        if (!lease)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Missing lease destination for "
                      << (allocation_name ? allocation_name : "unnamed allocation"));
            return false;
        }
        lease->reset();
        if (bytes == 0)
            return true;

        if (!physical_memory_authority_)
        {
            if (memory_authority_mode_ == MemoryAuthorityMode::IsolatedTest)
                return true;

            LOG_ERROR("[DeviceGraphOrchestrator] Production allocation has no physical-memory authority"
                      << " allocation="
                      << (allocation_name ? allocation_name : "unnamed")
                      << " owner=" << toString(owner)
                      << " device=" << device.toString()
                      << " bytes=" << bytes);
            return false;
        }

        try
        {
            auto typed_lease =
                physical_memory_authority_->claimNewAllocation(
                    device,
                    owner,
                    bytes);
            if (!typed_lease.valid())
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Physical-memory authority returned an invalid lease"
                          << " allocation="
                          << (allocation_name ? allocation_name : "unnamed")
                          << " owner=" << toString(owner)
                          << " device=" << device.toString()
                          << " bytes=" << bytes);
                return false;
            }
            *lease = std::make_shared<PhysicalMemoryAllocationLease>(
                std::move(typed_lease));
            return true;
        }
        catch (const std::exception &error)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Physical-memory allocation claim failed"
                      << " allocation="
                      << (allocation_name ? allocation_name : "unnamed")
                      << " owner=" << toString(owner)
                      << " device=" << device.toString()
                      << " bytes=" << bytes
                      << " error=" << error.what());
            return false;
        }
    }

    bool DeviceGraphOrchestrator::initializeLiveCheckpointStoragePool()
    {
        // Checkpoint roots belong to the retained model envelope, including
        // no-launch verifier capture for an execution-Off request.
        if (!graph_builder_ || !retainsMTPGraphCapacity(graph_builder_->config().mtp))
            return true;
        if (!state_.kv_cache || state_.d_model <= 0)
            return false;

        live_checkpoint_storage_pool_initialized_ = false;
        live_checkpoint_storage_pool_.clear();
        host_auxiliary_checkpoint_pool_ = HostCacheAuxiliaryCheckpointPool{};
        live_device_sequence_state_storage_pool_.clear();
        mtp_publication_main_kv_base_checkpoints_.clear();
        mtp_publication_main_kv_base_checkpoints_ready_ = false;
        mtp_publication_main_kv_base_checkpoint_request_count_ = 0;

        std::vector<PrefixPayloadLayout> payload_layouts_per_checkpoint;

        PrefixPayloadLayout main_layout = buildDensePrefixPayloadLayout(
            *state_.kv_cache,
            state_.device_id,
            /*cached_tokens=*/1);
        main_layout = liveHybridCheckpointLayout(
            main_layout,
            /*device_only=*/state_.device_id.is_gpu(), *state_.kv_cache);
        main_layout.terminal_hidden_bytes =
            ResidualRowGeometry(graph_builder_->config().residualWidth()).rowBytes();
        main_layout.includes_terminal_hidden = true;
        payload_layouts_per_checkpoint.push_back(main_layout);

        for (const auto &cache : state_.mtp_kv_caches)
        {
            if (!cache)
                continue;

            const PrefixPayloadLayout shifted_layout =
                liveHybridCheckpointLayout(
                    buildDensePrefixPayloadLayout(
                        *cache,
                        state_.device_id,
                        /*block_size=*/1),
                    /*device_only=*/state_.device_id.is_gpu(), *cache);
            /*
             * Shifted full-attention caches archive sequence metadata in the
             * dedicated metadata pool below, but captureLivePrefixCheckpoint()
             * never asks them for a PrefixBlockHandle. Reserve a payload slot
             * only when the shifted cache owns recurrent state that the live
             * checkpoint transaction will actually export.
             */
            if (shifted_layout.includes_hybrid_state &&
                shifted_layout.hybrid_state_bytes > 0)
            {
                payload_layouts_per_checkpoint.push_back(shifted_layout);
            }
        }
        const size_t total_storage_slots =
            kMTPConcurrentLiveCheckpointSets *
            payload_layouts_per_checkpoint.size();

        /*
         * Keep every acquired handle alive until all four logical checkpoint
         * sets are reserved. Shared ownership makes acquisition skip earlier
         * slots, so this creates exact main-plus-hybrid payload sets without a
         * mutable reservation flag or any hot-path allocation branch.
         */
        std::vector<PrefixBlockHandle> reservations;
        reservations.reserve(total_storage_slots);
        for (size_t checkpoint_index = 0;
             checkpoint_index < kMTPConcurrentLiveCheckpointSets;
             ++checkpoint_index)
        {
            for (const auto &layout : payload_layouts_per_checkpoint)
            {
                PrefixBlockHandle handle;
                handle.layout = layout;
                handle.total_bytes = layout.totalBytes();
                if (!acquireLiveCheckpointStorage(handle))
                    return false;
                reservations.push_back(std::move(handle));
            }
        }

        /*
         * Sequence metadata is required for every GPU cache, even when a cache
         * has no recurrent payload and therefore consumes no PrefixBlockHandle
         * during a live checkpoint. Reserve complete main-plus-shifted sets so
         * capture never allocates or creates events in decode.
         */
        std::vector<DeviceKVSequenceStateCheckpoint>
            sequence_state_reservations;
        if (state_.device_id.is_gpu())
        {
            sequence_state_reservations.reserve(total_storage_slots);
            for (size_t checkpoint_index = 0;
                 checkpoint_index < kMTPConcurrentLiveCheckpointSets;
                 ++checkpoint_index)
            {
                DeviceKVSequenceStateCheckpoint checkpoint;
                if (!acquireLiveDeviceSequenceStateStorage(
                        /*cache_depth=*/-1,
                        /*sequence_index=*/0,
                        *state_.kv_cache,
                        &checkpoint))
                {
                    return false;
                }
                sequence_state_reservations.push_back(
                    std::move(checkpoint));

                for (size_t depth = 0;
                     depth < state_.mtp_kv_caches.size();
                     ++depth)
                {
                    const auto &cache = state_.mtp_kv_caches[depth];
                    if (!cache)
                        continue;
                    DeviceKVSequenceStateCheckpoint shifted_checkpoint;
                    if (!acquireLiveDeviceSequenceStateStorage(
                            static_cast<int>(depth),
                            /*sequence_index=*/0,
                            *cache,
                            &shifted_checkpoint))
                    {
                        return false;
                    }
                    sequence_state_reservations.push_back(
                        std::move(shifted_checkpoint));
                }
            }

            /*
             * Accepted verifier publication has a distinct lifetime from a
             * scheduler rollback checkpoint. Reserve one immutable metadata
             * base per logical request now, rather than borrowing a pool slot
             * whose shared ownership could be released while a verifier graph
             * still consumes it. These tiny allocations live for the runner's
             * lifetime and are overwritten in stream order before each
             * verifier launch.
             */
            const size_t publication_checkpoint_bytes =
                state_.kv_cache->deviceSequenceStateCheckpointBytes();
            if (publication_checkpoint_bytes == 0 ||
                publication_checkpoint_bytes %
                        (sizeof(int32_t) * 2u) !=
                    0)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Main KV cache does not expose a valid verifier-publication checkpoint layout");
                return false;
            }
            mtp_publication_main_kv_base_checkpoints_.reserve(
                static_cast<size_t>(state_.batch_size));
            for (int sequence_index = 0;
                 sequence_index < state_.batch_size;
                 ++sequence_index)
            {
                std::shared_ptr<void> memory_lease;
                if (!claimPhysicalAllocation(
                        state_.device_id,
                        PhysicalMemoryOwner::SequenceMetadata,
                        publication_checkpoint_bytes,
                        "mtp verifier-publication sequence metadata",
                        &memory_lease))
                {
                    return false;
                }
                DeviceKVSequenceStateCheckpoint checkpoint{
                    .cache_depth = -1,
                    .sequence_index = sequence_index,
                    .metadata_layer_count = static_cast<int>(
                        state_.kv_cache->deviceSequenceStateMetadataBytes() /
                        (sizeof(int32_t) * 2u)),
                    .bytes = publication_checkpoint_bytes,
                    .device = state_.device_id,
                    .physical_memory_lease = std::move(memory_lease),
                    .storage = allocatePureDeviceStorage(
                        publication_checkpoint_bytes,
                        state_.device_id),
                    .ready_event =
                        allocatePersistentDeviceEvent(state_.device_id),
                };
                if (!checkpoint.valid())
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Failed to preallocate main-KV verifier-publication checkpoint for sequence "
                              << sequence_index);
                    return false;
                }
                mtp_publication_main_kv_base_checkpoints_.push_back(
                    std::move(checkpoint));
            }
        }
        if (!state_.device_id.is_gpu())
        {
            // CPU accepted tails have one authority too. Reserve the same
            // concurrent checkpoint sets as recurrent state; compressed pools
            // remain append-only and are hidden by restored logical frontiers.
            auto reserve_auxiliary = [&](const IKVCache &cache) -> bool
            {
                const auto *owner = cache.auxiliaryState();
                if (!owner) return true;
                size_t bytes = 0;
                for (int sequence = 0; sequence < state_.batch_size; ++sequence)
                    bytes = std::max(bytes, owner->checkpointCapacity(sequence));
                if (bytes == 0) return true;
                for (size_t checkpoint = 0; checkpoint < kMTPConcurrentLiveCheckpointSets; ++checkpoint)
                {
                    std::shared_ptr<void> lease;
                    if (!claimPhysicalAllocation(state_.device_id,
                            PhysicalMemoryOwner::RecurrentCheckpointState, bytes,
                            "CPU accepted auxiliary rollback", &lease)) return false;
                    host_auxiliary_checkpoint_pool_.add(*owner, std::move(lease),
                        std::make_shared<std::vector<uint8_t>>(bytes));
                }
                return true;
            };
            if (!reserve_auxiliary(*state_.kv_cache)) return false;
            for (const auto &cache : state_.mtp_kv_caches)
                if (cache && !reserve_auxiliary(*cache)) return false;
        }
        host_auxiliary_checkpoint_pool_.seal();
        live_checkpoint_storage_pool_initialized_ = true;

        size_t checkpoint_hybrid_host_bytes = 0;
        size_t checkpoint_hybrid_device_bytes = 0;
        size_t checkpoint_terminal_hidden_bytes = 0;
        for (const auto &layout : payload_layouts_per_checkpoint)
        {
            checkpoint_hybrid_host_bytes +=
                layout.hybrid_host_state_bytes;
            checkpoint_hybrid_device_bytes +=
                layout.hybrid_device_state_bytes;
            checkpoint_terminal_hidden_bytes +=
                layout.terminal_hidden_bytes;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "live_checkpoint_storage_slots_preallocated",
            static_cast<double>(total_storage_slots),
            "setup",
            state_.device_id.toString(),
            {{"concurrent_checkpoint_count",
              std::to_string(kMTPConcurrentLiveCheckpointSets)},
             {"handles_per_checkpoint",
              std::to_string(payload_layouts_per_checkpoint.size())},
             {"terminal_hidden_bytes",
              std::to_string(checkpoint_terminal_hidden_bytes)},
             {"hybrid_host_bytes",
              std::to_string(checkpoint_hybrid_host_bytes)},
             {"hybrid_device_bytes",
              std::to_string(checkpoint_hybrid_device_bytes)},
             {"device_sequence_state_slots",
              std::to_string(
                  live_device_sequence_state_storage_pool_.size())}});
        return true;
    }

    bool DeviceGraphOrchestrator::acquireLiveDeviceSequenceStateStorage(
        int cache_depth,
        int sequence_index,
        const IKVCache &cache,
        DeviceKVSequenceStateCheckpoint *checkpoint) const
    {
        if (!checkpoint ||
            !state_.device_id.is_gpu() ||
            sequence_index < 0 ||
            cache_depth < -1)
        {
            return false;
        }

        const size_t required_bytes =
            cache.deviceSequenceStateCheckpointBytes();
        if (required_bytes == 0 ||
            required_bytes % (sizeof(int32_t) * 2u) != 0)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] GPU KV cache does not expose a valid device sequence-state checkpoint layout"
                      << " cache_depth=" << cache_depth
                      << " bytes=" << required_bytes);
            return false;
        }

        auto assign_slot =
            [&](LiveDeviceSequenceStateStorageSlot &slot)
        {
            *checkpoint = DeviceKVSequenceStateCheckpoint{
                .cache_depth = cache_depth,
                .sequence_index = sequence_index,
                .metadata_layer_count = static_cast<int>(
                    cache.deviceSequenceStateMetadataBytes() / (sizeof(int32_t) * 2u)),
                .bytes = required_bytes,
                .device = state_.device_id,
                .physical_memory_lease = slot.physical_memory_lease,
                .storage = slot.storage,
                .ready_event = slot.ready_event,
            };
        };

        for (auto &slot : live_device_sequence_state_storage_pool_)
        {
            const bool free =
                slot.storage &&
                slot.storage.use_count() == 1 &&
                slot.ready_event &&
                slot.ready_event.use_count() == 1;
            if (!free ||
                slot.device != state_.device_id ||
                slot.capacity_bytes < required_bytes)
            {
                continue;
            }

            assign_slot(slot);
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_checkpoint_device_sequence_state_pool_hits",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"cache_depth", std::to_string(cache_depth)},
                 {"bytes", std::to_string(required_bytes)}});
            return true;
        }

        if (live_checkpoint_storage_pool_initialized_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Device sequence-state checkpoint pool exhausted after initialization"
                      << " cache_depth=" << cache_depth
                      << " bytes=" << required_bytes);
            return false;
        }

        LiveDeviceSequenceStateStorageSlot slot;
        slot.capacity_bytes = required_bytes;
        slot.device = state_.device_id;
        if (!claimPhysicalAllocation(
                state_.device_id,
                PhysicalMemoryOwner::SequenceMetadata,
                required_bytes,
                "live MTP sequence-state checkpoint slot",
                &slot.physical_memory_lease))
        {
            return false;
        }
        slot.storage =
            allocatePureDeviceStorage(required_bytes, state_.device_id);
        slot.ready_event =
            allocatePersistentDeviceEvent(state_.device_id);
        if (!slot.storage || !slot.ready_event)
        {
            return false;
        }
        live_device_sequence_state_storage_pool_.push_back(std::move(slot));
        assign_slot(live_device_sequence_state_storage_pool_.back());
        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_checkpoint_device_sequence_state_pool_misses",
            1.0,
            "setup",
            state_.device_id.toString(),
            {{"cache_depth", std::to_string(cache_depth)},
             {"bytes", std::to_string(required_bytes)},
             {"pool_size",
              std::to_string(
                  live_device_sequence_state_storage_pool_.size())}});
        return true;
    }

    bool DeviceGraphOrchestrator::acquireLiveCheckpointStorage(PrefixBlockHandle &handle) const
    {
        const size_t hybrid_host_bytes =
            handle.layout.includes_hybrid_state
                ? handle.layout.hybrid_host_state_bytes
                : 0;
        const size_t hybrid_device_bytes =
            handle.layout.includes_hybrid_state
                ? handle.layout.hybrid_device_state_bytes
                : 0;
        const size_t terminal_hidden_bytes =
            handle.layout.includes_terminal_hidden
                ? handle.layout.terminal_hidden_bytes
                : 0;
        const bool gpu_checkpoint = state_.device_id.is_gpu();
        const bool needs_hybrid_host = hybrid_host_bytes > 0;
        const bool needs_hybrid_device = hybrid_device_bytes > 0;
        const bool needs_terminal_hidden_host =
            !gpu_checkpoint && terminal_hidden_bytes > 0;
        const bool needs_terminal_hidden_device =
            gpu_checkpoint && terminal_hidden_bytes > 0;
        if (!needs_hybrid_host &&
            !needs_hybrid_device &&
            !needs_terminal_hidden_host &&
            !needs_terminal_hidden_device)
        {
            return true;
        }

        auto assign_slot = [&](LiveCheckpointStorageSlot &slot)
        {
            handle.recurrent_checkpoint_memory_lease =
                slot.physical_memory_lease;
            handle.live_checkpoint_ready_event =
                state_.device_id.is_gpu() ? slot.ready_event : nullptr;
            if (needs_hybrid_host)
            {
                slot.hybrid_host_storage->resize(hybrid_host_bytes);
                handle.hybrid_storage = slot.hybrid_host_storage;
                handle.hybrid_payload = slot.hybrid_host_storage->data();
            }
            else
            {
                handle.hybrid_storage.reset();
                handle.hybrid_payload = nullptr;
            }

            if (needs_hybrid_device)
            {
                handle.device_hybrid_storage.reset();
                handle.device_hybrid_allocation =
                    slot.hybrid_device_storage;
            }
            else
            {
                handle.device_hybrid_storage.reset();
                handle.device_hybrid_allocation.reset();
            }

            if (needs_terminal_hidden_host)
            {
                slot.terminal_hidden_host_storage->resize(terminal_hidden_bytes);
                handle.terminal_hidden_storage =
                    slot.terminal_hidden_host_storage;
                handle.terminal_hidden =
                    slot.terminal_hidden_host_storage->data();
            }
            else
            {
                handle.terminal_hidden_storage.reset();
                handle.terminal_hidden = nullptr;
            }

            if (needs_terminal_hidden_device)
            {
                handle.device_terminal_hidden_storage.reset();
                handle.device_terminal_hidden_allocation =
                    slot.terminal_hidden_device_storage;
            }
            else
            {
                handle.device_terminal_hidden_storage.reset();
                handle.device_terminal_hidden_allocation.reset();
            }
        };

        for (auto &slot : live_checkpoint_storage_pool_)
        {
            const bool slot_free =
                (!slot.hybrid_host_storage ||
                 slot.hybrid_host_storage.use_count() == 1) &&
                (!slot.hybrid_device_storage ||
                 slot.hybrid_device_storage.use_count() == 1) &&
                (!slot.terminal_hidden_host_storage ||
                 slot.terminal_hidden_host_storage.use_count() == 1) &&
                (!slot.terminal_hidden_device_storage ||
                 slot.terminal_hidden_device_storage.use_count() == 1) &&
                (!slot.ready_event ||
                 slot.ready_event.use_count() == 1);
            const bool hybrid_host_ok =
                !needs_hybrid_host ||
                (slot.hybrid_host_storage &&
                 slot.hybrid_host_capacity_bytes >= hybrid_host_bytes);
            const bool hybrid_device_ok =
                !needs_hybrid_device ||
                (slot.hybrid_device_storage &&
                 slot.hybrid_device_capacity_bytes >= hybrid_device_bytes &&
                 slot.device == state_.device_id);
            const bool terminal_hidden_host_ok =
                !needs_terminal_hidden_host ||
                (slot.terminal_hidden_host_storage &&
                 slot.terminal_hidden_host_capacity_bytes >=
                     terminal_hidden_bytes);
            const bool terminal_hidden_device_ok =
                !needs_terminal_hidden_device ||
                (slot.terminal_hidden_device_storage &&
                 slot.terminal_hidden_device_capacity_bytes >=
                     terminal_hidden_bytes &&
                 slot.device == state_.device_id);
            if (!slot_free ||
                !hybrid_host_ok ||
                !hybrid_device_ok ||
                !terminal_hidden_host_ok ||
                !terminal_hidden_device_ok)
            {
                continue;
            }

            assign_slot(slot);
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_checkpoint_storage_pool_hits",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"hybrid_host_bytes", std::to_string(hybrid_host_bytes)},
                 {"hybrid_device_bytes", std::to_string(hybrid_device_bytes)},
                 {"terminal_hidden_bytes", std::to_string(terminal_hidden_bytes)}});
            return true;
        }

        if (live_checkpoint_storage_pool_initialized_)
        {
            LOG_ERROR("[DeviceGraphOrchestrator] Transient MTP checkpoint storage pool exhausted after initialization");
            return false;
        }

        LiveCheckpointStorageSlot slot;
        slot.device = state_.device_id;
        size_t checkpoint_allocation_bytes = hybrid_host_bytes;
        auto add_checkpoint_bytes = [&](size_t bytes) -> bool
        {
            if (bytes > std::numeric_limits<size_t>::max() -
                            checkpoint_allocation_bytes)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Live MTP checkpoint slot size overflow");
                return false;
            }
            checkpoint_allocation_bytes += bytes;
            return true;
        };
        if (!add_checkpoint_bytes(hybrid_device_bytes) ||
            !add_checkpoint_bytes(
                needs_terminal_hidden_host ? terminal_hidden_bytes : 0u) ||
            !add_checkpoint_bytes(
                needs_terminal_hidden_device ? terminal_hidden_bytes : 0u) ||
            !claimPhysicalAllocation(
                state_.device_id,
                PhysicalMemoryOwner::RecurrentCheckpointState,
                checkpoint_allocation_bytes,
                "live MTP recurrent checkpoint payload slot",
                &slot.physical_memory_lease))
        {
            return false;
        }
        if (state_.device_id.is_gpu())
        {
            slot.ready_event =
                allocatePersistentDeviceEvent(state_.device_id);
            if (!slot.ready_event)
                return false;
        }

        if (needs_hybrid_host)
        {
            slot.hybrid_host_storage =
                std::make_shared<std::vector<uint8_t>>(hybrid_host_bytes);
            slot.hybrid_host_capacity_bytes =
                slot.hybrid_host_storage->capacity();
        }

        if (needs_hybrid_device)
        {
            slot.hybrid_device_storage =
                allocatePureDeviceStorage(
                    hybrid_device_bytes,
                    state_.device_id);
            if (!slot.hybrid_device_storage)
            {
                return false;
            }
            slot.hybrid_device_capacity_bytes = hybrid_device_bytes;
        }

        if (needs_terminal_hidden_host)
        {
            slot.terminal_hidden_host_storage =
                std::make_shared<std::vector<uint8_t>>(terminal_hidden_bytes);
            slot.terminal_hidden_host_capacity_bytes =
                slot.terminal_hidden_host_storage->capacity();
        }

        if (needs_terminal_hidden_device)
        {
            slot.terminal_hidden_device_storage =
                allocatePureDeviceStorage(
                    terminal_hidden_bytes,
                    state_.device_id);
            if (!slot.terminal_hidden_device_storage)
            {
                return false;
            }
            slot.terminal_hidden_device_capacity_bytes =
                terminal_hidden_bytes;
        }

        live_checkpoint_storage_pool_.push_back(std::move(slot));
        assign_slot(live_checkpoint_storage_pool_.back());
        PerfStatsCollector::addCounter(
            "mtp",
            "live_prefix_checkpoint_storage_pool_misses",
            1.0,
            "decode",
            state_.device_id.toString(),
            {{"hybrid_host_bytes", std::to_string(hybrid_host_bytes)},
             {"hybrid_device_bytes", std::to_string(hybrid_device_bytes)},
             {"terminal_hidden_bytes", std::to_string(terminal_hidden_bytes)},
             {"pool_size", std::to_string(live_checkpoint_storage_pool_.size())}});
        return true;
    }

    PrefixStateSnapshot DeviceGraphOrchestrator::captureLivePrefixCheckpoint(
        const PrefixCheckpointCaptureRequest &request) const
    {
        const int seq_idx = request.sequence_index;
        PrefixStateSnapshot snapshot;
        bool submitted_async_device_work = false;
        auto fail = [&](const std::string &reason) -> PrefixStateSnapshot
        {
            if (state_.device_id.is_gpu() &&
                submitted_async_device_work)
            {
                LOG_ERROR("[DeviceGraphOrchestrator] Live prefix checkpoint failed after submitting device work"
                          << " device=" << state_.device_id.toString()
                          << " seq_idx=" << seq_idx
                          << " reason=" << reason);
                /*
                 * A failed transaction has no completion event to retain its
                 * pooled destination. Returning would permit immediate slot
                 * reuse while a capture kernel or D2D copy still references
                 * that storage. This is an unrecoverable ownership violation.
                 */
                std::terminate();
            }
            /*
             * This path is the first safety gate for MTP transactions: before a
             * sidecar mutates speculative KV/GDN state, the runner captures a
             * rollback checkpoint for the live serial-decode state.  Returning
             * an empty snapshot is correct when the state is not safely
             * restorable, but the caller only sees a generic failure.  Keep a
             * structured reason here so multi-device tests can identify which
             * child and which checkpoint invariant failed.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "live_prefix_checkpoint_capture_failures",
                1.0,
                "decode",
                state_.device_id.toString(),
                {{"reason", reason},
                 {"seq_idx", std::to_string(seq_idx)}});
            LOG_WARN("[DeviceGraphOrchestrator] captureLivePrefixCheckpoint failed on "
                     << state_.device_id.toString()
                     << " seq_idx=" << seq_idx
                     << " reason=" << reason);
            return {};
        };

        if (!request.valid())
        {
            return fail("invalid_scheduler_checkpoint_cursor");
        }
        if (!state_.kv_cache || seq_idx >= state_.batch_size)
        {
            return fail("missing_kv_cache_or_sequence_index_out_of_range");
        }

        void *stream = explicitGPUStreamForOperation("captureLivePrefixCheckpoint");
        if (state_.device_id.is_gpu() && !stream)
        {
            return fail("missing_explicit_gpu_stream");
        }
        if (!waitForLiveInferenceStateReadyForObservation(
                stream,
                "capture_live_prefix_checkpoint",
                DeviceTimelineRole::PrefixCheckpointArchive))
        {
            return fail("live_state_observation_wait_failed");
        }
        if (!orderCheckpointMetadataPoolReuse(stream))
            return fail("checkpoint_metadata_pool_reuse_ordering_failed");

        /*
         * This cursor belongs to the request scheduler, which advances it at
         * every successful decode transaction boundary.  GPU capture uses it
         * only to size the checkpoint and annotate its logical identity; exact
         * canonical cache heads/counts are archived below as opaque device
         * metadata and restored without a D2H observation.
         */
        const int cached_tokens = request.logical_cached_tokens;
        if (!state_.device_id.is_gpu())
        {
            const int canonical_cached_tokens =
                restorablePrefixCachedTokens(*state_.kv_cache, seq_idx);
            if (canonical_cached_tokens != cached_tokens)
            {
                return fail(
                    "cpu_scheduler_cursor_disagrees_with_canonical_cache");
            }
        }
        if (cached_tokens < 0 || cached_tokens > state_.kv_cache->max_seq_len())
        {
            return fail("cached_token_count_out_of_range");
        }

        /* The scheduler seals the append extent before this checkpoint. Graph
         * workspace remains sized for the retained family, but it is not live
         * KV demand. In particular, a budget-one condition advances one row
         * even when dynamic depth can later select fifteen drafts. */
        const int main_headroom = request.maximum_main_append_tokens;
        if (liveCheckpointLacksHeadroom(*state_.kv_cache, cached_tokens, main_headroom))
        {
            return fail(
                "main_cache_lacks_speculative_headroom_cached_" +
                std::to_string(cached_tokens) +
                "_required_" + std::to_string(main_headroom) +
                "_capacity_" + std::to_string(state_.kv_cache->max_seq_len()));
        }

        /*
         * Plan every shifted-cache checkpoint before submitting the first
         * asynchronous copy. The former implementation discovered shifted GDN
         * state after copying the main cache, then abandoned the logical
         * checkpoint for a host payload snapshot. Besides making ordinary GPU
         * MTP host-owned, that could release a pool slot while its D2D copy was
         * still in flight. A complete preflight makes the transaction shape
         * immutable before any producer work reaches the stream.
         */
        struct ShiftedCheckpointPlan
        {
            int cached_tokens = -1;
            PrefixPayloadLayout hybrid_layout;
            bool captures_hybrid_state = false;
        };
        std::vector<ShiftedCheckpointPlan> shifted_checkpoint_plans(
            state_.mtp_kv_caches.size());
        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            if (!cache)
                continue;

            const int mtp_cached_tokens =
                state_.device_id.is_gpu()
                    ? std::max(
                          0,
                          cached_tokens -
                              static_cast<int>(depth) - 1)
                    : restorablePrefixCachedTokens(*cache, seq_idx);
            if (mtp_cached_tokens < 0 ||
                mtp_cached_tokens > cache->max_seq_len())
            {
                return fail(
                    "mtp_cached_token_count_out_of_range_depth_" +
                    std::to_string(depth));
            }
            if (liveCheckpointLacksHeadroom(
                    *cache,
                    mtp_cached_tokens,
                    request.maximum_shifted_append_tokens))
            {
                return fail(
                    "mtp_cache_lacks_speculative_headroom_depth_" +
                    std::to_string(depth) +
                    "_cached_" + std::to_string(mtp_cached_tokens) +
                    "_required_" +
                    std::to_string(request.maximum_shifted_append_tokens) +
                    "_capacity_" + std::to_string(cache->max_seq_len()));
            }

            ShiftedCheckpointPlan &plan = shifted_checkpoint_plans[depth];
            plan.cached_tokens = mtp_cached_tokens;
            if (mtp_cached_tokens > 0)
            {
                plan.hybrid_layout = liveHybridCheckpointLayout(
                    buildDensePrefixPayloadLayout(
                        *cache,
                        state_.device_id,
                        mtp_cached_tokens),
                    /*device_only=*/state_.device_id.is_gpu(), *cache);
                plan.captures_hybrid_state =
                    plan.hybrid_layout.includes_hybrid_state &&
                    plan.hybrid_layout.hybrid_state_bytes > 0;
            }
        }

        snapshot.valid = true;
        snapshot.logical_checkpoint = true;
        snapshot.provenance = PrefixStateProvenance::LogicalCheckpoint;
        snapshot.cached_tokens = cached_tokens;
        snapshot.mtp_shifted_origin=graph_builder_->config().mtpShiftedCacheOrigin();
        snapshot.mtp_cached_tokens.assign(state_.mtp_kv_caches.size(), -1);
        for (size_t depth = 0; depth < shifted_checkpoint_plans.size(); ++depth)
        {
            snapshot.mtp_cached_tokens[depth] =
                shifted_checkpoint_plans[depth].cached_tokens;
        }
        bool queued_async_device_checkpoint_payload = false;

        if (state_.device_id.is_gpu())
        {
            auto capture_sequence_state =
                [&](int cache_depth,
                    const IKVCache &cache) -> bool
            {
                DeviceKVSequenceStateCheckpoint checkpoint;
                if (!acquireLiveDeviceSequenceStateStorage(
                        cache_depth,
                        seq_idx,
                        cache,
                        &checkpoint))
                {
                    return false;
                }

                std::string checkpoint_error;
                /*
                 * Treat the call as submitted before invoking the backend.
                 * A backend may discover a launch error only after partially
                 * accepting work; failure must therefore retain no pretense
                 * that the pooled destination can safely be reused.
                 */
                submitted_async_device_work = true;
                if (!cache.captureDeviceSequenceStateCheckpoint(
                        seq_idx,
                        checkpoint.data(),
                        checkpoint.bytes,
                        stream,
                        &checkpoint_error))
                {
                    LOG_ERROR("[DeviceGraphOrchestrator] Device sequence-state checkpoint capture failed"
                              << " cache_depth=" << cache_depth
                              << " error=" << checkpoint_error);
                    return false;
                }

                PerfStatsCollector::addCounter(
                    "mtp",
                    "live_prefix_checkpoint_device_sequence_state_captures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"cache",
                      cache_depth < 0 ? "main" : "shifted"},
                     {"depth", std::to_string(cache_depth)},
                     {"bytes", std::to_string(checkpoint.bytes)},
                     {"implementation", "device_kernel"}});
                snapshot.device_sequence_state_checkpoints.push_back(
                    std::move(checkpoint));
                return true;
            };

            if (!capture_sequence_state(-1, *state_.kv_cache))
            {
                return fail("main_device_sequence_state_capture_failed");
            }
            for (size_t depth = 0;
                 depth < state_.mtp_kv_caches.size();
                 ++depth)
            {
                const auto &cache = state_.mtp_kv_caches[depth];
                if (!cache)
                    continue;
                if (!capture_sequence_state(
                        static_cast<int>(depth),
                        *cache))
                {
                    return fail(
                        "shifted_device_sequence_state_capture_failed_depth_" +
                        std::to_string(depth));
                }
            }
            queued_async_device_checkpoint_payload = true;
        }

        if (!state_.device_id.is_gpu())
        {
            auto capture_auxiliary = [&](int depth, const IKVCache &cache)
            {
                if (const auto *owner = cache.auxiliaryState();
                    owner && owner->checkpointCapacity(seq_idx) > 0)
                    snapshot.host_auxiliary_checkpoints.push_back(
                        host_auxiliary_checkpoint_pool_.capture(depth, seq_idx, *owner));
            };
            capture_auxiliary(-1, *state_.kv_cache);
            for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
                if (state_.mtp_kv_caches[depth])
                    capture_auxiliary(static_cast<int>(depth), *state_.mtp_kv_caches[depth]);
        }

        PrefixPayloadLayout main_layout;
        if (cached_tokens > 0)
        {
            PerfStatsCollector::ScopedTimer timer("mtp",
                                                  "live_prefix_checkpoint_layout",
                                                  "decode",
                                                  state_.device_id.toString(),
                                                  {{"cache", "main"}});
            main_layout = buildDensePrefixPayloadLayout(
                *state_.kv_cache,
                state_.device_id,
                cached_tokens);
        }

        size_t live_terminal_hidden_bytes = 0;
        const float *live_terminal_hidden_source = nullptr;
        const bool mtp_terminal_hidden_checkpoint_enabled =
            graph_builder_ &&
            graph_builder_->config().mtp.enabled &&
            cached_tokens > 0 &&
            state_.d_model > 0;
        if (graph_builder_ &&
            graph_builder_->config().mtp.enabled &&
            state_.prefix_terminal_hidden &&
            state_.mtp_terminal_hidden_publication.current() &&
            cached_tokens > 0 &&
            state_.d_model > 0)
        {
            live_terminal_hidden_bytes =
                ResidualRowGeometry(graph_builder_->config().residualWidth()).rowBytes();
        }
        else if (mtp_terminal_hidden_checkpoint_enabled &&
                 !state_.device_id.is_gpu() &&
                 state_.last_forward_batch_size == 1 &&
                 state_.last_forward_seq_len > 0 &&
                 state_.hidden)
        {
            /*
             * CPU logical checkpoints do not need to materialize the compact
             * PREFIX_TERMINAL_HIDDEN tensor just to publish accepted MTP state.
             * If the main forward produced a multi-row hidden tensor, copy the
             * terminal row directly from HIDDEN_STATE.  GPU keeps using the
             * explicit row-select path so stream ownership remains visible.
             */
            const auto &hidden_shape = state_.hidden->shape();
            const size_t hidden_rows = hidden_shape.empty() ? 0 : hidden_shape[0];
            const size_t terminal_row =
                static_cast<size_t>(std::max(0, state_.last_forward_seq_len - 1));
            const float *hidden_rows_fp32 = state_.hidden->fp32_data();
            if (hidden_rows_fp32 &&
                hidden_rows > terminal_row &&
                hidden_shape.size() >= 2 &&
                hidden_shape[1] == static_cast<size_t>(graph_builder_->config().residualWidth()))
            {
                live_terminal_hidden_source =
                    hidden_rows_fp32 + ResidualRowGeometry(graph_builder_->config().residualWidth()).elementOffset(hidden_shape, terminal_row);
                live_terminal_hidden_bytes =
                    ResidualRowGeometry(graph_builder_->config().residualWidth()).rowBytes();
            }
        }

        if ((main_layout.includes_hybrid_state && main_layout.hybrid_state_bytes > 0) ||
            live_terminal_hidden_bytes > 0)
        {
            PrefixBlockHandle handle;
            handle.key.fingerprint = prefix_identity_.key != 0 ? prefix_identity_.key : 1;
            handle.key.block_index = 0;
            handle.key.token_start = 0;
            handle.key.token_count = cached_tokens;
            const bool device_only_checkpoint = state_.device_id.is_gpu();
            handle.layout = liveHybridCheckpointLayout(
                main_layout,
                device_only_checkpoint, *state_.kv_cache);
            if (live_terminal_hidden_bytes > 0)
            {
                handle.layout.terminal_hidden_bytes = live_terminal_hidden_bytes;
                handle.layout.includes_terminal_hidden = true;
            }
            handle.total_bytes = handle.layout.totalBytes();
            PerfStatsCollector::addCounter("mtp",
                                           "live_prefix_checkpoint_hybrid_host_bytes",
                                           static_cast<double>(handle.layout.hybrid_host_state_bytes),
                                           "decode",
                                           state_.device_id.toString());
            PerfStatsCollector::addCounter("mtp",
                                           "live_prefix_checkpoint_hybrid_device_bytes",
                                           static_cast<double>(handle.layout.hybrid_device_state_bytes),
                                           "decode",
                                           state_.device_id.toString());
            if (device_only_checkpoint)
            {
                PerfStatsCollector::addCounter("mtp",
                                               "live_prefix_checkpoint_hybrid_device_only_captures",
                                               1.0,
                                               "decode",
                                               state_.device_id.toString());
            }
            {
                PerfStatsCollector::ScopedTimer timer("mtp",
                                                      "live_prefix_checkpoint_hybrid_storage",
                                                      "decode",
                                                      state_.device_id.toString());
                if (!ensureLiveCheckpointStorage(handle))
                    return fail("checkpoint_storage_unavailable");
            }
            if (state_.device_id.is_gpu() &&
                !handle.live_checkpoint_ready_event)
            {
                return fail(
                    "checkpoint_storage_missing_preallocated_ready_event");
            }
            if (handle.layout.includes_hybrid_state && handle.layout.hybrid_state_bytes > 0)
            {
                PerfStatsCollector::ScopedTimer timer("mtp",
                                                      "live_prefix_checkpoint_hybrid_export",
                                                      "decode",
                                                      state_.device_id.toString());
                if (!exportHybridPrefixPayload(
                        *state_.kv_cache,
                        seq_idx,
                        cached_tokens,
                        &handle,
                        /*synchronize=*/false,
                        stream, HybridPrefixStatePurpose::LocalCheckpoint))
                {
                    return fail("hybrid_checkpoint_export_failed");
                }
                if (!handle.has_hybrid_state)
                {
                    return fail("hybrid_checkpoint_export_missing_payload");
                }
                queued_async_device_checkpoint_payload =
                    state_.device_id.is_gpu() &&
                    handle.layout.hybrid_device_state_bytes > 0;
            }
            if (live_terminal_hidden_bytes > 0)
            {
                if (state_.device_id.is_gpu())
                {
                    IBackend *backend = getBackendFor(state_.device_id);
                    const void *hidden_source =
                        state_.prefix_terminal_hidden
                            ? static_cast<const std::byte *>(state_.prefix_terminal_hidden->gpu_data_ptr()) +
                                  ResidualRowGeometry(graph_builder_->config().residualWidth()).elementOffset(
                                      state_.prefix_terminal_hidden->shape(), seq_idx) * sizeof(float)
                            : nullptr;
                    void *hidden_destination =
                        handle.deviceTerminalHiddenData();
                    if (!backend ||
                        !hidden_source ||
                        !hidden_destination ||
                        !backend->deviceCopyAsync(
                            hidden_destination,
                            hidden_source,
                            live_terminal_hidden_bytes,
                            state_.device_id.gpu_ordinal(),
                            stream))
                    {
                        return fail(
                            "terminal_hidden_checkpoint_device_copy_failed");
                    }
                    queued_async_device_checkpoint_payload = true;
                }
                else
                {
                    const void *hidden_source =
                        live_terminal_hidden_source
                            ? static_cast<const void *>(
                                  live_terminal_hidden_source)
                            : static_cast<const std::byte *>(state_.prefix_terminal_hidden->raw_data()) +
                        ResidualRowGeometry(graph_builder_->config().residualWidth()).elementOffset(
                            state_.prefix_terminal_hidden->shape(), seq_idx) * sizeof(float);
                    if (!hidden_source || !handle.terminal_hidden)
                    {
                        return fail(
                            "terminal_hidden_checkpoint_host_storage_unavailable");
                    }
                    std::memcpy(
                        handle.terminal_hidden,
                        hidden_source,
                        live_terminal_hidden_bytes);
                }
                handle.has_terminal_hidden = true;
                PerfStatsCollector::addCounter(
                    "mtp",
                    "live_prefix_checkpoint_terminal_hidden_captures",
                    1.0,
                    "decode",
                    state_.device_id.toString(),
                    {{"bytes", std::to_string(live_terminal_hidden_bytes)},
                     {"implementation",
                      state_.device_id.is_gpu()
                          ? "device_to_device"
                          : "host_copy"}});
            }
            snapshot.blocks.push_back(std::move(handle));
            if (main_layout.includes_hybrid_state && main_layout.hybrid_state_bytes > 0)
            {
                PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_hybrid_state_captures", 1.0, "decode",
                                               state_.device_id.toString());
            }
        }

        for (size_t depth = 0; depth < state_.mtp_kv_caches.size(); ++depth)
        {
            const auto &cache = state_.mtp_kv_caches[depth];
            const ShiftedCheckpointPlan &plan =
                shifted_checkpoint_plans[depth];
            if (!cache || !plan.captures_hybrid_state)
                continue;

            PrefixBlockHandle handle;
            handle.key.fingerprint =
                prefix_identity_.key != 0 ? prefix_identity_.key : 1;
            handle.key.block_index = static_cast<int>(depth);
            handle.key.token_start = 0;
            handle.key.token_count = plan.cached_tokens;
            handle.layout = plan.hybrid_layout;
            handle.total_bytes = handle.layout.totalBytes();

            if (!ensureLiveCheckpointStorage(handle))
            {
                return fail(
                    "shifted_hybrid_checkpoint_storage_unavailable_depth_" +
                    std::to_string(depth));
            }
            if (state_.device_id.is_gpu() &&
                !handle.live_checkpoint_ready_event)
            {
                return fail(
                    "shifted_hybrid_checkpoint_missing_preallocated_ready_event_depth_" +
                    std::to_string(depth));
            }
            if (!exportHybridPrefixPayload(
                    *cache,
                    seq_idx,
                    plan.cached_tokens,
                    &handle,
                    /*synchronize=*/false,
                    stream, HybridPrefixStatePurpose::LocalCheckpoint))
            {
                return fail(
                    "shifted_hybrid_checkpoint_export_failed_depth_" +
                    std::to_string(depth));
            }
            if (!handle.has_hybrid_state)
            {
                return fail(
                    "shifted_hybrid_checkpoint_payload_missing_depth_" +
                    std::to_string(depth));
            }

            queued_async_device_checkpoint_payload =
                queued_async_device_checkpoint_payload ||
                (state_.device_id.is_gpu() &&
                 handle.layout.hybrid_device_state_bytes > 0);
            snapshot.mtp_blocks.push_back(std::move(handle));
            PerfStatsCollector::addCounterWithSequence(
                "mtp",
                "live_prefix_checkpoint_shifted_hybrid_state_captures",
                1.0,
                {static_cast<uint64_t>(plan.cached_tokens)},
                "decode",
                state_.device_id.toString(),
                {{"depth", std::to_string(depth)},
                 {"implementation",
                  state_.device_id.is_gpu()
                      ? "device_to_device"
                      : "host_copy"}});
        }

        if (queued_async_device_checkpoint_payload &&
            !recordLivePrefixCheckpointReady(
                &snapshot,
                stream,
                "capture_live_prefix_checkpoint"))
        {
            return fail("checkpoint_ready_event_record_failed");
        }

        PerfStatsCollector::addCounter("mtp", "live_prefix_checkpoint_logical_captures", 1.0, "decode",
                                       state_.device_id.toString());
        return snapshot;
    }
} // namespace llaminar2
