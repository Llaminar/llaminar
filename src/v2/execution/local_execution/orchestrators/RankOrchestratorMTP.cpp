/**
 * @file RankOrchestratorMTP.cpp
 * @brief RankOrchestrator stochastic local-TP sampling and MTP speculative coordination.
 */
#include "RankOrchestratorInternal.h"

namespace llaminar2
{

    bool RankOrchestrator::buildRankStochasticTargetDistributionsFromLocalTPRows(
        DeviceLogitsSource source,
        int first_row,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (source != DeviceLogitsSource::AllPosition ||
            first_row < 0 ||
            first_slot < 0 ||
            row_count <= 0 ||
            first_slot + row_count > rank_stochastic_slot_capacity_ ||
            params.top_k <= 0 ||
            params.top_k > sampling_math::kMaxTopK ||
            vocab_size <= 0)
        {
            return false;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasAllPositionLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        auto build_row_from_consumed_infos =
            [&](int row, int slot) -> bool
        {
            struct Candidate
            {
                float value = -std::numeric_limits<float>::infinity();
                int token = -1;
            };

            std::vector<Candidate> candidates;
            candidates.reserve(
                local_infos.size() * static_cast<size_t>(params.top_k));
            size_t implicit_vocab_offset = 0;
            const bool use_explicit_vocab_offsets =
                std::any_of(
                    local_infos.begin(),
                    local_infos.end(),
                    [](const LogitsLocalInfo &info)
                    {
                        return info.vocab_offset != 0;
                    });

            for (const LogitsLocalInfo &info : local_infos)
            {
                const auto &shape = info.tensor->shape();
                const size_t rows = shape.size() >= 2 ? shape[0] : 1;
                const size_t cols = info.vocab_local;
                const size_t row_stride =
                    info.row_stride > 0 ? info.row_stride : info.vocab_local;
                if (cols == 0 ||
                    row_stride < cols ||
                    static_cast<size_t>(row) >= rows ||
                    cols > static_cast<size_t>(std::numeric_limits<int>::max()))
                {
                    return false;
                }

                const int local_k =
                    std::min(params.top_k, static_cast<int>(cols));
                std::vector<float> top_values(static_cast<size_t>(local_k));
                std::vector<int> top_indices(static_cast<size_t>(local_k), -1);
                int selected = 0;

                if (info.device.has_value() && info.device->is_gpu())
                {
                    if (!info.gpu_ptr || !info.stream)
                        return false;
                    IBackend *backend = resolveLogitsBackend(*info.device);
                    if (!backend)
                        return false;
                    const float *row_ptr =
                        static_cast<const float *>(info.gpu_ptr) +
                        static_cast<size_t>(row) * row_stride;
                    if (!backend->topKF32(
                            row_ptr,
                            static_cast<int>(cols),
                            local_k,
                            info.device->gpu_ordinal(),
                            top_values.data(),
                            top_indices.data(),
                            info.stream))
                    {
                        return false;
                    }
                    selected = local_k;
                }
                else
                {
                    const float *data = info.tensor->fp32_data();
                    if (!data)
                        return false;
                    const float *row_ptr =
                        data + static_cast<size_t>(row) * row_stride;
                    selected = cpu_sampling::select_topk(
                        row_ptr,
                        static_cast<int>(cols),
                        local_k,
                        top_values.data(),
                        top_indices.data());
                }

                const size_t vocab_offset =
                    use_explicit_vocab_offsets
                        ? info.vocab_offset
                        : implicit_vocab_offset;
                for (int i = 0; i < selected; ++i)
                {
                    const int local_index =
                        top_indices[static_cast<size_t>(i)];
                    if (local_index < 0)
                        continue;
                    const int token =
                        static_cast<int>(vocab_offset) + local_index;
                    if (token >= 0 && token < vocab_size)
                    {
                        candidates.push_back(
                            Candidate{
                                top_values[static_cast<size_t>(i)],
                                token});
                    }
                }
                implicit_vocab_offset += cols;
            }

            if (candidates.empty())
                return false;
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate &a, const Candidate &b)
                {
                    if (a.value != b.value)
                        return a.value > b.value;
                    return a.token < b.token;
                });

            const int selected =
                std::min(params.top_k, static_cast<int>(candidates.size()));
            std::vector<float> sorted_logits(static_cast<size_t>(selected));
            std::vector<int> sorted_ids(static_cast<size_t>(selected));
            std::vector<float> scratch(static_cast<size_t>(selected), 0.0f);
            std::vector<int> out_ids(static_cast<size_t>(selected), -1);
            std::vector<float> out_probs(static_cast<size_t>(selected), 0.0f);
            for (int i = 0; i < selected; ++i)
            {
                sorted_logits[static_cast<size_t>(i)] =
                    candidates[static_cast<size_t>(i)].value;
                sorted_ids[static_cast<size_t>(i)] =
                    candidates[static_cast<size_t>(i)].token;
            }
            sampling_math::build_topk_topp_distribution_from_sorted(
                sorted_logits.data(),
                sorted_ids.data(),
                selected,
                params.top_p,
                params.temperature,
                out_ids.data(),
                out_probs.data(),
                scratch.data());

            const size_t slot_offset =
                static_cast<size_t>(slot) *
                static_cast<size_t>(sampling_math::kMaxTopK);
            for (int i = 0; i < sampling_math::kMaxTopK; ++i)
            {
                rank_stochastic_target_token_ids_[slot_offset +
                                                  static_cast<size_t>(i)] = -1;
                rank_stochastic_target_probs_[slot_offset +
                                              static_cast<size_t>(i)] = 0.0f;
            }
            for (int i = 0; i < selected; ++i)
            {
                rank_stochastic_target_token_ids_[slot_offset +
                                                  static_cast<size_t>(i)] =
                    out_ids[static_cast<size_t>(i)];
                rank_stochastic_target_probs_[slot_offset +
                                              static_cast<size_t>(i)] =
                    out_probs[static_cast<size_t>(i)];
            }
            rank_stochastic_target_top_k_[static_cast<size_t>(slot)] =
                selected;
            rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] =
                -1;
            return true;
        };

        for (int row = 0; row < row_count; ++row)
        {
            if (!build_row_from_consumed_infos(
                    first_row + row,
                    first_slot + row))
            {
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_batch_rows",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"first_row", std::to_string(first_row)},
             {"first_slot", std::to_string(first_slot)},
             {"implementation", "cross_shard_topk_candidate_merge"}});
        return true;
    }

    bool RankOrchestrator::sampleRankStochasticDistribution(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold,
        int32_t *out_token)
    {
        if (buffer != DeviceDistributionBuffer::Target ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_)
        {
            return false;
        }

        const int active_top_k =
            rank_stochastic_target_top_k_[static_cast<size_t>(slot)];
        if (active_top_k <= 0 ||
            active_top_k > sampling_math::kMaxTopK)
        {
            return false;
        }

        const size_t slot_offset =
            static_cast<size_t>(slot) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        float selected_probability = 0.0f;
        const int token =
            sampling_math::sample_distribution_with_threshold_and_probability(
                rank_stochastic_target_token_ids_.data() + slot_offset,
                rank_stochastic_target_probs_.data() + slot_offset,
                active_top_k,
                threshold,
                &selected_probability);
        if (token < 0)
            return false;

        rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] =
            static_cast<int32_t>(token);
        if (!stageRankStochasticTargetTokenForLocalTP(
                slot,
                static_cast<int32_t>(token)))
        {
            return false;
        }
        if (out_token)
            *out_token = static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_samples",
            1.0,
            "decode",
            "rank",
            {{"slot", std::to_string(slot)},
             {"top_k", std::to_string(active_top_k)},
             {"token", std::to_string(token)},
             {"probability", std::to_string(selected_probability)}});
        return true;
    }

    bool RankOrchestrator::sampleRankGreedyMTPLogitsToLocalTPDraftSlot(
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (device_runners_.size() < 2 ||
            draft_sample_slot < 0 ||
            draft_sample_slot >= rank_stochastic_slot_capacity_)
        {
            return false;
        }

        /*
         * Each child owns only its local MTP-head shard.  The rank samples the
         * global greedy token by reducing child-local argmax metadata, then
         * stages that one compact token into every child draft slot.  The
         * verifier can therefore consume child device slots exactly as it does
         * for stochastic MTP, without gathering full logits or replaying rows.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, /*row=*/0);
        if (token < 0)
            return false;

        const int32_t staged_token = static_cast<int32_t>(token);
        if (!stageStochasticDraftTokensForDeviceVerification(
                &staged_token,
                /*draft_token_count=*/1,
                draft_sample_slot))
        {
            return false;
        }
        if (out_token)
            *out_token = staged_token;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_greedy_mtp_draft_slot_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(draft_sample_slot)},
             {"token", std::to_string(token)},
             {"implementation", "cross_shard_argmax_device_partial"}});
        return true;
    }

    bool RankOrchestrator::sampleRankGreedyMainLogitsToLocalTPTargetSlot(
        int target_sample_slot,
        int32_t *out_token)
    {
        if (device_runners_.size() < 2 ||
            target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            return false;
        }

        /*
         * LocalTP main logits are vocab-sharded: each child owns a compact row
         * for its LM-head slice.  DeviceSampler performs the economical
         * per-shard argmax and returns only the winning global token.  That small
         * rank decision is then staged into every child target slot so downstream
         * verifier graph input remains device-resident and participant-symmetric.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasLogitsLocal())
                return false;
            LogitsLocalInfo info =
                runner->consumeLogitsLocalInfoForSampling();
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, /*row=*/0);
        if (token < 0)
            return false;

        if (!stageRankStochasticTargetTokenForLocalTP(
                target_sample_slot,
                static_cast<int32_t>(token)))
        {
            return false;
        }

        const size_t slot_offset =
            static_cast<size_t>(target_sample_slot) *
            static_cast<size_t>(sampling_math::kMaxTopK);
        for (int i = 0; i < sampling_math::kMaxTopK; ++i)
        {
            rank_stochastic_target_token_ids_[slot_offset +
                                              static_cast<size_t>(i)] = -1;
            rank_stochastic_target_probs_[slot_offset +
                                          static_cast<size_t>(i)] = 0.0f;
        }
        rank_stochastic_target_token_ids_[slot_offset] = token;
        rank_stochastic_target_probs_[slot_offset] = 1.0f;
        rank_stochastic_target_top_k_[static_cast<size_t>(target_sample_slot)] = 1;
        rank_stochastic_target_sample_tokens_[
            static_cast<size_t>(target_sample_slot)] =
            static_cast<int32_t>(token);
        if (out_token)
            *out_token = static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_greedy_main_target_slot_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(target_sample_slot)},
             {"token", std::to_string(token)},
             {"implementation", "cross_shard_argmax_device_partial"}});
        return true;
    }

    bool RankOrchestrator::stageRankStochasticTargetTokenForLocalTP(
        int slot,
        int32_t token)
    {
        if (device_runners_.size() < 2 ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_ ||
            token < 0)
        {
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !device_runners_[i]->stageStochasticTargetTokenForDeviceSampling(
                    token,
                    slot))
            {
                LOG_ERROR("[RankOrchestrator] LocalTP child "
                          << i
                          << " could not stage rank stochastic target token slot="
                          << slot);
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_target_token_device_stages",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"token", std::to_string(token)}});
        return true;
    }

    bool RankOrchestrator::stageRankStochasticDraftTokensForLocalTP(
        const int32_t *tokens,
        int token_count,
        int first_slot)
    {
        if (device_runners_.size() < 2 ||
            !tokens ||
            token_count <= 0 ||
            first_slot < 0 ||
            first_slot + token_count > rank_stochastic_slot_capacity_)
        {
            return false;
        }

        for (int i = 0; i < token_count; ++i)
        {
            if (tokens[i] < 0)
                return false;
        }

        for (size_t child = 0; child < device_runners_.size(); ++child)
        {
            if (!device_runners_[child] ||
                !device_runners_[child]->stageStochasticDraftTokensForDeviceVerification(
                    tokens,
                    token_count,
                    first_slot))
            {
                LOG_ERROR("[RankOrchestrator] LocalTP child "
                          << child
                          << " could not stage rank stochastic draft tokens first_slot="
                          << first_slot
                          << " count=" << token_count);
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_token_device_stages",
            static_cast<double>(token_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"first_slot", std::to_string(first_slot)}});
        return true;
    }

    bool RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren(
        int slot)
    {
        if (device_runners_.size() < 2 ||
            !tp_ctx_ ||
            !publishesFullVocabularyMTPLogits() ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_)
        {
            return false;
        }

        std::vector<DeviceStochasticSampleSlotHandle> child_slots(
            device_runners_.size());
        for (size_t child = 0; child < device_runners_.size(); ++child)
        {
            IInferenceRunner *runner = device_runners_[child].get();
            if (!runner ||
                !runner->publishesFullVocabularyMTPLogits() ||
                !runner->supportsDeviceStochasticMTPVerification())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP "
                          "main-target slot broadcast requires every child to expose mirrored device stochastic support; participant "
                          << child << " is not ready");
                return false;
            }

            child_slots[child] =
                child == 0
                    ? runner->deviceStochasticTargetSampleProducerSlot(slot)
                    : runner
                          ->deviceStochasticTargetSampleBroadcastDestinationSlot(
                              slot);
            if (!child_slots[child].valid())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP "
                          "main-target slot broadcast could not acquire "
                          << (child == 0
                                  ? "ready primary producer"
                                  : "predecessor-ordered destination")
                          << " slot=" << slot
                          << " for participant " << child);
                return false;
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] mirrored sample slot broadcast failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             slot,
             &child_slots,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t child) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto device_id = device_runners_[child]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                LocalTPCollectiveSidebandBuffer sampled_token;
                sampled_token.kind = LocalTPCollectiveSidebandKind::Broadcast;
                sampled_token.send_buffer =
                    child == 0 ? child_slots[0].token_device : nullptr;
                sampled_token.recv_buffer = child_slots[child].token_device;
                sampled_token.element_count = 1;
                sampled_token.dtype = CollectiveDataType::INT32;
                sampled_token.root_device_index = 0;
                sampled_token.name = "mtp_mirrored_main_target_token";

                const bool broadcast_ok = tp_ctx_->collectiveSidebandOnStream(
                    {sampled_token},
                    static_cast<int>(child),
                    child_slots[child].stream,
                    "mtp_rank_mirrored_main_target_token_broadcast");
                if (!broadcast_ok)
                    return false;

                /*
                 * The collective write is now the producer for every participant's
                 * slot, including the root.  Record a fresh event after the
                 * broadcast so sidecars and verifier reducers wait on the
                 * common rank-owned token rather than any pre-broadcast
                 * child-local sampler event.
                 */
                return device_runners_[child]
                    ->recordStochasticTargetSampleSlotReadyFromDevice(
                        slot,
                        child_slots[child].stream,
                        /*verifier_consumer_pending=*/true);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_child = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: participant "
                          << r.worker_index << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_child = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: participant "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "broadcastPrimaryLocalTPMainTargetSlotToChildren",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::broadcastPrimaryLocalTPMainTargetSlotToChildren: rethrowing primary exception from participant "
                      << first_exception_child);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_main_target_slot_broadcasts",
                1.0,
                "decode",
                "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                 {"buffer", "target"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
        }
        return all_success;
    }

    const void *RankOrchestrator::prepareRankVerifierTokenSlotsForLocalTP(
        bool first_token_from_device,
        int32_t first_token,
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 != total_verifier_input_tokens ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count > rank_stochastic_slot_capacity_)
        {
            return nullptr;
        }
        if (first_token_from_device &&
            (first_target_sample_slot < 0 ||
             first_target_sample_slot >= rank_stochastic_slot_capacity_))
        {
            return nullptr;
        }
        if (!first_token_from_device && first_token < 0)
            return nullptr;

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ =
            total_verifier_input_tokens;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;

            rank_mtp_verifier_child_token_inputs_[i] =
                first_token_from_device
                    ? device_runners_[i]->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                          first_target_sample_slot,
                          first_draft_slot,
                          draft_token_count,
                          total_verifier_input_tokens)
                    : device_runners_[i]->prepareMTPVerifierInputTokensOnDevice(
                          first_token,
                          first_draft_slot,
                          draft_token_count,
                          total_verifier_input_tokens);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_rows_prepared_from_device_slots",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"tokens", std::to_string(total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(draft_token_count)},
             {"first_token_from_device",
              first_token_from_device ? "true" : "false"}});
        return rank_mtp_verifier_child_token_inputs_.data();
    }

    const void *RankOrchestrator::stageRankVerifierTokenRowForLocalTP(
        const int32_t *tokens,
        int total_verifier_input_tokens,
        int draft_token_count)
    {
        if (device_runners_.size() < 2 ||
            !tokens ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count >= total_verifier_input_tokens)
        {
            return nullptr;
        }

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ =
            total_verifier_input_tokens;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;
            rank_mtp_verifier_child_token_inputs_[i] =
                device_runners_[i]->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                    tokens,
                    total_verifier_input_tokens,
                    draft_token_count);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_rows_staged_from_compact_samples",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"tokens", std::to_string(total_verifier_input_tokens)},
             {"draft_tokens", std::to_string(draft_token_count)}});
        return rank_mtp_verifier_child_token_inputs_.data();
    }

    int32_t RankOrchestrator::rankStochasticDraftSampleToken(int slot) const
    {
        if (slot < 0 || slot >= rank_stochastic_slot_capacity_)
            return -1;
        return rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)];
    }

    int32_t RankOrchestrator::rankStochasticTargetSampleToken(int slot) const
    {
        if (slot < 0 || slot >= rank_stochastic_slot_capacity_)
            return -1;
        return rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)];
    }

    bool RankOrchestrator::supportsGreedyAllPositionBatchOutcomeOnDevice() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsGreedyAllPositionBatchOutcomeOnDevice();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsGreedyAllPositionBatchOutcomeOnDevice();
        }

        /*
         * GPU LocalTP does not advertise the rank-compact reducer as a production
         * resident verifier.  Every child must own a mirrored full-vocab verifier
         * head so it can reduce its own compact outcome and publish that exact
         * device handle later.
         */
        return supportsDeviceResidentMTPSpecStatePublication();
    }

    bool RankOrchestrator::publishesFullVocabularyMTPLogits() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishesFullVocabularyMTPLogits();
        }
        if (device_runners_.empty())
            return false;

        /*
         * The rank is only an aggregator here. Complete vocabulary publication
         * belongs to each child's graph, through a full head or an explicit
         * native gather. The aggregate requires every participant's publication.
         * Do not route through any rank-side compact reducer:
         * resident LocalTP MTP publication feeds each child its own device handle.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->publishesFullVocabularyMTPLogits();
            });
    }

    bool RankOrchestrator::supportsDeviceStochasticMTPVerification() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsDeviceStochasticMTPVerification();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsDeviceStochasticMTPVerification();
        }

        /*
         * Stochastic GPU LocalTP mirrors the full verifier head so each child runs
         * the native full-vocabulary resident stochastic reducer.  Rank-owned
         * sharded top-k tables are useful diagnostics, but they are not a
         * production publication path.
         */
        return device_runners_.size() >= 2 &&
               supportsDeviceResidentMTPSpecStatePublication() &&
               std::all_of(
                   device_runners_.begin(),
                   device_runners_.end(),
                   [](const std::unique_ptr<IInferenceRunner> &runner)
                   {
                       return runner &&
                              runner->supportsDeviceStochasticMTPVerification();
                   });
    }

    bool RankOrchestrator::buildStochasticDistributionOnDevice(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticDistributionOnDevice(
                source,
                row,
                buffer,
                slot,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticDistributionOnDevice(
                source,
                row,
                buffer,
                slot,
                params,
                vocab_size);
        }
        if (slot >= 0 && slot < rank_stochastic_slot_capacity_)
        {
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
        }
        if ((source == DeviceLogitsSource::Main ||
             source == DeviceLogitsSource::MainRequestBatch) &&
            publishesFullVocabularyMTPLogits())
        {
            if (buffer != DeviceDistributionBuffer::Target ||
                slot < 0 ||
                slot >= rank_stochastic_slot_capacity_ ||
                device_runners_.size() < 2)
            {
                return false;
            }
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                const IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->publishesFullVocabularyMTPLogits() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic main-target distribution requires full-head device support on participant "
                              << child);
                    return false;
                }
            }

            /*
             * Main-target sampling needs one distribution and one common token,
             * not one independently sampled token per mirrored participant.
             * Build on child zero now; sampleStochasticDistributionOnDevice*()
             * will publish that child's target slot through NCCL/RCCL.
             */
            if (!device_runners_.front()->buildStochasticDistributionOnDevice(
                    source,
                    row,
                    buffer,
                    slot,
                    params,
                    vocab_size))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic main-target distribution build failed on primary participant");
                return false;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = true;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source",
                  source == DeviceLogitsSource::MainRequestBatch
                      ? "main_request_batch"
                      : "main"},
                 {"rows", "1"},
                 {"implementation", "primary_child_full_vocab"}});
            return true;
        }
        if (source == DeviceLogitsSource::AllPosition &&
            publishesFullVocabularyMTPLogits())
        {
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                IInferenceRunner *child = device_runners_[i].get();
                if (!child ||
                    !child->supportsDeviceStochasticMTPVerification() ||
                    !child->buildStochasticDistributionOnDevice(
                        source,
                        row,
                        buffer,
                        slot,
                        params,
                        vocab_size))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic target distribution build failed on participant "
                              << i);
                    return false;
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source", "all_position"},
                 {"rows", "1"},
                 {"implementation", "mirrored_child_full_vocab"}});
            return true;
        }
        if (primaryDeviceId().is_gpu() &&
            source == DeviceLogitsSource::AllPosition)
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP stochastic verifier target distributions require mirrored full-head child reducers");
            return false;
        }
        return buildRankStochasticDistributionFromLocalTP(
            source,
            row,
            buffer,
            slot,
            params,
            vocab_size);
    }

    bool RankOrchestrator::buildStochasticDistributionsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticDistributionsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticDistributionsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (buffer != DeviceDistributionBuffer::Target)
            return false;
        if (source == DeviceLogitsSource::AllPosition &&
            publishesFullVocabularyMTPLogits())
        {
            for (size_t i = 0; i < device_runners_.size(); ++i)
            {
                IInferenceRunner *child = device_runners_[i].get();
                if (!child ||
                    !child->supportsDeviceStochasticMTPVerification() ||
                    !child->buildStochasticDistributionsOnDevice(
                        source,
                        first_row,
                        buffer,
                        first_slot,
                        row_count,
                        params,
                        vocab_size))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic target-row distribution build failed on participant "
                              << i);
                    return false;
                }
            }
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_distribution_builds",
                static_cast<double>(row_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"source", "all_position"},
                 {"rows", std::to_string(row_count)},
                 {"implementation", "mirrored_child_full_vocab"}});
            return true;
        }
        if (primaryDeviceId().is_gpu() &&
            source == DeviceLogitsSource::AllPosition)
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP stochastic verifier target-row distributions require mirrored full-head child reducers");
            return false;
        }
        return buildRankStochasticTargetDistributionsFromLocalTPRows(
            source,
            first_row,
            first_slot,
            row_count,
            params,
            vocab_size);
    }

    bool RankOrchestrator::
        buildCapturedStochasticVerifierTargetDistributions(
            int row_count,
            const SamplingParams &params,
            const MTPRequestPenaltyPolicy &penalty_policy,
            int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size);
        }
        if (!primaryDeviceId().is_gpu() ||
            !publishesFullVocabularyMTPLogits())
        {
            LOG_ERROR("[RankOrchestrator] Captured stochastic verifier target distributions require mirrored GPU verifier heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->buildCapturedStochasticVerifierTargetDistributions(
                    row_count,
                    params,
                    penalty_policy,
                    vocab_size))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored participant "
                          << participant
                          << " failed captured stochastic verifier target preparation");
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_captured_stochastic_verifier_target_rows",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"top_k", std::to_string(params.top_k)},
             {"implementation", "mirrored_monolithic_graph"}});
        return true;
    }

    bool RankOrchestrator::buildStochasticProcessedLogitRowsOnDevice(
        DeviceLogitsSource source,
        int first_row,
        DeviceDistributionBuffer buffer,
        int first_slot,
        int row_count,
        const SamplingParams &params,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->buildStochasticProcessedLogitRowsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->buildStochasticProcessedLogitRowsOnDevice(
                source,
                first_row,
                buffer,
                first_slot,
                row_count,
                params,
                vocab_size);
        }
        return false;
    }

    bool RankOrchestrator::publishCapturedMTPDraftToken(
        int row,
        int slot,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishCapturedMTPDraftToken(
                row,
                slot,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->publishCapturedMTPDraftToken(
                row,
                slot,
                penalty_policy);
        }
        if (!primaryDeviceId().is_gpu() ||
            !publishesFullVocabularyMTPLogits())
        {
            LOG_ERROR("[RankOrchestrator] Captured MTP draft publication requires mirrored full-vocabulary GPU MTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->publishCapturedMTPDraftToken(
                    row,
                    slot,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored participant "
                          << participant
                          << " failed captured MTP draft publication");
                return false;
            }
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_captured_mtp_draft_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"row", std::to_string(row)},
             {"slot", std::to_string(slot)},
             {"implementation", "mirrored_monolithic_graph"}});
        return true;
    }

    int RankOrchestrator::sampleStochasticDraftProposalOnDevice(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDraftProposalOnDevice(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDraftProposalOnDevice(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }

        if (source != DeviceLogitsSource::MTP ||
            row < 0 ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_ ||
            device_runners_.size() < 2)
        {
            return -1;
        }

        if (publishesFullVocabularyMTPLogits())
        {
            /*
             * This host-returning API is a response/diagnostic boundary, not
             * the production deferred lane. Child zero materializes its local
             * token for the caller while every peer performs the same sampling
             * operation without a host read. Each slot remains produced by its
             * own sampler stream; no H2D restaging or rank collective follows.
             */
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->publishesFullVocabularyMTPLogits() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic draft proposal requires every child to expose a mirrored full-head device sampler; participant "
                              << child << " is not ready");
                    return -1;
                }
            }

            int sampled_token = -1;
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                const bool ok =
                    child == 0
                        ? ((sampled_token = device_runners_[child]
                                                ->sampleStochasticDraftProposalOnDevice(
                                                    source,
                                                    row,
                                                    slot,
                                                    params,
                                                    vocab_size,
                                                    threshold)) >= 0)
                        : device_runners_[child]
                              ->sampleStochasticDraftProposalOnDeviceDeferred(
                                  source,
                                  row,
                                  slot,
                                  params,
                                  vocab_size,
                                  threshold);
                if (!ok)
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic draft proposal participant "
                              << child
                              << " failed local device sampling for slot="
                              << slot);
                    return -1;
                }
            }

            rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] =
                static_cast<int32_t>(sampled_token);
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_draft_proposals",
                1.0,
                "decode",
                 "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                  {"implementation", "participant_local_identical_samples"},
                  {"collective", "none"},
                  {"host_reads", "primary_response_only"}});
            return sampled_token;
        }

        (void)params;
        (void)vocab_size;
        (void)threshold;

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
                return -1;
            LogitsLocalInfo info =
                runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info)
                return -1;
            local_infos.push_back(info);
        }

        const int token =
            DeviceSampler::sampleGreedyFromLocalInfos(local_infos, row);
        if (token < 0)
            return -1;
        rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] =
            static_cast<int32_t>(token);

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_greedy_proposals",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"implementation", "cross_shard_argmax"}});
        return token;
    }

    bool RankOrchestrator::sampleStochasticDraftProposalOnDeviceDeferred(
        DeviceLogitsSource source,
        int row,
        int slot,
        const SamplingParams &params,
        int vocab_size,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDraftProposalOnDeviceDeferred(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDraftProposalOnDeviceDeferred(
                source,
                row,
                slot,
                params,
                vocab_size,
                threshold);
        }

        if (source == DeviceLogitsSource::MTP &&
            publishesFullVocabularyMTPLogits())
        {
            if (row < 0 ||
                slot < 0 ||
                slot >= rank_stochastic_slot_capacity_ ||
                device_runners_.size() < 2)
            {
                return false;
            }
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner ||
                    !runner->publishesFullVocabularyMTPLogits() ||
                    !runner->supportsDeviceStochasticMTPVerification())
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP deferred stochastic draft proposal requires every child to expose a mirrored full-head device sampler; participant "
                              << child << " is not ready");
                    return false;
                }
            }

            /*
             * Each child consumes its exact pending MTP-sidecar stream and
             * records its own sample-ready event. The argmax kernel is
             * deterministic over byte-identical mirrored logits, so this
             * locally complete publication is the production invariant. A
             * host comparison or corrective token broadcast would merely hide
             * a grouped-decode parity defect and is intentionally absent.
             */
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                if (!device_runners_[child]
                         ->sampleStochasticDraftProposalOnDeviceDeferred(
                             source,
                             row,
                             slot,
                             params,
                             vocab_size,
                             threshold))
                {
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP deferred stochastic draft proposal participant "
                              << child
                              << " failed local device proposal sampling for slot="
                              << slot);
                    return false;
                }
            }

            rank_stochastic_draft_sample_tokens_[static_cast<size_t>(slot)] = -1;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_deferred_draft_proposals",
                1.0,
                "decode",
                 "rank",
                 {{"participants", std::to_string(device_runners_.size())},
                  {"slot", std::to_string(slot)},
                  {"implementation", "participant_local_identical_samples"},
                  {"collective", "none"}});
            return true;
        }

        const int token = sampleStochasticDraftProposalOnDevice(
            source,
            row,
            slot,
            params,
            vocab_size,
            threshold);
        if (token < 0)
            return false;
        const int32_t staged_token = static_cast<int32_t>(token);
        return stageRankStochasticDraftTokensForLocalTP(
            &staged_token,
            /*token_count=*/1,
            slot);
    }

    int RankOrchestrator::sampleStochasticDistributionOnDevice(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDistributionOnDevice(
                buffer,
                slot,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDistributionOnDevice(
                buffer,
                slot,
                threshold);
        }

        if (buffer == DeviceDistributionBuffer::Target &&
            slot >= 0 &&
            slot < rank_stochastic_slot_capacity_ &&
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)])
        {
            const int token = device_runners_.front()
                                  ->sampleStochasticDistributionOnDevice(
                                      buffer,
                                      slot,
                                      threshold);
            if (token < 0 ||
                !broadcastPrimaryLocalTPMainTargetSlotToChildren(slot))
            {
                return -1;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
            rank_stochastic_target_sample_tokens_[
                static_cast<size_t>(slot)] = static_cast<int32_t>(token);
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_main_target_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(slot)},
                 {"host_shadow", "true"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return token;
        }

        int32_t token = -1;
        if (!sampleRankStochasticDistribution(
                buffer,
                slot,
                threshold,
                &token))
        {
            return -1;
        }
        return token;
    }

    bool RankOrchestrator::sampleStochasticDistributionOnDeviceDeferred(
        DeviceDistributionBuffer buffer,
        int slot,
        float threshold)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleStochasticDistributionOnDeviceDeferred(
                buffer,
                slot,
                threshold);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleStochasticDistributionOnDeviceDeferred(
                buffer,
                slot,
                threshold);
        }
        if (buffer == DeviceDistributionBuffer::Target &&
            slot >= 0 &&
            slot < rank_stochastic_slot_capacity_ &&
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)])
        {
            if (!device_runners_.front()
                     ->sampleStochasticDistributionOnDeviceDeferred(
                         buffer,
                         slot,
                         threshold) ||
                !broadcastPrimaryLocalTPMainTargetSlotToChildren(slot))
            {
                return false;
            }
            rank_mirrored_target_distribution_ready_[
                static_cast<size_t>(slot)] = false;
            rank_stochastic_target_sample_tokens_[
                static_cast<size_t>(slot)] = -1;
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_main_target_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(slot)},
                 {"host_shadow", "false"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return true;
        }
        return sampleRankStochasticDistribution(
            buffer,
            slot,
            threshold,
            /*out_token=*/nullptr);
    }

    bool RankOrchestrator::stageStochasticTargetTokenForDeviceSampling(
        int32_t target_token,
        int target_sample_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->stageStochasticTargetTokenForDeviceSampling(
                target_token,
                target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->stageStochasticTargetTokenForDeviceSampling(
                target_token,
                target_sample_slot);
        }
        if (!stageRankStochasticTargetTokenForLocalTP(
                target_sample_slot,
                target_token))
        {
            return false;
        }
        rank_stochastic_target_sample_tokens_[
            static_cast<size_t>(target_sample_slot)] = target_token;
        return true;
    }

    bool RankOrchestrator::publishMTPDirectTokenHistory(int target_sample_slot,
        MTPDirectTokenHistoryDisposition disposition)
    {
        if (target_sample_slot != 0 ||
            (disposition != MTPDirectTokenHistoryDisposition::NewlyEmitted &&
             disposition != MTPDirectTokenHistoryDisposition::AlreadyPublished))
            return false;
        if (IInferenceRunner *tail = finalPPSidecarRunner())
            return tail->publishMTPDirectTokenHistory(target_sample_slot, disposition);
        if (device_runners_.size() == 1 && device_runners_[0])
            return device_runners_[0]->publishMTPDirectTokenHistory(target_sample_slot, disposition);
        if (device_runners_.size() < 2 || !primaryDeviceId().is_gpu())
            return false;
        // Sampling history is a participant-local graph input, not a model row
        // or a new distributed controller. The existing dispatch owns each exact
        // device worker and leaves the prior main-logit publication untouched.
        return dispatchLocalTPMTPMainCondition("publishMTPDirectTokenHistory",
            LocalTPConditionEffect::PublishInput,
            [target_sample_slot, disposition](IInferenceRunner &child, size_t) {
                return child.publishMTPDirectTokenHistory(target_sample_slot, disposition);
            });
    }

    bool RankOrchestrator::publishForcedDeviceResidentConditionToken(
        int32_t token,
        int target_sample_slot)
    {
        if (target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            LOG_ERROR(
                "[RankOrchestrator] Forced policy condition target slot is outside rank capacity");
            return false;
        }
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishForcedDeviceResidentConditionToken(
                token, target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->publishForcedDeviceResidentConditionToken(
                    token, target_sample_slot);
        }
        if (device_runners_.size() < 2)
            return false;

        const bool published = dispatchLocalTPMTPMainCondition(
            "publishForcedDeviceResidentConditionToken",
            LocalTPConditionEffect::PublishInput,
            [token, target_sample_slot](
                IInferenceRunner &child, size_t)
            {
                return child.publishForcedDeviceResidentConditionToken(
                    token, target_sample_slot);
            });
        if (!published)
            return false;

        std::string adoption_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                /*expected_request_count=*/1,
                "forced_policy_condition_publication",
                &adoption_error))
        {
            LOG_ERROR(
                "[RankOrchestrator] Forced policy condition publication could not adopt participant mailboxes: "
                << adoption_error);
            return false;
        }
        rank_stochastic_target_sample_tokens_[
            static_cast<size_t>(target_sample_slot)] = token;
        return true;
    }

    bool RankOrchestrator::publishDeviceResidentConditionTokenToTargetSampleSlot(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int target_sample_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar
                ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                    logical_state,
                    request_index,
                    target_sample_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                    logical_state,
                    request_index,
                    target_sample_slot);
        }
        if (device_runners_.size() < 2 ||
            !logical_state.coversRequest(request_index) ||
            logical_state.target_positions_device !=
                rank_resident_logical_state_marker_.data() ||
            logical_state.live_state_epoch != rank_resident_logical_state_epoch_ ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Resident condition-token target publication received a stale or foreign LocalTP mailbox");
            return false;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] resident condition-token target publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             request_index,
             target_sample_slot,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                if (!device_runners_[i])
                    return false;
                const DeviceId device_id =
                    device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i]
                    ->publishDeviceResidentConditionTokenToTargetSampleSlot(
                        rank_resident_child_logical_state_handles_[i],
                        request_index,
                        target_sample_slot);
            });

        bool all_success = true;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &result : results)
        {
            if (!result.completed || !result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident condition-token target publication rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_resident_condition_token_target_publications",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)},
                 {"target_slot", std::to_string(target_sample_slot)},
                 {"transfer", "d2d"}});
        }
        return all_success;
    }

    bool RankOrchestrator::stageStochasticDraftTokensForDeviceVerification(
        const int32_t *draft_tokens,
        int draft_token_count,
        int first_draft_slot)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * Device-side stochastic verification for LocalPP is final-stage
             * owned: the final stage owns verifier logits/distributions and the
             * draft-token slots consumed by the verifier summary kernel.
             */
            return pp_sidecar->stageStochasticDraftTokensForDeviceVerification(
                draft_tokens,
                draft_token_count,
                first_draft_slot);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->stageStochasticDraftTokensForDeviceVerification(
                draft_tokens,
                draft_token_count,
                first_draft_slot);
        }
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            first_draft_slot < 0 ||
            first_draft_slot + draft_token_count > rank_stochastic_slot_capacity_)
        {
            return false;
        }

        if (first_draft_slot == 0)
        {
            std::fill(
                rank_stochastic_draft_sample_tokens_.begin(),
                rank_stochastic_draft_sample_tokens_.end(),
                -1);
        }
        if (rank_stochastic_staged_draft_tokens_.size() <
            static_cast<size_t>(first_draft_slot + draft_token_count))
        {
            rank_stochastic_staged_draft_tokens_.resize(
                static_cast<size_t>(first_draft_slot + draft_token_count),
                -1);
        }
        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token = draft_tokens[i];
            if (token < 0)
                return false;
            rank_stochastic_draft_sample_tokens_[
                static_cast<size_t>(first_draft_slot + i)] = token;
            rank_stochastic_staged_draft_tokens_[
                static_cast<size_t>(first_draft_slot + i)] = token;
        }
        if (!stageRankStochasticDraftTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                first_draft_slot))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_draft_token_stages",
            static_cast<double>(draft_token_count),
            "decode",
            "rank",
            {{"first_slot", std::to_string(first_draft_slot)},
             {"participants", std::to_string(device_runners_.size())}});
        return true;
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDevice(
        int32_t first_token,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * LocalPP verifier replay runs on the logits-owning final stage.
             * The prepared verifier token row is consumed by that same final
             * stage, so it follows final-stage ownership rather than the
             * pipeline-head token input ownership used by normal prefill/decode.
             */
            return pp_sidecar->prepareMTPVerifierInputTokensOnDevice(
                first_token,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDevice(
                first_token,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 > total_verifier_input_tokens)
            return nullptr;

        if (publishesFullVocabularyMTPLogits())
        {
            return prepareRankVerifierTokenSlotsForLocalTP(
                /*first_token_from_device=*/false,
                first_token,
                /*first_target_sample_slot=*/-1,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token =
                rankStochasticDraftSampleToken(first_draft_slot + i);
            if (token < 0)
                return nullptr;
        }
        return prepareRankVerifierTokenSlotsForLocalTP(
            /*first_token_from_device=*/false,
            first_token,
            /*first_target_sample_slot=*/-1,
            first_draft_slot,
            draft_token_count,
            total_verifier_input_tokens);
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokenBatchOnDevice(
        const DeviceMTPVerifierInputBatchRequest *requests,
        int request_count,
        int logical_padded_seq_len)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareMTPVerifierInputTokenBatchOnDevice(
                requests,
                request_count,
                logical_padded_seq_len);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokenBatchOnDevice(
                requests,
                request_count,
                logical_padded_seq_len);
        }
        if (device_runners_.size() < 2 ||
            !requests ||
            request_count <= 0 ||
            logical_padded_seq_len <= 0)
        {
            return nullptr;
        }

        /*
         * Verifier conditions use participant-local canonical target slots.
         * Initial sampling and accepted-state publication both refresh those
         * slots, so rank orchestration forwards immutable slot indices and
         * never translates aggregate logical-mailbox pointers into child state.
         */
        for (int row_index = 0; row_index < request_count; ++row_index)
        {
            const DeviceMTPVerifierInputBatchRequest &row =
                requests[row_index];
            if (!row.first_token_from_device ||
                row.first_target_sample_slot < 0)
            {
                LOG_ERROR("[RankOrchestrator] Request-batched GPU verifier token row "
                          << row_index
                          << " does not name its canonical device target slot");
                return nullptr;
            }
        }

        rank_mtp_verifier_child_token_inputs_.assign(
            device_runners_.size(),
            nullptr);
        rank_mtp_verifier_child_token_count_ = logical_padded_seq_len;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                return nullptr;

            rank_mtp_verifier_child_token_inputs_[i] =
                device_runners_[i]->prepareMTPVerifierInputTokenBatchOnDevice(
                    requests,
                    request_count,
                    logical_padded_seq_len);
            if (!rank_mtp_verifier_child_token_inputs_[i])
            {
                rank_mtp_verifier_child_token_inputs_.clear();
                rank_mtp_verifier_child_token_count_ = 0;
                return nullptr;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_verifier_token_batches_prepared_from_device_slots",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_count)},
             {"logical_padded_seq_len",
              std::to_string(logical_padded_seq_len)}});
        return rank_mtp_verifier_child_token_inputs_.data();
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromHostRow(
        const int32_t *verifier_tokens,
        int total_verifier_input_tokens,
        int draft_token_count)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                verifier_tokens,
                total_verifier_input_tokens,
                draft_token_count);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDeviceFromHostRow(
                verifier_tokens,
                total_verifier_input_tokens,
                draft_token_count);
        }

        if (device_runners_.size() < 2 ||
            !verifier_tokens ||
            total_verifier_input_tokens <= 0)
        {
            return nullptr;
        }

        return stageRankVerifierTokenRowForLocalTP(
            verifier_tokens,
            total_verifier_input_tokens,
            draft_token_count);
    }

    const void *RankOrchestrator::prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
        int first_target_sample_slot,
        int first_draft_slot,
        int draft_token_count,
        int total_verifier_input_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * The first target sample slot and draft sample slots are produced
             * by the final-stage stochastic verifier hooks, so compose the
             * verifier token row on that same stage to avoid cross-stage device
             * pointer aliasing.
             */
            return pp_sidecar->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->prepareMTPVerifierInputTokensOnDeviceFromDeviceFirstToken(
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        if (device_runners_.size() < 2 ||
            total_verifier_input_tokens <= 0 ||
            draft_token_count < 0 ||
            draft_token_count + 1 > total_verifier_input_tokens)
            return nullptr;

        if (publishesFullVocabularyMTPLogits())
        {
            return prepareRankVerifierTokenSlotsForLocalTP(
                /*first_token_from_device=*/true,
                /*first_token=*/-1,
                first_target_sample_slot,
                first_draft_slot,
                draft_token_count,
                total_verifier_input_tokens);
        }

        const int32_t first_token =
            rankStochasticTargetSampleToken(first_target_sample_slot);
        if (first_token < 0)
            return nullptr;

        for (int i = 0; i < draft_token_count; ++i)
        {
            const int32_t token =
                rankStochasticDraftSampleToken(first_draft_slot + i);
            if (token < 0)
                return nullptr;
        }
        return prepareRankVerifierTokenSlotsForLocalTP(
            /*first_token_from_device=*/true,
            /*first_token=*/-1,
            first_target_sample_slot,
            first_draft_slot,
            draft_token_count,
            total_verifier_input_tokens);
    }

    bool RankOrchestrator::verifyStochasticDistributionsOnDevice(
        int target_slot,
        int draft_slot,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        DeviceSpeculativeVerifyResult *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsOnDevice(
                target_slot,
                draft_slot,
                draft_token,
                accept_threshold,
                residual_threshold,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsOnDevice(
                target_slot,
                draft_slot,
                draft_token,
                accept_threshold,
                residual_threshold,
                out);
        }
        return false;
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        DeviceSpeculativeVerifyResult *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsBatchOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                out);
        }
        return false;
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDevice(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int32_t first_token,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsBatchOutcomeOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOutcomeOnDevice(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }

        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_token,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
    }

    bool RankOrchestrator::verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
        int first_target_slot,
        int first_draft_slot,
        const int32_t *draft_tokens,
        const float *accept_thresholds,
        const float *residual_thresholds,
        int row_count,
        int first_target_sample_slot,
        const int32_t *stop_tokens,
        int stop_token_count,
        int bonus_target_slot,
        float bonus_threshold,
        DeviceSpeculativeVerifyBatchOutcome *out,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        bool use_vllm_probability_rejection)
    {
        if (finalPPSidecarRunner())
        {
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsBatchOutcomeOnDeviceFirstToken(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_target_sample_slot,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                out,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection);
        }

        DeviceSpeculativeOutcomeHandle handle;
        if (!verifyStochasticDistributionsBatchOutcomeOnDeviceFirstTokenResident(
                first_target_slot,
                first_draft_slot,
                draft_tokens,
                accept_thresholds,
                residual_thresholds,
                row_count,
                first_target_sample_slot,
                stop_tokens,
                stop_token_count,
                bonus_target_slot,
                bonus_threshold,
                &handle,
                inverse_sample_seed,
                inverse_sample_first_logical_position,
                use_vllm_probability_rejection))
        {
            return false;
        }
        return copyDeviceSpeculativeOutcomesToHostForDiagnostics(handle, out);
    }

    DeviceGenerationExecutionPolicy
    RankOrchestrator::deviceGenerationExecutionPolicy(
        DeviceGenerationLoopTopology topology) const noexcept
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
            return DeviceGenerationExecutionPolicy::Unsupported;

        DeviceGenerationExecutionPolicy rank_policy =
            DeviceGenerationExecutionPolicy::NativeConditionalGraph;
        for (const auto &participant : participants)
        {
            if (!participant)
                return DeviceGenerationExecutionPolicy::Unsupported;

            const DeviceGenerationExecutionPolicy participant_policy =
                participant->deviceGenerationExecutionPolicy(topology);
            if (participant_policy ==
                DeviceGenerationExecutionPolicy::Unsupported)
            {
                return DeviceGenerationExecutionPolicy::Unsupported;
            }
            if (participant_policy ==
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions)
            {
                rank_policy = DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions;
            }
        }
        return rank_policy;
    }

    bool RankOrchestrator::beginDeviceResidentGeneration(
        const DeviceGenerationAdmissionRequest &request)
    {
        if (!pp_stage_runners_.empty())
        {
            if (!pp_device_generation_)
                pp_device_generation_ = std::make_unique<PipelineDeviceGeneration>(pp_stage_runners_, config_.original_pipeline_mpi_ctx);
            return pp_device_generation_->begin(request);
        }
        if (!request.valid())
        {
            LOG_ERROR("[RankOrchestrator] Invalid device-resident generation admission: requests="
                      << request.request_count << " max_new_tokens="
                      << request.max_new_tokens << " leading_row_disposition="
                      << static_cast<int>(
                             request.initial_leading_row_disposition));
            return false;
        }
        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        materialized_device_generation_execution_policy_.reset();
        materialized_device_generation_loop_topology_.reset();
        rank_hosted_device_generation_tickets_.clear();
        hosted_initial_graph_sequence_ =
            request.depth_policy.isOrdinary() &&
                    request.initial_leading_row_disposition ==
                        sampling_math::DeviceGenerationLeadingRowDisposition::
                            AlreadyEmitted
                ? HostedInitialGraphSequence::Required
                : HostedInitialGraphSequence::NotRequired;
        admitted_device_generation_max_new_tokens_ = 0;

        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-resident generation admission has no participants");
            return false;
        }

        // Admission may enqueue original peer setup/header publications. Every
        // participant must enter before any host joins its enqueue completion;
        // a full prefix hit can reach this boundary without a preceding forward.
        // The same original source attempt retains rollback until ALL callbacks
        // have drained, exactly as for main-forward and parent preparation.
        for (const auto& participant : participants)
            if (!participant)
                throw std::logic_error("Device-generation admission has an absent original participant");
        if (participants.size() == 1)
        {
            if (!participants.front()->beginDeviceResidentGeneration(request)) return false;
        }
        else
        {
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ = makeParticipantWorkerPool(participants);
                if (tp_ctx_)
                    tp_worker_pool_->setFailureCallback([this]() {
                        LOG_WARN("[TPWorkerPool] Device-generation admission failure detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
            if (tp_worker_pool_->numWorkers() != participants.size())
                throw std::logic_error("Device-generation admission changed its original participant worker roster");
            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch([this, request, kernel_phase, rocm_phase,
                cuda_phase, kv_phase, executor_phase](size_t i) -> bool {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);
                const auto device = device_runners_.at(i)->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return device_runners_[i]->beginDeviceResidentGeneration(request);
            });
            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            const auto results = tp_worker_pool_->collectAll(timeout_ms);
            bool success = results.size() == participants.size();
            bool incomplete = false;
            std::exception_ptr first_exception;
            for (const auto& result : results)
            {
                incomplete = incomplete || !result.completed;
                success = success && result.completed && result.success;
                if (result.exception && !first_exception) first_exception = result.exception;
            }
            if (incomplete && timeout_ms > 0)
                abortAfterTPWorkerTimeout("beginDeviceResidentGeneration",timeout_ms,
                    tp_worker_pool_->completedCount(),tp_worker_pool_->numWorkers());
            if (first_exception) std::rethrow_exception(first_exception);
            if (!success) return false;
        }
        source_attempt.keep=true;
        admitted_device_generation_max_new_tokens_ = request.max_new_tokens;
        return true;
    }

    bool RankOrchestrator::materializeDeviceResidentGeneration(
        int request_count,
        int draft_depth,
        DeviceGenerationLoopTopology topology,
        DeviceGenerationSamplingMode sampling_mode)
    {
        if (pp_device_generation_ && pp_device_generation_->active())
            return pp_device_generation_->materialize(request_count, draft_depth, topology, sampling_mode);
        // Zero is the only valid draft depth for the separately typed ordinary
        // generation policy.  Negative values remain invalid: they can never
        // name a captured transaction geometry.
        if (request_count <= 0 || draft_depth < 0 ||
            !isValidDeviceGenerationSamplingMode(sampling_mode))
        {
            LOG_ERROR("[RankOrchestrator] Invalid device-generation parent preparation geometry"
                      << " requests=" << request_count
                      << " draft_depth=" << draft_depth
                      << " sampling_mode="
                      << deviceGenerationSamplingModeName(sampling_mode));
            return false;
        }

        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto &participants =
            use_pp_participants ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation has no participants");
            return false;
        }
        const DeviceGenerationExecutionPolicy execution_policy =
            deviceGenerationExecutionPolicy(topology);
        if (execution_policy ==
            DeviceGenerationExecutionPolicy::Unsupported)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation materialization has no complete rank execution policy");
            return false;
        }
        if (participants.size() == 1)
        {
            const bool materialized =
                participants.front() &&
                participants.front()->materializeDeviceResidentGeneration(
                    request_count,
                    draft_depth,
                    topology,
                    sampling_mode);
            if (materialized)
            {
                materialized_device_generation_execution_policy_ =
                    execution_policy;
                materialized_device_generation_loop_topology_ = topology;
                if (execution_policy ==
                    DeviceGenerationExecutionPolicy::
                        HostScheduledCapturedTransactions)
                {
                    rank_hosted_device_generation_tickets_.resize(1);
                }
            }
            return materialized;
        }

        /*
         * Parent composition calls backend graph-cloning/instantiation APIs and
         * therefore belongs to each participant's persistent worker context.
         * Complete every composition before launchDeviceResident... submits any
         * graph: a rejected child on participant N must never be discovered
         * after participant zero has entered a collective-bearing WHILE body.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(participants);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN("[TPWorkerPool] Device-generation parent preparation failure detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation participant count does not match the persistent TP worker pool");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this, use_pp_participants, request_count, draft_depth, topology,
             sampling_mode,
             kernel_phase, rocm_phase, cuda_phase, kv_phase,
             executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto &worker_participants =
                    use_pp_participants
                        ? pp_stage_runners_
                        : device_runners_;
                if (i >= worker_participants.size() ||
                    !worker_participants[i])
                {
                    return false;
                }

                const DeviceId device =
                    worker_participants[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return worker_participants[i]
                    ->materializeDeviceResidentGeneration(
                        request_count,
                        draft_depth,
                        topology,
                        sampling_mode);
            });

        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
            }
            if (!result.success)
                all_success = false;
            if (result.exception && !first_exception)
            {
                first_exception = result.exception;
                first_exception_device = result.worker_index;
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "materializeDeviceResidentGeneration",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent preparation re-throwing primary exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if(all_success && replicated_ple_cohort_)
        {
            // All child captures are complete, but NO participant has launched.
            // Inspect actual source/sampler/publication declarations before the
            // existing materialized-policy authority can admit any collective loop.
            const auto *first=dynamic_cast<const DeviceGraphOrchestrator *>(participants.front().get());
            if(!first)throw std::invalid_argument("Fixed-head PLE requires actual device-owned generation participants");
            for(const auto &participant:participants)
            {
                const auto *child=dynamic_cast<const DeviceGraphOrchestrator *>(participant.get());
                if(!child)throw std::invalid_argument("Fixed-head PLE participant has no actual DGO producer");
                child->requireReplicatedPLEGenerationProducer(*first);
            }
        }
        if (all_success)
        {
            materialized_device_generation_execution_policy_ =
                execution_policy;
            materialized_device_generation_loop_topology_ = topology;
            if (execution_policy ==
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions)
            {
                rank_hosted_device_generation_tickets_.resize(
                    participants.size());
            }
        }
        return all_success;
    }

    bool RankOrchestrator::observeDeviceGenerationDispatchTicket(
        sampling_math::DeviceGenerationDispatchTicket *out_ticket)
    {
        if (out_ticket)
            *out_ticket = sampling_math::DeviceGenerationDispatchTicket{};
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (!out_ticket ||
            !materialized_device_generation_execution_policy_ ||
            *materialized_device_generation_execution_policy_ !=
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions ||
            participants.empty() ||
            rank_hosted_device_generation_tickets_.size() !=
                participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Hosted device-generation ticket observation has no exact materialized rank policy");
            return false;
        }

        /*
         * A continued ordinary response enters a sparse MainDecode graph before
         * its first ticket exists. Open the one rank-wide sequence exactly once
         * before any child can reserve a participant binding. Fresh-prefill
         * sampling and non-overlay hosted graphs have no such boundary.
         */
        if (hosted_initial_graph_sequence_ ==
            HostedInitialGraphSequence::Required)
        {
            if (moe_overlay_inference_transaction_coordinator_)
            {
                std::string sequence_error;
                if (!moe_overlay_inference_transaction_coordinator_
                         ->beginGraphSequence(
                             /*draft_depth=*/0, &sequence_error))
                {
                    LOG_ERROR(
                        "[RankOrchestrator] Hosted ordinary continuation "
                        "could not admit its initial ExpertOverlay graph "
                        "sequence: "
                        << sequence_error);
                    return false;
                }
            }
            hosted_initial_graph_sequence_ =
                HostedInitialGraphSequence::Admitted;
        }

        const bool use_pp_participants = !pp_stage_runners_.empty();
        const auto observe_participant =
            [this, &participants](size_t participant_index) -> bool
        {
            return participant_index < participants.size() &&
                participants[participant_index] &&
                participants[participant_index]
                    ->observeDeviceGenerationDispatchTicket(
                        &rank_hosted_device_generation_tickets_[
                            participant_index]);
        };

        bool observations_complete = true;
        if (participants.size() == 1)
        {
            observations_complete = observe_participant(0);
        }
        else
        {
            /*
             * Initial ExpertOverlay replay enters one symmetric sparse graph.
             * Every participant must therefore submit from its persistent TP
             * worker before any one of them waits for the resulting ticket.
             * Sequential observation would deadlock at the first collective
             * participant and was the hidden lifecycle defect exposed by
             * bounded HTTP streaming continuations.
             */
            if (!tp_worker_pool_ ||
                tp_worker_pool_->numWorkers() != participants.size())
            {
                LOG_ERROR("[RankOrchestrator] Hosted ticket observation has no matching persistent TP worker pool");
                return false;
            }
            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, use_pp_participants, kernel_phase, rocm_phase,
                 cuda_phase, kv_phase, executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    auto &worker_participants =
                        use_pp_participants ? pp_stage_runners_
                                            : device_runners_;
                    if (index >= worker_participants.size() ||
                        !worker_participants[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        worker_participants[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return worker_participants[index]
                        ->observeDeviceGenerationDispatchTicket(
                            &rank_hosted_device_generation_tickets_[index]);
                });

            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            auto results = tp_worker_pool_->collectAll(timeout_ms);
            for (const auto &result : results)
            {
                if (!result.completed || !result.success || result.exception)
                {
                    observations_complete = false;
                    if (!result.completed && timeout_ms > 0)
                    {
                        abortAfterTPWorkerTimeout(
                            "observeDeviceGenerationDispatchTicket",
                            timeout_ms,
                            tp_worker_pool_->completedCount(),
                            tp_worker_pool_->numWorkers());
                    }
                    break;
                }
            }
        }
        if (!observations_complete)
        {
            LOG_ERROR("[RankOrchestrator] Hosted device-generation ticket observation failed on one or more participants");
            return false;
        }

        const auto &authoritative =
            rank_hosted_device_generation_tickets_.front();
        for (size_t participant_index = 1;
             participant_index <
                 rank_hosted_device_generation_tickets_.size();
             ++participant_index)
        {
            if (!authoritative.hasSameDispatchDecision(
                    rank_hosted_device_generation_tickets_[
                        participant_index]))
            {
                const auto &divergent =
                    rank_hosted_device_generation_tickets_[
                        participant_index];
                LOG_ERROR("[RankOrchestrator] Device-generation controllers selected divergent hosted graph branches"
                          << " participant=" << participant_index
                          << " authoritative_transaction="
                          << authoritative.transaction_count
                          << " divergent_transaction="
                          << divergent.transaction_count
                          << " authoritative_depth="
                          << authoritative.next_draft_depth
                          << " divergent_depth="
                          << divergent.next_draft_depth
                          << " authoritative_complete="
                          << authoritative.complete
                          << " divergent_complete=" << divergent.complete
                          << " authoritative_maintenance="
                          << authoritative.maintenance_due
                          << " divergent_maintenance="
                          << divergent.maintenance_due);
                std::terminate();
            }
        }

        *out_ticket = authoritative;
        return true;
    }

    bool RankOrchestrator::submitHostScheduledDeviceGenerationAdvance(
        const sampling_math::DeviceGenerationDispatchTicket &ticket)
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (!materialized_device_generation_execution_policy_ ||
            *materialized_device_generation_execution_policy_ !=
                DeviceGenerationExecutionPolicy::
                    HostScheduledCapturedTransactions ||
            participants.empty() ||
            rank_hosted_device_generation_tickets_.size() !=
                participants.size() ||
            !ticket.hasSameDispatchDecision(
                rank_hosted_device_generation_tickets_.front()))
        {
            LOG_ERROR("[RankOrchestrator] Hosted device-generation submission rejected a stale rank decision");
            return false;
        }

        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto dispatch_to_all =
            [&](const char *operation,
                const std::function<bool(IInferenceRunner *, size_t)> &fn)
                -> bool
        {
            if (participants.size() == 1)
                return fn(participants.front().get(), 0);
            if (!tp_worker_pool_ ||
                tp_worker_pool_->numWorkers() != participants.size())
            {
                LOG_ERROR("[RankOrchestrator] " << operation
                          << " has no matching persistent TP worker pool");
                return false;
            }

            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, use_pp_participants, fn, kernel_phase, rocm_phase,
                 cuda_phase, kv_phase, executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    auto &worker_participants =
                        use_pp_participants ? pp_stage_runners_
                                            : device_runners_;
                    if (index >= worker_participants.size() ||
                        !worker_participants[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        worker_participants[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return fn(worker_participants[index].get(), index);
                });

            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            auto results = tp_worker_pool_->collectAll(timeout_ms);
            for (const auto &result : results)
            {
                if (!result.completed || !result.success || result.exception)
                {
                    if (!result.completed && timeout_ms > 0)
                    {
                        abortAfterTPWorkerTimeout(
                            operation,
                            timeout_ms,
                            tp_worker_pool_->completedCount(),
                            tp_worker_pool_->numWorkers());
                    }
                    return false;
                }
            }
            return true;
        };

        /*
         * Submit the complete authenticated transaction on every persistent
         * participant worker in one rank-level fan-out. Each child walks the
         * same immutable producer-ordered branch and its sparse graph scopes
         * provide the only rendezvous required between participants. Keeping
         * the worker alive for the whole transaction removes one host
         * dispatch/collection barrier per semantic fragment without changing
         * device stream order or allowing either the rank or host to inspect
         * mutable generation state.
         */
        if (!dispatch_to_all(
                "submitHostScheduledDeviceGenerationAdvance",
                [this](IInferenceRunner *participant,
                       size_t participant_index) -> bool
                {
                    return participant &&
                        participant
                            ->submitHostScheduledDeviceGenerationAdvance(
                                rank_hosted_device_generation_tickets_[
                                    participant_index]);
                }))
        {
            LOG_ERROR("[RankOrchestrator] Hosted transaction-bundle submission failed after distributed submission began");
            std::terminate();
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_hosted_transaction_bundle_submissions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"worker_fanouts", "1"},
             {"fragment_barriers", "0"}});
        return true;
    }

    bool RankOrchestrator::prepareDeviceResidentGenerationContinuation(DeviceGenerationSamplingMode sampling_mode)
    {
        if (pp_device_generation_ && pp_device_generation_->active())
            return pp_device_generation_->prepareContinuation(sampling_mode);
        const auto &participants = !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty() || !isValidDeviceGenerationSamplingMode(sampling_mode) ||
            admitted_device_generation_max_new_tokens_ <= 0 ||
            deviceGenerationExecutionPolicy(DeviceGenerationLoopTopology::DynamicDepth) !=
                DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
        {
            LOG_ERROR("[RankOrchestrator] ContinueResponse lacks its complete retained hosted rank family");
            return false;
        }
        const auto publish_retained_policy = [&]() {
            materialized_device_generation_execution_policy_ = DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions;
            materialized_device_generation_loop_topology_ = DeviceGenerationLoopTopology::DynamicDepth;
            rank_hosted_device_generation_tickets_.resize(participants.size());
        };
        if (participants.size() == 1)
        {
            if (!participants.front() || !participants.front()->prepareDeviceResidentGenerationContinuation(sampling_mode))
            {
                LOG_ERROR("[RankOrchestrator] ContinueResponse retained-family preparation failed on the sole participant");
                return false;
            }
            publish_retained_policy();
            return true;
        }
        if (!tp_worker_pool_ || tp_worker_pool_->numWorkers() != participants.size())
        {
            LOG_ERROR("[RankOrchestrator] ContinueResponse has no exact persistent worker cohort"
                << " participants=" << participants.size() << " workers=" << (tp_worker_pool_ ? tp_worker_pool_->numWorkers() : 0));
            return false;
        }
        tp_worker_pool_->dispatch([&participants, sampling_mode](size_t index) {
            return index < participants.size() && participants[index] &&
                participants[index]->prepareDeviceResidentGenerationContinuation(sampling_mode);
        });
        const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(timeout_ms);
        bool success = results.size() == participants.size();
        for (const auto &result : results)
        {
            if (!result.completed && timeout_ms > 0)
                abortAfterTPWorkerTimeout("prepareDeviceResidentGenerationContinuation", timeout_ms, result.worker_index, participants.size());
            if (result.exception) std::rethrow_exception(result.exception);
            if (!result.completed || !result.success)
                LOG_ERROR("[RankOrchestrator] ContinueResponse retained-family preparation failed"
                    << " participant=" << result.worker_index << " completed=" << result.completed
                    << " success=" << result.success);
            success = success && result.completed && result.success;
        }
        if (success) publish_retained_policy();
        return success;
    }

    bool RankOrchestrator::launchDeviceResidentGeneration()
    {
        if (pp_device_generation_ && pp_device_generation_->active())
            return pp_device_generation_->launch();
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Device-generation parent launch has no participants");
            return false;
        }
        if (!materialized_device_generation_execution_policy_ ||
            !materialized_device_generation_loop_topology_)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation launch has no exact materialized policy/topology");
            return false;
        }

        if (*materialized_device_generation_execution_policy_ ==
            DeviceGenerationExecutionPolicy::
                HostScheduledCapturedTransactions)
        {
            RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};
            if (admitted_device_generation_max_new_tokens_ <= 0)
            {
                LOG_ERROR("[RankOrchestrator] Hosted device-generation loop has no admitted transaction bound");
                return false;
            }
            for (std::int64_t observation = 0;
                 observation <= admitted_device_generation_max_new_tokens_;
                 ++observation)
            {
                sampling_math::DeviceGenerationDispatchTicket ticket;
                if (!observeDeviceGenerationDispatchTicket(&ticket) ||
                    !submitHostScheduledDeviceGenerationAdvance(ticket))
                {
                    LOG_ERROR("[RankOrchestrator] Hosted device-generation transaction loop failed");
                    return false;
                }
                if (ticket.complete != 0)
                {
                    PerfStatsCollector::addCounter(
                        "mtp",
                        "rank_device_generation_parent_launches",
                        1.0,
                        "decode",
                        "rank",
                        {{"participants",
                          std::to_string(participants.size())},
                         {"launch_order",
                          "all_participants_before_ticket_wait"},
                         {"execution",
                          "hosted_ticket_selected_captured_transactions"}});
                    source_attempt.keep=true;
                    return true;
                }
            }
            LOG_ERROR("[RankOrchestrator] Hosted device-generation loop exceeded its admitted response bound without completing");
            return false;
        }
        if (*materialized_device_generation_execution_policy_ !=
            DeviceGenerationExecutionPolicy::NativeConditionalGraph)
        {
            LOG_ERROR("[RankOrchestrator] Device-generation launch encountered an invalid materialized policy");
            return false;
        }

        /*
         * Enqueue every participant before waiting for any terminal result.
         * Captured NCCL/RCCL nodes may rendezvous across these graphs, so
         * synchronizing participant zero before participant one is launched
         * would deadlock the domain. A failure after any earlier launch leaves
         * a partially submitted distributed transaction and is therefore
         * process-fatal rather than recoverable.
         */
        for (size_t participant_index = 0;
             participant_index < participants.size();
             ++participant_index)
        {
            if (!participants[participant_index] ||
                !participants[participant_index]
                     ->launchDeviceResidentGeneration())
            {
                LOG_ERROR("[RankOrchestrator] Device-generation parent launch failed on participant "
                          << participant_index);
                if (participant_index != 0)
                    std::terminate();
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_device_generation_parent_launches",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"launch_order", "all_participants_before_terminal_wait"}});
        return true;
    }

    bool RankOrchestrator::finishDeviceResidentGeneration(
        DeviceGenerationTerminalResult *out_result)
    {
        if (pp_device_generation_ && pp_device_generation_->active())
        {
            if (!pp_device_generation_->finish(out_result)) return false;
            // Observe completed domain traversal only at the authenticated
            // terminal. Depth zero is the ordinary controller: its first
            // sampled token can use existing prefill logits without another
            // forward. Only its committed state rows traverse the pipeline.
            // A speculative transaction traverses every domain once regardless
            // of depth. These terminal fields are evidence, never a scheduler.
            std::size_t transactions = 0;
            for (const auto &request : out_result->requests)
                transactions += request.final_draft_depth > 0 ? request.transaction_count
                    : request.published_state_commit_count;
            if (transactions)
                recordCompletedPPGraphTransactions("decode", pp_stage_runners_.size(), transactions);
            return true;
        }
        if (out_result)
            *out_result = DeviceGenerationTerminalResult{};
        if (!out_result)
        {
            LOG_ERROR("[RankOrchestrator] Terminal device-generation result requires a destination");
            return false;
        }

        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Terminal device-generation result has no participants");
            return false;
        }

        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};
        std::vector<DeviceGenerationTerminalResult> participant_results;
        participant_results.reserve(participants.size());
        for (size_t participant_index = 0;
             participant_index < participants.size();
             ++participant_index)
        {
            DeviceGenerationTerminalResult participant_result;
            auto* original=retained_source_admission_
                ? dynamic_cast<DeviceGraphOrchestrator*>(participants[participant_index].get()):nullptr;
            const bool completed=participants[participant_index] &&
                (original?original->finishGroupedDeviceResidentGeneration(&participant_result)
                    :participants[participant_index]->finishDeviceResidentGeneration(&participant_result));
            if (!completed || !participant_result.valid())
            {
                LOG_ERROR("[RankOrchestrator] Terminal device-generation materialization failed on participant "
                          << participant_index);
                return false;
            }
            participant_results.push_back(
                std::move(participant_result));
        }

        // Each original result event has completed, including its captured close.
        // A fast participant cannot cancel a peer whose terminal is still queued.
        if (retained_source_admission_)
            awaitRetainedSourceAdmission();

        const auto &authoritative_requests =
            participant_results.front().requests;
        for (size_t participant_index = 1;
             participant_index < participant_results.size();
             ++participant_index)
        {
            if (participant_results[participant_index].requests !=
                authoritative_requests)
            {
                std::vector<std::vector<MTPMirroredTensorDigest>>
                    participant_digests;
                participant_digests.reserve(participants.size());
                for (const auto &participant : participants)
                {
                    auto *device_orchestrator =
                        dynamic_cast<DeviceGraphOrchestrator *>(
                            participant.get());
                    participant_digests.push_back(
                        device_orchestrator
                            ? device_orchestrator
                                  ->captureMirroredMTPDigests()
                            : std::vector<MTPMirroredTensorDigest>{});
                }
                LOG_ERROR("[RankOrchestrator] Mirrored terminal device-generation ledgers disagree"
                          << " primary="
                          << participant_results.front().device.toString()
                          << " participant="
                          << participant_results[participant_index]
                                 .device.toString()
                          << " participant_index=" << participant_index
                          << describeTerminalLedgerMismatch(
                                 authoritative_requests,
                                 participant_results[participant_index]
                                     .requests)
                          << describeMirroredDigestMismatch(
                                 participant_digests));
                return false;
            }
        }

        /*
         * A complete resident generation publishes a fresh terminal logical
         * mailbox on every LocalTP child.  The rank aggregate is an immutable
         * set of those child mailbox identities, so the aggregate retained
         * before launch is necessarily stale after the terminal publication.
         * Adopt the new identities before exposing the terminal ledger to the
         * caller.  This is pointer/event bookkeeping only; payload ownership
         * and ordering remain participant-local and device-resident.
         *
         * A stopped request has no continuation consumer.  Retire the rank
         * aggregate explicitly instead of leaving a terminal mailbox
         * accidentally observable across the request boundary.
         */
        if (pp_stage_runners_.empty() && device_runners_.size() > 1)
        {
            const bool has_live_continuation =
                std::any_of(
                    authoritative_requests.begin(),
                    authoritative_requests.end(),
                    [](const DeviceGenerationTerminalRequestResult &request)
                    {
                        return !request.model_stopped;
                    });
            if (has_live_continuation)
            {
                std::string mailbox_error;
                if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                        static_cast<int>(authoritative_requests.size()),
                        "device_generation_terminal",
                        &mailbox_error))
                {
                    LOG_ERROR(
                        "[RankOrchestrator] Terminal device-generation result "
                        "could not adopt the child continuation mailboxes: "
                        << mailbox_error);
                    return false;
                }
            }
            else
            {
                invalidateRankResidentLogicalStateAggregate(
                    "device_generation_terminal",
                    "all_requests_stopped");
            }
        }

        *out_result = std::move(participant_results.front());
        materialized_device_generation_execution_policy_.reset();
        materialized_device_generation_loop_topology_.reset();
        rank_hosted_device_generation_tickets_.clear();
        hosted_initial_graph_sequence_ =
            HostedInitialGraphSequence::Inactive;
        admitted_device_generation_max_new_tokens_ = 0;
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_device_generation_terminal_response_bridges",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(participants.size())},
             {"validation", "byte_identical_host_terminal_results"}});
        source_attempt.keep=true;
        return true;
    }

    bool RankOrchestrator::verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (out_handle)
            *out_handle = DeviceSpeculativeOutcomeHandle{};
        rank_compact_outcome_valid_ = false;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_resident_child_logical_state_handles_.clear();

        if (publishesFullVocabularyMTPLogits())
        {
            return verifyStochasticMirroredLocalTPRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (primaryDeviceId().is_gpu())
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident stochastic verification requires mirrored full-head child verifier outcomes");
            return false;
        }

        if (!requests ||
            request_count != 1 ||
            !out_handle ||
            device_runners_.size() < 2 ||
            !supportsDeviceStochasticMTPVerification())
        {
            return false;
        }

        const DeviceStochasticBatchOutcomeRequest &request = requests[0];
        if (request.row_count <= 0 ||
            request.row_count > rank_max_draft_depth_ ||
            request.first_target_slot < 0 ||
            request.first_draft_slot < 0 ||
            request.first_target_slot + request.row_count >
                rank_stochastic_slot_capacity_ ||
            request.first_draft_slot + request.row_count >
                rank_stochastic_slot_capacity_ ||
            request.stop_token_count < 0 ||
            request.stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (request.bonus_target_slot >= rank_stochastic_slot_capacity_) ||
            (!request.use_device_draft_tokens &&
             request.draft_tokens.size() <
                 static_cast<size_t>(request.row_count)) ||
            (!request.derive_thresholds_from_seed &&
             (request.accept_thresholds.size() <
                  static_cast<size_t>(request.row_count) ||
              request.residual_thresholds.size() <
                  static_cast<size_t>(request.row_count))))
        {
            return false;
        }

        const bool derive_thresholds =
            request.derive_thresholds_from_seed &&
            request.draw_position_source ==
                DeviceStochasticDrawPositionSource::HostLogicalPosition &&
            request.use_vllm_probability_rejection &&
            request.inverse_sample_seed != 0 &&
            request.inverse_sample_first_logical_position >= 0;
        if (request.derive_thresholds_from_seed != derive_thresholds)
        {
            LOG_ERROR("[RankOrchestrator] CPU rank stochastic verifier cannot consume a device-owned draw-position descriptor");
            return false;
        }
        std::vector<float> accept_thresholds(
            static_cast<size_t>(request.row_count),
            0.0f);
        std::vector<float> residual_thresholds(
            static_cast<size_t>(request.row_count),
            0.0f);
        for (int row = 0; row < request.row_count; ++row)
        {
            if (derive_thresholds)
            {
                const int logical_position =
                    request.inverse_sample_first_logical_position + row;
                accept_thresholds[static_cast<size_t>(row)] =
                    mtp_spec_threshold_from_seed(
                        request.inverse_sample_seed,
                        logical_position,
                        1 /* MTPSpecStochasticDrawPurpose::Accept */);
                residual_thresholds[static_cast<size_t>(row)] =
                    mtp_spec_threshold_from_seed(
                        request.inverse_sample_seed,
                        logical_position,
                        2 /* MTPSpecStochasticDrawPurpose::Residual */);
            }
            else
            {
                accept_thresholds[static_cast<size_t>(row)] =
                    request.accept_thresholds[static_cast<size_t>(row)];
                residual_thresholds[static_cast<size_t>(row)] =
                    request.residual_thresholds[static_cast<size_t>(row)];
            }
        }

        const int32_t first_token =
            request.first_token_from_device
                ? rankStochasticTargetSampleToken(
                      request.first_target_sample_slot)
                : request.first_token;
        if (first_token < 0)
            return false;

        std::vector<int32_t> actual_draft_tokens;
        actual_draft_tokens.reserve(
            static_cast<size_t>(request.row_count + 1));
        actual_draft_tokens.push_back(first_token);

        std::vector<int> row_tokens(
            static_cast<size_t>(request.row_count),
            -1);
        std::vector<int> row_accepted(
            static_cast<size_t>(request.row_count),
            0);

        for (int row = 0; row < request.row_count; ++row)
        {
            const int draft_slot = request.first_draft_slot + row;
            const int32_t draft_token =
                request.use_device_draft_tokens
                    ? rankStochasticDraftSampleToken(draft_slot)
                    : request.draft_tokens[static_cast<size_t>(row)];
            if (draft_token < 0)
                return false;
            actual_draft_tokens.push_back(draft_token);

            const int target_slot = request.first_target_slot + row;
            const int target_top_k =
                rank_stochastic_target_top_k_[
                    static_cast<size_t>(target_slot)];
            if (target_top_k <= 0 ||
                target_top_k > kMaxTopK)
            {
                return false;
            }

            const size_t slot_offset =
                static_cast<size_t>(target_slot) *
                static_cast<size_t>(kMaxTopK);
            int out_token = -1;
            int out_accepted = 0;
            float accept_probability = 0.0f;
            float accept_threshold = 0.0f;
            if (request.use_vllm_probability_rejection)
            {
                speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
                    rank_stochastic_target_token_ids_.data() + slot_offset,
                    rank_stochastic_target_probs_.data() + slot_offset,
                    target_top_k,
                    vocab_size(),
                    draft_token,
                    accept_thresholds[static_cast<size_t>(row)],
                    request.inverse_sample_seed,
                    request.inverse_sample_first_logical_position + row,
                    &out_token,
                    &out_accepted,
                    &accept_probability,
                    &accept_threshold);
            }
            else
            {
                speculative_verify_with_thresholds_one_hot_draft(
                    rank_stochastic_target_token_ids_.data() + slot_offset,
                    rank_stochastic_target_probs_.data() + slot_offset,
                    target_top_k,
                    draft_token,
                    accept_thresholds[static_cast<size_t>(row)],
                    residual_thresholds[static_cast<size_t>(row)],
                    &out_token,
                    &out_accepted,
                    &accept_probability,
                    &accept_threshold);
            }
            (void)accept_probability;
            (void)accept_threshold;
            if (out_token < 0)
                return false;
            row_tokens[static_cast<size_t>(row)] = out_token;
            row_accepted[static_cast<size_t>(row)] = out_accepted;
        }

        int bonus_token = -1;
        int has_bonus = 0;
        if (request.bonus_target_slot >= 0)
        {
            int32_t sampled_bonus = -1;
            if (!sampleRankStochasticDistribution(
                    DeviceDistributionBuffer::Target,
                    request.bonus_target_slot,
                    request.bonus_threshold,
                    &sampled_bonus))
            {
                return false;
            }
            bonus_token = sampled_bonus;
            has_bonus = 1;
        }

        std::vector<int> output_tokens_int(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens{};
        meta.fill(0);
        packed_stop_tokens.fill(-1);
        for (int i = 0; i < request.stop_token_count; ++i)
        {
            packed_stop_tokens[static_cast<size_t>(i)] =
                static_cast<int>(request.stop_tokens[static_cast<size_t>(i)]);
        }
        summarize_speculative_verify_batch(
            first_token,
            row_tokens.data(),
            row_accepted.data(),
            request.row_count,
            request.stop_token_count > 0 ? packed_stop_tokens.data() : nullptr,
            request.stop_token_count,
            bonus_token,
            has_bonus,
            output_tokens_int.data(),
            static_cast<int>(output_tokens_int.size()),
            meta.data());
        if (meta[kSpecBatchMetaOk] == 0)
            return false;

        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
        for (size_t i = 0; i < rank_compact_output_tokens_.size(); ++i)
        {
            rank_compact_output_tokens_[i] =
                static_cast<int32_t>(output_tokens_int[i]);
        }
        rank_compact_output_meta_ = meta;
        rank_compact_last_draft_tokens_ = std::move(actual_draft_tokens);
        rank_compact_last_stop_tokens_.clear();
        for (int i = 0; i < request.stop_token_count; ++i)
        {
            rank_compact_last_stop_tokens_.push_back(
                request.stop_tokens[static_cast<size_t>(i)]);
        }

        out_handle->output_tokens_device = rank_compact_output_tokens_.data();
        out_handle->meta_device = rank_compact_output_meta_.data();
        out_handle->request_count = 1;
        out_handle->logical_verifier_rows_per_request = request.row_count + 1;
        out_handle->physical_verifier_rows_per_request = request.row_count + 1;
        out_handle->output_token_stride = rank_compact_output_token_stride_;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = primaryDeviceId();
        out_handle->stream = &rank_compact_outcome_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &rank_compact_outcome_ready_event_token_,
                [](void *) {});

        rank_compact_outcome_kind_ = RankCompactOutcomeKind::Stochastic;
        rank_compact_outcome_valid_ = out_handle->valid();
        if (!rank_compact_outcome_valid_)
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_resident_compatible_outcomes",
            static_cast<double>(request.row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"implementation", "rank_owned_compact_stochastic_topk"}});
        return true;
    }

    bool RankOrchestrator::setComputeAllPositionLogits(bool enabled)
    {
        // Pipeline followers have no vocabulary output. Their typed main-forward
        // policy is installed by the pipeline compiler, never by a tail sampler.
        if (pp_device_generation_)
        {
            const bool ok = pp_stage_runners_.back()->setComputeAllPositionLogits(enabled);
            if (ok && !enabled) current_all_position_logit_rows_ = 0;
            return ok;
        }
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setComputeAllPositionLogits(enabled))
            {
                all_success = false;
            }
        }
        if (all_success && !enabled)
        {
            // Child runners clear their row-indexed graph-builder state when
            // all-position logits are disabled. Mirror that aggregate state so
            // TP logits gathering uses the full-row shape on the next request.
            current_all_position_logit_rows_ = 0;
        }
        return all_success;
    }

    bool RankOrchestrator::setComputeRowIndexedAllPositionLogits(bool enabled, int row_count)
    {
        if (pp_device_generation_)
        {
            const bool ok = pp_stage_runners_.back()->setComputeRowIndexedAllPositionLogits(enabled, row_count);
            if (ok) current_all_position_logit_rows_ = enabled ? row_count : 0;
            return ok;
        }
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setComputeRowIndexedAllPositionLogits(enabled, row_count))
            {
                all_success = false;
            }
        }
        if (all_success)
        {
            current_all_position_logit_rows_ = enabled ? row_count : 0;
        }
        return all_success;
    }

    bool RankOrchestrator::setMTPSpecVerifierInputPlan(
        const MTPSpecDecodeVerifierInputPlan &plan)
    {
        if (pp_device_generation_)
            return pp_stage_runners_.back()->setMTPSpecVerifierInputPlan(plan);
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        bool all_success = true;
        for (auto &runner : participants)
        {
            if (!runner || !runner->setMTPSpecVerifierInputPlan(plan))
            {
                all_success = false;
            }
        }
        return all_success;
    }

    void RankOrchestrator::clearMTPSpecVerifierInputPlan()
    {
        if (pp_device_generation_)
        {
            pp_stage_runners_.back()->clearMTPSpecVerifierInputPlan();
            return;
        }
        auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        for (auto &runner : participants)
        {
            if (runner)
                runner->clearMTPSpecVerifierInputPlan();
        }
    }

    bool RankOrchestrator::supportsLogicalMTPVerifierBaseCheckpoint() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        for (const auto &runner : participants)
        {
            if (!runner || !runner->supportsLogicalMTPVerifierBaseCheckpoint())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::supportsMTPSpecStatePublication() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            return false;
        }

        for (const auto &runner : participants)
        {
            if (!runner || !runner->supportsMTPSpecStatePublication())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::supportsDeviceResidentMTPSpecStatePublication() const
    {
        /*
         * Single-child ranks are only a wrapper around the real runner, so keep
         * their resident publication capability identical to the child.
         */
        if (pp_stage_runners_.empty() &&
            device_runners_.size() == 1 &&
            device_runners_[0])
        {
            return device_runners_[0]->supportsDeviceResidentMTPSpecStatePublication();
        }

        // A prepared native pipeline owns the captured acceptance exchange and
        // each follower's local commit. It never mirrors the tail's controller.
        if (!pp_stage_runners_.empty())
            return pp_device_generation_ && pp_device_generation_->supportsMTPPublication();
        if (device_runners_.size() < 2)
        {
            return false;
        }
        if (!publishesFullVocabularyMTPLogits())
        {
            return false;
        }

        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->primaryDeviceId().is_gpu() &&
                       runner->supportsDeviceResidentMTPSpecStatePublication();
            });
    }

    bool RankOrchestrator::validateCurrentMirroredLocalTPOutcome(
        const DeviceSpeculativeOutcomeHandle &outcome,
        const char *operation,
        std::string *error) const
    {
        const bool mirrored_kind =
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredGreedy ||
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic;
        const bool complete_participant_set =
            rank_mirrored_child_outcomes_.size() == device_runners_.size();
        const bool primary_valid = rank_mirrored_primary_outcome_.valid();
        const bool exact_primary_identity =
            outcome.output_tokens_device ==
                rank_mirrored_primary_outcome_.output_tokens_device &&
            outcome.meta_device == rank_mirrored_primary_outcome_.meta_device &&
            outcome.device == rank_mirrored_primary_outcome_.device &&
            outcome.request_count ==
                rank_mirrored_primary_outcome_.request_count &&
            outcome.logical_verifier_rows_per_request ==
                rank_mirrored_primary_outcome_
                    .logical_verifier_rows_per_request &&
            outcome.physical_verifier_rows_per_request ==
                rank_mirrored_primary_outcome_
                    .physical_verifier_rows_per_request &&
            outcome.output_token_stride ==
                rank_mirrored_primary_outcome_.output_token_stride &&
            outcome.meta_stride == rank_mirrored_primary_outcome_.meta_stride &&
            outcome.stream == rank_mirrored_primary_outcome_.stream &&
            outcome.response_ready_event.get() ==
                rank_mirrored_primary_outcome_.response_ready_event.get() &&
            outcome.mtp_transaction.state.get() ==
                rank_mirrored_primary_outcome_.mtp_transaction.state.get() &&
            outcome.device_generation_controller_owned ==
                rank_mirrored_primary_outcome_
                    .device_generation_controller_owned &&
            outcome.sampling_mode ==
                rank_mirrored_primary_outcome_.sampling_mode &&
            outcome.local_tp_participant_complete ==
                rank_mirrored_primary_outcome_.local_tp_participant_complete;

        if (mirrored_kind &&
            rank_compact_outcome_valid_ &&
            rank_mirrored_child_outcomes_valid_ &&
            complete_participant_set &&
            outcome.valid() &&
            primary_valid &&
            exact_primary_identity)
        {
            return true;
        }

        if (error)
        {
            std::ostringstream message;
            message << (operation && operation[0] != '\0'
                            ? operation
                            : "mirrored LocalTP outcome consumer")
                    << " rejected a non-current mirrored verifier outcome: "
                    << "kind=" << static_cast<int>(rank_compact_outcome_kind_)
                    << " mirrored_kind=" << mirrored_kind
                    << " rank_valid=" << rank_compact_outcome_valid_
                    << " child_set_valid=" << rank_mirrored_child_outcomes_valid_
                    << " child_handles=" << rank_mirrored_child_outcomes_.size()
                    << " participants=" << device_runners_.size()
                    << " candidate_valid=" << outcome.valid()
                    << " primary_valid=" << primary_valid
                    << " exact_primary_identity=" << exact_primary_identity
                    << " candidate_tokens="
                    << static_cast<const void *>(outcome.output_tokens_device)
                    << " primary_tokens="
                    << static_cast<const void *>(
                           rank_mirrored_primary_outcome_.output_tokens_device)
                    << " candidate_meta="
                    << static_cast<const void *>(outcome.meta_device)
                    << " primary_meta="
                    << static_cast<const void *>(
                           rank_mirrored_primary_outcome_.meta_device)
                    << " candidate_stream=" << outcome.stream
                    << " primary_stream=" << rank_mirrored_primary_outcome_.stream
                    << " candidate_ready=" << outcome.response_ready_event.get()
                    << " primary_ready="
                    << rank_mirrored_primary_outcome_.response_ready_event.get()
                    << " candidate_local_complete="
                    << outcome.local_tp_participant_complete
                    << " primary_local_complete="
                    << rank_mirrored_primary_outcome_
                           .local_tp_participant_complete;
            *error = message.str();
        }
        return false;
    }

    bool RankOrchestrator::copyDeviceSpeculativeOutcomesToHostForDiagnostics(
        const DeviceSpeculativeOutcomeHandle &handle,
        DeviceSpeculativeVerifyBatchOutcome *outcomes)
    {
        const bool mirrored_local_tp_domain =
            pp_stage_runners_.empty() &&
            device_runners_.size() > 1 &&
            publishesFullVocabularyMTPLogits();
        if (mirrored_local_tp_domain)
        {
            std::string outcome_error;
            if (!outcomes ||
                !validateCurrentMirroredLocalTPOutcome(
                    handle,
                    "response materialization",
                    &outcome_error))
            {
                LOG_ERROR("[RankOrchestrator] "
                          << (outcomes
                                  ? outcome_error
                                  : "response materialization received a null outcome destination"));
                return false;
            }

            if (!device_runners_.front() ||
                !device_runners_.front()->copyDeviceSpeculativeOutcomesToHostForDiagnostics(
                    rank_mirrored_child_outcomes_.front(),
                    outcomes))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP response materialization failed in primary child after rank ownership validation: device="
                          << (device_runners_.front()
                                  ? device_runners_.front()->primaryDeviceId().toString()
                                  : "missing")
                          << " request_count=" << handle.request_count
                          << " output_stride=" << handle.output_token_stride
                          << " meta_stride=" << handle.meta_stride
                          << " local_complete="
                          << handle.local_tp_participant_complete);
                return false;
            }

            PerfStatsCollector::addCounter(
                "mtp",
                rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic
                    ? "rank_mirrored_localtp_stochastic_outcome_host_materializations"
                    : "rank_mirrored_localtp_greedy_outcome_host_materializations",
                1.0,
                "decode",
                "rank",
                {{"requests", std::to_string(handle.request_count)},
                 {"source", "primary_child_device_outcome"}});
            return true;
        }

        if (!outcomes ||
            !rank_compact_outcome_valid_ ||
            !handle.valid() ||
            handle.output_tokens_device != rank_compact_output_tokens_.data() ||
            handle.meta_device != rank_compact_output_meta_.data() ||
            handle.request_count != 1 ||
            handle.output_token_stride != rank_compact_output_token_stride_ ||
            handle.meta_stride != sampling_math::kSpeculativeBatchMetaCount)
        {
            return false;
        }

        /*
         * This bridge decodes the compact SamplingMath arrays that the rank
         * already used for publication planning.  It exists only to populate
         * response tokens and host diagnostics after live state has been
         * published; no row replay, model forward, or child mutation is hidden
         * behind this call.
         */
        if (!fillRankSpeculativeVerifyOutcomeFromMeta(
                rank_compact_output_tokens_,
                rank_compact_output_meta_,
                outcomes))
        {
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::Stochastic
                ? "rank_compact_stochastic_outcome_host_materializations"
                : "rank_compact_greedy_outcome_host_materializations",
            1.0,
            "decode",
            "rank",
            {{"requests", std::to_string(handle.request_count)},
             {"source", "rank_compact_sampling_math"}});
        return true;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        if (pp_device_generation_)
            return pp_device_generation_->publishMTPOutcome(request, error);
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        if (pp_stage_runners_.empty() &&
            device_runners_.size() == 1 &&
            device_runners_[0])
        {
            const bool ok =
                device_runners_[0]->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                    request,
                    error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!supportsDeviceResidentMTPSpecStatePublication())
        {
            return fail(
                "GPU LocalTP MTP resident publication requires mirrored full-head child verifier outcomes on every participant");
        }
        if (!request.valid())
        {
            return fail(
                "mirrored LocalTP MTP publication received an invalid device outcome request");
        }
        if (rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredGreedy ||
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic)
        {
            return publishMirroredLocalTPDeviceResidentMTPSpecStateBatch(
                request,
                error);
        }
        return fail(
            "GPU LocalTP MTP publication refuses rank-owned compact metadata; run mirrored child verifier reduction first");
    }

    bool RankOrchestrator::adoptMirroredLocalTPResidentLogicalStateMailboxes(
        int expected_request_count,
        const char *lifecycle,
        std::string *error)
    {
        const std::string lifecycle_name =
            lifecycle && lifecycle[0] != '\0' ? lifecycle : "unspecified";
        auto invalidate_and_fail = [&](const std::string &reason) -> bool
        {
            invalidateRankResidentLogicalStateAggregate(
                lifecycle_name.c_str(),
                reason.c_str());
            if (error)
                *error = reason;
            return false;
        };

        if (expected_request_count <= 0 || device_runners_.size() < 2)
        {
            return invalidate_and_fail(
                "resident mailbox adoption requires a non-empty multi-device request batch");
        }

        std::vector<DeviceResidentLogicalSequenceStateHandle> next_handles;
        next_handles.reserve(device_runners_.size());
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child)
            {
                return invalidate_and_fail(
                    "resident mailbox adoption lost participant " +
                    std::to_string(participant));
            }

            DeviceResidentLogicalSequenceStateHandle child_handle =
                child->deviceResidentLogicalSequenceState();
            if (!child_handle.valid())
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found no live child mailbox on participant " +
                    std::to_string(participant));
            }
            if (child_handle.request_count != expected_request_count)
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found request-count mismatch on participant " +
                    std::to_string(participant) + ": expected " +
                    std::to_string(expected_request_count) + ", got " +
                    std::to_string(child_handle.request_count));
            }
            if (child_handle.device != child->primaryDeviceId())
            {
                return invalidate_and_fail(
                    "resident mailbox adoption found cross-device ownership on participant " +
                    std::to_string(participant));
            }
            next_handles.push_back(child_handle);
        }

        rank_resident_child_logical_state_handles_.swap(next_handles);
        ++rank_resident_logical_state_epoch_;
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_mailbox_adoptions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(expected_request_count)},
             {"lifecycle", lifecycle_name},
             {"payload_owner", "child_device_mailboxes"}});
        return true;
    }

    bool RankOrchestrator::publishMirroredLocalTPDeviceResidentMTPSpecStateBatch(
        const DeviceSpeculativePublicationRequest &request,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                request.outcome,
                "mirrored LocalTP state publication",
                &outcome_error))
        {
            return fail(outcome_error);
        }
        if (!request.valid())
        {
            return fail("mirrored LocalTP MTP publication request has invalid capture geometry");
        }

        /*
         * Each child owns the verifier state, compact outcome, and exact stream
         * that produced both. Passing child-local handles preserves that
         * ownership while byte-identical mirrored inputs and deterministic
         * reducers provide the common accepted count and next-condition token.
         */
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child)
            {
                return fail(
                    "mirrored LocalTP MTP resident publication lost a participant");
            }

            DeviceSpeculativePublicationRequest child_request = request;
            child_request.outcome = rank_mirrored_child_outcomes_[i];
            std::string publish_error;
            if (!child->publishAcceptedMTPSpecStateBatchFromDeviceOutcome(
                    child_request,
                    &publish_error))
            {
                return fail(
                    "mirrored LocalTP MTP resident publication failed on participant " +
                    std::to_string(i) +
                    (publish_error.empty() ? std::string()
                                           : ": " + publish_error));
            }
        }

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request.requestCount(),
                "accepted_state_publication",
                &mailbox_error))
        {
            return fail(
                "mirrored LocalTP MTP resident publication could not adopt child logical-state mailboxes: " +
                mailbox_error);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            rank_compact_outcome_kind_ == RankCompactOutcomeKind::MirroredStochastic
                ? "rank_mirrored_localtp_stochastic_device_outcome_publications"
                : "rank_mirrored_localtp_greedy_device_outcome_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(request.requestCount())},
             {"logical_verifier_rows",
              std::to_string(request.logicalVerifierRowsPerRequest())},
             {"physical_verifier_rows",
              std::to_string(request.physicalVerifierRowsPerRequest())},
             {"implementation", "mirrored_child_device_publish"}});
        return true;
    }

    MTPVerifierRowCapability RankOrchestrator::mtpVerifierRowCapability() const
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        MTPVerifierRowCapability capability;
        if (participants.empty())
        {
            return capability;
        }

        bool initialized = false;
        for (const auto &runner : participants)
        {
            if (!runner)
            {
                return {};
            }
            const MTPVerifierRowCapability child =
                runner->mtpVerifierRowCapability();
            if (!initialized)
            {
                capability = child;
                initialized = true;
            }
            else
            {
                capability.intersectWith(child);
            }
        }
        return capability;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecState(
        const MTPSpecStepPlan &plan,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_spec_state_publication_total",
            "decode",
            "rank",
            {{"participants",
              std::to_string(!pp_stage_runners_.empty()
                                 ? pp_stage_runners_.size()
                                 : device_runners_.size())}});

        if (!pp_stage_runners_.empty())
        {
            /*
             * Pipeline publication is an all-stage local-state mutation:
             * earlier stages own their KV/GDN rows, while the final stage also
             * owns logits and terminal hidden.  Publish every stage to the same
             * common accepted count before the request continues.
             */
            std::vector<MTPSpecStepPlan> participant_plans(
                pp_stage_runners_.size(),
                plan);
            MTPSpecCommonStepPlan common_plan =
                coordinateMTPSpecCommonAcceptedPrefix(participant_plans);
            if (!common_plan.ok ||
                common_plan.clamped_steps.size() != pp_stage_runners_.size())
            {
                return fail(
                    std::string("PP MTP spec-state common-prefix coordination failed: ") +
                    (common_plan.ok
                         ? std::string("missing clamped stage plans")
                         : common_plan.error));
            }
            bool all_success = true;
            std::vector<std::string> child_errors(pp_stage_runners_.size());
            for (size_t i = 0; i < pp_stage_runners_.size(); ++i)
            {
                if (!pp_stage_runners_[i])
                {
                    child_errors[i] = "stage runner is unavailable";
                    all_success = false;
                    continue;
                }
                if (!pp_stage_runners_[i]->supportsMTPSpecStatePublication())
                {
                    child_errors[i] =
                        "stage does not support verifier-state publication";
                    all_success = false;
                    continue;
                }
                MTPSpecStepPlan stage_plan = common_plan.clamped_steps[i];
                stage_plan.publish_mtp_shifted_kv =
                    i + 1 == pp_stage_runners_.size();
                if (!pp_stage_runners_[i]->publishAcceptedMTPSpecState(
                        stage_plan,
                        &child_errors[i]))
                {
                    all_success = false;
                }
            }

            if (!all_success)
            {
                for (size_t i = 0; i < child_errors.size(); ++i)
                {
                    if (!child_errors[i].empty())
                    {
                        std::ostringstream msg;
                        msg << "PP MTP spec-state publication failed on stage "
                            << i << ": " << child_errors[i];
                        return fail(msg.str());
                    }
                }
                return fail("PP MTP spec-state publication failed on at least one stage");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mtp_spec_state_publications",
                1.0,
                "decode",
                "rank",
                {{"topology", "local_pp"},
                 {"participants", std::to_string(pp_stage_runners_.size())},
                 {"accepted_count",
                  std::to_string(common_plan.common_accepted_count)}});
            refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return true;
        }
        if (device_runners_.empty())
        {
            return fail("MTP spec-state publication requires at least one rank participant");
        }

        /*
         * Build one participant plan per child even though the current
         * rank-level verifier has a single accepted-count decision.  Keeping
         * this path on the shared common-prefix helper prevents the LocalTP,
         * NodeTP, PP, and routed-expert implementations from drifting
         * once participant-local accepted-state availability is introduced.
         */
        std::vector<MTPSpecStepPlan> participant_plans(
            device_runners_.size(),
            plan);
        MTPSpecCommonStepPlan common_plan =
            coordinateMTPSpecCommonAcceptedPrefix(participant_plans);
        if (!common_plan.ok ||
            common_plan.clamped_steps.size() != device_runners_.size())
        {
            return fail(
                std::string("MTP spec-state common-prefix coordination failed: ") +
                (common_plan.ok ? std::string("missing clamped participant plans")
                                : common_plan.error));
        }
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication participant " << i
                    << " is unavailable";
                return fail(msg.str());
            }
            if (!device_runners_[i]->supportsMTPSpecStatePublication())
            {
                std::ostringstream msg;
                msg << "MTP spec-state publication participant " << i
                    << " does not support verifier-state publication";
                return fail(msg.str());
            }
        }

        if (device_runners_.size() == 1)
        {
            const bool ok = device_runners_[0]->publishAcceptedMTPSpecState(
                common_plan.clamped_steps.front(),
                error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP spec-state publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_spec_state_publication_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &common_plan, &child_errors, kernel_phase, rocm_phase,
                 cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_publication_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " accepted_count="
                                  << common_plan.clamped_steps[i].accepted_count);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        device_runners_[i]->publishAcceptedMTPSpecState(
                            common_plan.clamped_steps[i],
                            &child_errors[i]);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_publication_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_spec_state_publication_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "publishAcceptedMTPSpecState",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::publishAcceptedMTPSpecState: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Device "
                          << r.worker_index << " publication failed: "
                          << child_errors[static_cast<size_t>(r.worker_index)]);
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "publishAcceptedMTPSpecState",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecState: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (!all_success)
        {
            for (size_t i = 0; i < child_errors.size(); ++i)
            {
                if (!child_errors[i].empty())
                {
                    std::ostringstream msg;
                    msg << "MTP spec-state publication failed on participant "
                        << i << ": " << child_errors[i];
                    return fail(msg.str());
                }
            }
            return fail("MTP spec-state publication failed on at least one participant");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mtp_spec_state_publications",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"accepted_count", std::to_string(common_plan.common_accepted_count)}});
        refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
        return true;
    }

    bool RankOrchestrator::publishAcceptedMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            if (error)
                *error = reason;
            LOG_ERROR("[RankOrchestrator] " << reason);
            return false;
        };

        const size_t participant_count =
            !pp_stage_runners_.empty()
                ? pp_stage_runners_.size()
                : device_runners_.size();

        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_spec_state_batch_publication_total",
            "decode",
            "rank",
            {{"participants", std::to_string(participant_count)},
             {"request_count", std::to_string(plans.request_count)}});

        if (!plans.ok)
        {
            return fail("MTP spec-state batch publication received an invalid plan batch: " +
                        plans.error);
        }
        if (plans.request_count <= 0 ||
            static_cast<int>(plans.steps.size()) != plans.request_count)
        {
            return fail("MTP spec-state batch publication received inconsistent request count");
        }

        auto build_common_batches =
            [&](size_t count,
                bool local_pp,
                std::vector<MTPSpecStepPlanBatch> &out_batches) -> bool
        {
            out_batches.assign(count, plans);

            /*
             * Clamp each logical request independently.  Every participant sees
             * the same request batch today, but keeping the per-request
             * common-prefix step here prevents future per-participant state
             * availability from mutating live state divergently.
             */
            for (size_t step_idx = 0; step_idx < plans.steps.size(); ++step_idx)
            {
                std::vector<MTPSpecStepPlan> participant_steps(
                    count,
                    plans.steps[step_idx]);
                MTPSpecCommonStepPlan common_plan =
                    coordinateMTPSpecCommonAcceptedPrefix(participant_steps);
                if (!common_plan.ok ||
                    common_plan.clamped_steps.size() != count)
                {
                    return fail(
                        std::string("MTP spec-state batch common-prefix coordination failed: ") +
                        (common_plan.ok
                             ? std::string("missing clamped participant plans")
                             : common_plan.error));
                }
                for (size_t participant = 0; participant < count; ++participant)
                {
                    MTPSpecStepPlan clamped_step =
                        common_plan.clamped_steps[participant];
                    if (local_pp)
                    {
                        /*
                         * Non-final PP stages own main per-layer state only.
                         * The final stage owns the sidecar shifted KV cache.
                         */
                        clamped_step.publish_mtp_shifted_kv =
                            participant + 1 == count;
                    }
                    out_batches[participant].steps[step_idx] = clamped_step;
                }
            }
            return true;
        };

        if (!pp_stage_runners_.empty())
        {
            std::vector<MTPSpecStepPlanBatch> stage_batches;
            if (!build_common_batches(
                    pp_stage_runners_.size(),
                    /*local_pp=*/true,
                    stage_batches))
            {
                return false;
            }

            bool all_success = true;
            std::vector<std::string> child_errors(pp_stage_runners_.size());
            for (size_t i = 0; i < pp_stage_runners_.size(); ++i)
            {
                if (!pp_stage_runners_[i])
                {
                    child_errors[i] = "stage runner is unavailable";
                    all_success = false;
                    continue;
                }
                if (!grouped_decode_equivalent_spec_publication_scope_ &&
                    !pp_stage_runners_[i]->supportsMTPSpecStatePublication())
                {
                    child_errors[i] =
                        "stage does not support verifier-state publication";
                    all_success = false;
                    continue;
                }
                const bool published =
                    grouped_decode_equivalent_spec_publication_scope_
                        ? pp_stage_runners_[i]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                              stage_batches[i],
                              &child_errors[i])
                        : pp_stage_runners_[i]->publishAcceptedMTPSpecStateBatch(
                              stage_batches[i],
                              &child_errors[i]);
                if (!published)
                {
                    all_success = false;
                }
            }

            if (!all_success)
            {
                for (size_t i = 0; i < child_errors.size(); ++i)
                {
                    if (!child_errors[i].empty())
                    {
                        std::ostringstream msg;
                        msg << "PP MTP spec-state batch publication failed on stage "
                            << i << ": " << child_errors[i];
                        return fail(msg.str());
                    }
                }
                return fail("PP MTP spec-state batch publication failed on at least one stage");
            }

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mtp_spec_state_batch_publications",
                1.0,
                "decode",
                "rank",
                {{"topology", "local_pp"},
                 {"participants", std::to_string(pp_stage_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return true;
        }

        if (device_runners_.empty())
        {
            return fail("MTP spec-state batch publication requires at least one rank participant");
        }

        std::vector<MTPSpecStepPlanBatch> participant_batches;
        if (!build_common_batches(
                device_runners_.size(),
                /*local_pp=*/false,
                participant_batches))
        {
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                std::ostringstream msg;
                msg << "MTP spec-state batch publication participant " << i
                    << " is unavailable";
                return fail(msg.str());
            }
            if (!grouped_decode_equivalent_spec_publication_scope_ &&
                !device_runners_[i]->supportsMTPSpecStatePublication())
            {
                std::ostringstream msg;
                msg << "MTP spec-state batch publication participant " << i
                    << " does not support verifier-state publication";
                return fail(msg.str());
            }
        }

        if (device_runners_.size() == 1)
        {
            const bool ok =
                grouped_decode_equivalent_spec_publication_scope_
                    ? device_runners_[0]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                          participant_batches.front(),
                          error)
                    : device_runners_[0]->publishAcceptedMTPSpecStateBatch(
                          participant_batches.front(),
                          error);
            if (ok)
                refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
            return ok;
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP spec-state batch publication failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_spec_state_batch_publication_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            tp_worker_pool_->dispatch(
                [this, &participant_batches, &child_errors, kernel_phase,
                 rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_batch_publication_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " request_count="
                                  << participant_batches[i].request_count);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        (grouped_decode_equivalent_spec_publication_scope_
                             ? device_runners_[i]->publishGroupedDecodeEquivalentMTPSpecStateBatch(
                                   participant_batches[i],
                                   &child_errors[i])
                             : device_runners_[i]->publishAcceptedMTPSpecStateBatch(
                                   participant_batches[i],
                                   &child_errors[i]));

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_spec_state_batch_publication_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_spec_state_batch_publication_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_count", std::to_string(plans.request_count)}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "publishAcceptedMTPSpecStateBatch",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::publishAcceptedMTPSpecStateBatch: requesting TP context abort after worker timeout");
                    tp_ctx_->requestAbort();
                }
                continue;
            }

            if (r.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                }
                continue;
            }

            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Device "
                          << r.worker_index << " publication failed: "
                          << child_errors[static_cast<size_t>(r.worker_index)]);
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "publishAcceptedMTPSpecStateBatch",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::publishAcceptedMTPSpecStateBatch: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (!all_success)
        {
            for (size_t i = 0; i < child_errors.size(); ++i)
            {
                if (!child_errors[i].empty())
                {
                    std::ostringstream msg;
                    msg << "MTP spec-state batch publication failed on participant "
                        << i << ": " << child_errors[i];
                    return fail(msg.str());
                }
            }
            return fail("MTP spec-state batch publication failed on at least one participant");
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mtp_spec_state_batch_publications",
            1.0,
            "decode",
            "rank",
            {{"topology", "local_tp"},
             {"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(plans.request_count)}});
        refreshAggregateSequenceStateFromPrimaryRunnerAfterPublication();
        return true;
    }
} // namespace llaminar2
