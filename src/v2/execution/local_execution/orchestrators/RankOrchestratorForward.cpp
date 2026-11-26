/**
 * @file RankOrchestratorForward.cpp
 * @brief RankOrchestrator PP/TP forward and batch execution.
 */
#include "RankOrchestratorInternal.h"

namespace llaminar2
{

    bool RankOrchestrator::publishMainLogitsBatchSamplesToDeviceResidentState(
        int request_count,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->publishMainLogitsBatchSamplesToDeviceResidentState(
                request_count,
                params,
                stochastic_position_seeds);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->publishMainLogitsBatchSamplesToDeviceResidentState(
                request_count,
                params,
                stochastic_position_seeds);
        }
        if (request_count <= 0 ||
            device_runners_.size() < 2 ||
            !publishesFullVocabularyMTPLogits() ||
            !supportsDeviceResidentMTPSpecStatePublication())
        {
            return false;
        }

        /*
         * Every child must execute the sampler. Besides selecting the response
         * token, DeviceGraphOrchestrator publishes that child's initial target
         * position, sequence length, condition token, and accepted-count row
         * into its resident mailbox. Sampling only child zero and broadcasting
         * the token would leave the remaining publication transactions without
         * authoritative device-owned logical state.
         *
         * No token vector is materialized for host comparison. Mirrored
         * participants execute the same immutable sampling policy and seeds;
         * rank-level generation later authenticates their dispatch tickets and
         * terminal ledgers before surfacing the final response.
         */
        for (size_t child_index = 0;
             child_index < device_runners_.size();
             ++child_index)
        {
            const IInferenceRunner *child =
                device_runners_[child_index].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                return false;
            }
        }

        /*
         * This is a request boundary, not setup. Creating one std::async
         * thread per participant here left a new glibc allocator arena behind
         * on long-running servers and steadily raised resident host memory.
         * Reuse the same persistent LocalTP workers that own graph replay so
         * profiler/device thread state and allocator caches remain bounded by
         * participant count rather than request count.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                {
                    LOG_WARN(
                        "[TPWorkerPool] Mirrored LocalTP request-batch "
                        "sampling failure detected - aborting collective backend");
                    tp_ctx_->requestAbort();
                });
            }
        }
        if (tp_worker_pool_->numWorkers() != device_runners_.size())
        {
            LOG_ERROR(
                "[RankOrchestrator] Mirrored LocalTP sampling worker topology "
                "does not match the participant count");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             request_count,
             &params,
             stochastic_position_seeds,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](std::size_t child_index) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);
                const DeviceId device =
                    device_runners_[child_index]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return device_runners_[child_index]
                    ->publishMainLogitsBatchSamplesToDeviceResidentState(
                        request_count,
                        params,
                        stochastic_position_seeds);
            });

        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        const auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (const auto &result : results)
        {
            if (!result.completed)
            {
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "publishMainLogitsBatchSamplesToDeviceResidentState",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                return false;
            }
            if (result.exception)
                std::rethrow_exception(result.exception);
            if (!result.success)
            {
                LOG_ERROR(
                    "[RankOrchestrator] Mirrored LocalTP request-batch "
                    "prefill sampling failed on child "
                    << result.worker_index);
                return false;
            }
        }

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request_count,
                "request_batch_prefill_sample",
                &mailbox_error))
        {
            LOG_ERROR("[RankOrchestrator] Mirrored LocalTP request-batch "
                      "prefill could not adopt child logical-state mailboxes: "
                      << mailbox_error);
            return false;
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_request_batch_prefill_publications",
            static_cast<double>(request_count),
            /*phase=*/"decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"sampling", params.is_greedy() ? "greedy" : "stochastic"},
             {"logical_state_owner", "child_device_mailboxes"},
             {"host_token_materializations", "0"},
             {"boundary", "first_grouped_verifier_transaction"}});
        return true;
    }

    bool RankOrchestrator::requiresMPICoordinatedDecodeSampling(
        const SamplingParams &params) const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->requiresMPICoordinatedDecodeSampling(params);
        }

        /*
         * This method must mirror RankOrchestrator::sampleGreedyOnDevice() /
         * sampleOnDevice(), not a child runner's latent capability.  The TP path
         * samples through DeviceSampler over local child runners or falls back
         * to rank-local gathered logits; it does not call
         * child->sampleGreedyOnDevice(), so worker ranks must not enter child
         * sampling collectives on its behalf.
         */
        return false;
    }

    size_t RankOrchestrator::mainHostLogitsStorageBytes() const
    {
        return logits_gatherer_ ? logits_gatherer_->bufferNumel() * sizeof(float) : 0u;
    }

    const float *RankOrchestrator::logits() const
    {
        PerfStatsCollector::addCounter(
            "sampling",
            "host_logits_access",
            1.0,
            {},
            primaryDeviceId().toString(),
            {{"source", "rank_orchestrator_logits"},
             {"mode", std::to_string(static_cast<int>(mode_))},
             {"tp_children", std::to_string(device_runners_.size())},
             {"pp_children", std::to_string(pp_stage_runners_.size())},
             {"gatherer_allocated",
              (logits_gatherer_ && logits_gatherer_->isAllocated()) ? "true" : "false"}});

        if (!ownsMainForwardLogits())
        {
            throw std::logic_error(
                "RankOrchestrator::logits: nested non-head PP stage publishes "
                "hidden state and has no logits boundary");
        }

        if (primaryDeviceId().is_gpu())
        {
            if (!last_logits_forward_phase_.has_value())
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: GPU host logits were requested before a forward transaction");
            }

            const bool host_observation_disabled =
                *last_logits_forward_phase_ == LogitsForwardPhase::Decode
                    ? skip_logits_gather_decode_
                    : skip_logits_gather_prefill_;
            if (host_observation_disabled)
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: GPU logits are device-owned for the active forward phase; use the device sampler/result publication API");
            }
        }

        // For PP mode: return combined logits (copied from final stage)
        if (mode_ == ParallelismMode::PP || mode_ == ParallelismMode::TP_PP)
        {
            const float *gathered = logits_gatherer_ ? logits_gatherer_->data() : nullptr;
            if (gathered)
                return gathered;
            if (primaryDeviceId().is_gpu())
            {
                throw std::logic_error(
                    "RankOrchestrator::logits: the GPU pipeline did not publish current host logits");
            }
            return nullptr;
        }

        // For TP mode: return combined logits if available (multi-device)
        if (logits_gatherer_ && logits_gatherer_->data() && device_runners_.size() > 1)
        {
            return logits_gatherer_->data();
        }

        // For single device, return primary device's logits
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->logits();
        }

        if (primaryDeviceId().is_gpu())
        {
            throw std::logic_error(
                "RankOrchestrator::logits: the active GPU forward did not publish current host logits");
        }

        return nullptr;
    }

    DeviceId RankOrchestrator::primaryDeviceId() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->primaryDeviceId();
        }
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->primaryDeviceId();
        }
        return DeviceId::cpu();
    }

    bool RankOrchestrator::forwardMTP(int32_t draft_condition_token)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_forward_mtp_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * PP verifier replay stays pipeline-wide through forwardPP(), but
             * the MTP sidecar consumes the terminal hidden row and produces MTP
             * logits on the final stage.  Running it on earlier stages would be
             * nonsensical: they do not own final norm, LM head, or sidecar
             * logits.  This is intentionally not TP fan-out.
             */
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_pp_final_stage_calls",
                1.0,
                "decode",
                "rank");
            return pp_sidecar->forwardMTP(draft_condition_token);
        }
        if (device_runners_.empty())
        {
            return false;
        }

        if (device_runners_.size() == 1)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_single_participant_calls",
                1.0,
                "decode",
                "rank");
            return device_runners_[0] && device_runners_[0]->forwardMTP(draft_condition_token);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP worker failure detected — aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_forward_mtp_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, draft_condition_token, kernel_phase, rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
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
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << draft_condition_token);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->forwardMTP(draft_condition_token);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_leave"
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
                "rank_forward_mtp_collect",
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
                LOG_ERROR("RankOrchestrator::forwardMTP: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "forwardMTP",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::forwardMTP: requesting TP context abort after worker timeout");
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
                LOG_ERROR("RankOrchestrator::forwardMTP: Device "
                          << r.worker_index << " MTP forward failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTP",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTP: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::forwardMTPForDeviceSampling(int32_t draft_condition_token)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardMTPForDeviceSampling(draft_condition_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPForDeviceSampling(
                draft_condition_token);
        }
        if (!supportsMTPSidecarLogitsStreamHandoff())
        {
            LOG_ERROR("[RankOrchestrator] LocalTP MTP device-sampling sidecar requires stream handoff on every participant");
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
                    LOG_WARN("[TPWorkerPool] MTP device-sampling worker failure detected - aborting collective backend");
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
             draft_condition_token,
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

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]->forwardMTPForDeviceSampling(
                           draft_condition_token);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Device "
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
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_sampling_calls",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
        }
        return all_success;
    }

    bool RankOrchestrator::supportsChainedMTPDrafts() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsChainedMTPDrafts();
        }

        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->supportsChainedMTPDrafts())
            {
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::forwardMTPFromLastDraft(int32_t draft_condition_token, int position_id)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_forward_mtp_chained_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            if (!pp_sidecar->supportsChainedMTPDrafts())
            {
                LOG_ERROR("[RankOrchestrator] Final PP stage does not support chained MTP sidecar execution");
                return false;
            }
            return pp_sidecar->forwardMTPFromLastDraft(
                draft_condition_token,
                position_id);
        }
        if (device_runners_.empty())
        {
            LOG_ERROR("[RankOrchestrator] Chained MTP drafts require at least one rank participant");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i] ||
                !device_runners_[i]->supportsChainedMTPDrafts())
            {
                LOG_ERROR("[RankOrchestrator] Chained MTP draft participant "
                          << i << " does not support chained sidecar execution");
                return false;
            }
        }

        if (device_runners_.size() == 1)
        {
            return device_runners_[0]->forwardMTPFromLastDraft(
                draft_condition_token,
                position_id);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] Chained MTP worker failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_forward_mtp_chained_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, draft_condition_token, position_id, kernel_phase,
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
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_chained_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << draft_condition_token
                                  << " position=" << position_id);
                    }

                    const bool ok =
                        device_runners_[i] &&
                        device_runners_[i]->forwardMTPFromLastDraft(
                            draft_condition_token,
                            position_id);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=forward_mtp_chained_leave"
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
                "rank_forward_mtp_chained_collect",
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "forwardMTPFromLastDraft",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::forwardMTPFromLastDraft: requesting TP context abort after worker timeout");
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
                LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Device "
                          << r.worker_index << " chained MTP forward failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromLastDraft",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromLastDraft: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromLastDraftForDeviceSampling(
        int32_t draft_condition_token,
        int position_id)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->forwardMTPFromLastDraftForDeviceSampling(
                draft_condition_token,
                position_id);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromLastDraftForDeviceSampling(
                draft_condition_token,
                position_id);
        }
        return false;
    }

    bool RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
        int draft_sample_slot,
        int position_offset)
    {
        if (IInferenceRunner *tail = finalPPSidecarRunner())
        {
            // Draft chaining stays on the tail. The captured main-verifier
            // row exchange, not a borrowed tail pointer, later feeds the head.
            return pp_device_generation_ && tail->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                draft_sample_slot, position_offset);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                draft_sample_slot,
                position_offset);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            draft_sample_slot < 0 ||
            draft_sample_slot >= rank_stochastic_slot_capacity_ ||
            position_offset < 0)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-draft sidecar requires staged child draft slots and live device positions on every participant");
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
                    LOG_WARN("[TPWorkerPool] MTP device-draft worker failure detected - aborting collective backend");
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
             draft_sample_slot,
             position_offset,
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

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]
                           ->forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling(
                           draft_sample_slot,
                           position_offset);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Device "
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
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceDraftAtLivePositionForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_draft_slot_calls",
                1.0,
                 "decode",
                 "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(draft_sample_slot)},
                 {"position_offset", std::to_string(position_offset)},
                 {"position_owner", "child_live_device_kv_counts"}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
        int target_sample_slot)
    {
        if (IInferenceRunner *tail = finalPPSidecarRunner())
            return pp_device_generation_ && tail->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(target_sample_slot);
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                    target_sample_slot);
        }
        if (!supportsMTPDeviceDraftTokenInput() ||
            target_sample_slot < 0 ||
            target_sample_slot >= rank_stochastic_slot_capacity_)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP device-target sidecar requires staged child target slots and live device positions on every participant");
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
                    LOG_WARN("[TPWorkerPool] MTP device-target worker failure detected - aborting collective backend");
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

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return device_runners_[i] &&
                       device_runners_[i]
                           ->forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling(
                               target_sample_slot);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        bool worker_timeout = false;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Device "
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
                    first_exception_device = r.worker_index;
                }
                continue;
            }
            if (!r.success)
            {
                LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Device "
                          << r.worker_index << " failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::forwardMTPFromDeviceTargetAtLivePositionForDeviceSampling: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_forward_mtp_device_target_slot_calls",
                1.0,
                 "decode",
                 "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(target_sample_slot)},
                 {"position_owner", "child_live_device_kv_counts"}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index)
    {
        if (IInferenceRunner *tail = finalPPSidecarRunner())
            return pp_device_generation_ && tail->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                logical_state, request_index);
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                logical_state,
                request_index);
        }
        if (!logical_state.valid() ||
            logical_state.target_positions_device !=
                rank_resident_logical_state_marker_.data() ||
            logical_state.live_state_epoch !=
                rank_resident_logical_state_epoch_ ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Resident logical-state sidecar received a non-owned LocalTP mailbox");
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
                    LOG_WARN("[TPWorkerPool] resident logical-state sidecar failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        std::vector<std::string> child_errors(device_runners_.size());
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        tp_worker_pool_->dispatch(
            [this,
             request_index,
             &child_errors,
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

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                const bool ok =
                    device_runners_[i] &&
                    device_runners_[i]->forwardMTPFromDeviceResidentLogicalStateForDeviceSampling(
                        rank_resident_child_logical_state_handles_[i],
                        request_index);
                if (!ok)
                    child_errors[i] = "resident logical-state sidecar failed";
                return ok;
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &r : results)
        {
            if (!r.completed || !r.success)
                all_success = false;
            if (r.exception && !first_exception)
            {
                first_exception = r.exception;
                first_exception_device = r.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident logical-state sidecar rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_resident_logical_state_sidecar_prelaunches",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)}});
        }
        return all_success;
    }

    bool RankOrchestrator::forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_draft_slot,
        int slot_stride)
    {
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batched MTP is a LocalTP operation and cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                    logical_state,
                    request_batch,
                    first_draft_slot,
                    slot_stride);
        }

        return dispatchMirroredLocalTPResidentMTPRequestBatch(
            logical_state,
            request_batch,
            first_draft_slot,
            slot_stride,
            "forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots",
            "resident_logical_state",
            [request_batch,
             first_draft_slot,
             slot_stride](
                IInferenceRunner &child,
                const DeviceResidentLogicalSequenceStateHandle &child_state)
            {
                return child
                    .forwardMTPBatchFromDeviceResidentLogicalStateAndSampleGreedyToDeviceDraftSlots(
                        child_state,
                        request_batch,
                        first_draft_slot,
                        slot_stride);
            });
    }

    bool RankOrchestrator::advanceMTPRequestBatchConditionOnDevice(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        const SamplingParams &params,
        const uint64_t *stochastic_position_seeds)
    {
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->advanceMTPRequestBatchConditionOnDevice(
                logical_state,
                request_batch,
                params,
                stochastic_position_seeds);
        }

        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance failed: "
                      << reason);
            return false;
        };
        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (device_runners_.size() < 2 || !tp_ctx_ ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count != request_batch ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            return fail("rank or child logical-state mailbox is stale or incomplete");
        }
        if (tp_ctx_->backend() != CollectiveBackendType::NCCL &&
            tp_ctx_->backend() != CollectiveBackendType::RCCL)
        {
            return fail("LocalTP condition advance requires NCCL/RCCL collectives");
        }
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            if (!device_runners_[participant] ||
                !rank_resident_child_logical_state_handles_[participant].valid() ||
                rank_resident_child_logical_state_handles_[participant]
                        .request_count != request_batch)
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " has no matching resident mailbox");
            }
        }

        // This transaction executes the main model before its sidecar. Keep
        // source admission at the rank boundary, before concurrent child work.
        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            tp_worker_pool_->setFailureCallback([this]()
                                                {
                LOG_WARN("[TPWorkerPool] resident request-batch condition advance failed; aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();

        /*
         * Main and sidecar graphs contain LocalTP collectives, so every child
         * must enter the grouped condition transition concurrently. Request
         * rows remain grouped inside each participant; the worker pool is only
         * the device-participant fanout.
         */
        tp_worker_pool_->dispatch(
            [this,
             request_batch,
             &params,
             stochastic_position_seeds,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
                {
                    LOG_INFO(
                        "[MTPConditionGraphContract] event=participant_enter"
                        << " participant=" << participant
                        << " device=" << device_id.toString()
                        << " requests=" << request_batch);
                }
                const bool success = device_runners_[participant]
                    ->advanceMTPRequestBatchConditionOnDevice(
                        rank_resident_child_logical_state_handles_[participant],
                        request_batch,
                        params,
                        stochastic_position_seeds);
                if (debugEnv().runtime_debug.mtp_condition_graph_contract_trace)
                {
                    LOG_INFO(
                        "[MTPConditionGraphContract] event=participant_leave"
                        << " participant=" << participant
                        << " device=" << device_id.toString()
                        << " requests=" << request_batch
                        << " success=" << (success ? "true" : "false"));
                }
                return success;
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_participant = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_participant = result.worker_index;
                }
                continue;
            }
            all_success = all_success && result.success;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "advanceMTPRequestBatchConditionOnDevice",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Resident request-batch condition advance rethrowing exception from participant "
                      << first_exception_participant);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        std::string mailbox_error;
        if (!adoptMirroredLocalTPResidentLogicalStateMailboxes(
                request_batch,
                "request_batch_condition_advance",
                &mailbox_error))
        {
            return fail("could not adopt advanced child mailboxes: " + mailbox_error);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_request_batch_condition_advances",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"sampling", params.is_greedy() ? "greedy" : "stochastic"},
             {"logical_state_owner", "child_device_mailboxes"}});
        source_attempt.keep=true;
        return true;
    }

    bool RankOrchestrator::forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_condition_slot,
        int condition_slot_stride,
        int position_offset,
        int first_draft_slot,
        int draft_slot_stride)
    {
        const int64_t last_condition_slot =
            static_cast<int64_t>(first_condition_slot) +
            static_cast<int64_t>(request_batch - 1) *
                static_cast<int64_t>(condition_slot_stride);
        if (request_batch <= 0 ||
            first_condition_slot < 0 ||
            condition_slot_stride <= 0 ||
            last_condition_slot < 0 ||
            last_condition_slot >= rank_stochastic_slot_capacity_ ||
            position_offset <= 0)
        {
            LOG_ERROR("[RankOrchestrator] Chained resident request-batched MTP received invalid source-slot or depth geometry"
                      << " request_batch=" << request_batch
                      << " first_condition_slot=" << first_condition_slot
                      << " condition_slot_stride=" << condition_slot_stride
                      << " position_offset=" << position_offset);
            return false;
        }
        if (finalPPSidecarRunner())
        {
            LOG_ERROR("[RankOrchestrator] Chained resident request-batched MTP is a LocalTP operation and cannot cross a LocalPP sidecar boundary");
            return false;
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                    logical_state,
                    request_batch,
                    first_condition_slot,
                    condition_slot_stride,
                    position_offset,
                    first_draft_slot,
                    draft_slot_stride);
        }

        return dispatchMirroredLocalTPResidentMTPRequestBatch(
            logical_state,
            request_batch,
            first_draft_slot,
            draft_slot_stride,
            "forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots",
            "device_draft_slots",
            [request_batch,
             first_condition_slot,
             condition_slot_stride,
             position_offset,
             first_draft_slot,
             draft_slot_stride](
                IInferenceRunner &child,
                const DeviceResidentLogicalSequenceStateHandle &child_state)
            {
                return child
                    .forwardMTPBatchFromDeviceDraftSlotsAndSampleGreedyToDeviceDraftSlots(
                        child_state,
                        request_batch,
                        first_condition_slot,
                        condition_slot_stride,
                        position_offset,
                        first_draft_slot,
                        draft_slot_stride);
            });
    }

    bool RankOrchestrator::dispatchLocalTPMTPMainCondition(
        const char *operation_name,
        LocalTPConditionEffect effect,
        const std::function<bool(IInferenceRunner &, size_t)> &child_operation)
    {
        const std::string operation =
            operation_name && operation_name[0] != '\0'
                ? operation_name
                : "mtp_main_condition";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("RankOrchestrator::" << operation << ": " << reason);
            return false;
        };

        if (!child_operation)
            return fail("missing participant operation");
        if (effect != LocalTPConditionEffect::PublishInput &&
            effect != LocalTPConditionEffect::ForwardMainRow)
            return fail("unknown main-condition effect");
        if (device_runners_.size() < 2)
            return fail("requires at least two local participants");
        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            if (!device_runners_[participant] ||
                !device_runners_[participant]->primaryDeviceId().is_gpu())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " is missing or is not GPU-backed");
            }
        }

        // The retained buffer is storage, not proof that this transaction has
        // published a value. Invalidate before fan-out, including partial
        // failure; input-only publication has not consumed a model row.
        if (effect == LocalTPConditionEffect::ForwardMainRow && logits_gatherer_)
            logits_gatherer_->invalidate();
        if (effect == LocalTPConditionEffect::ForwardMainRow &&
            ownsMainForwardLogits() && !skip_logits_gather_decode_ &&
            (!logits_gatherer_ || !logits_gatherer_->isAllocated()))
            return fail("requested host decode observation has no retained output owner");

        // Prefix restoration can skip ordinary forwardTP admission entirely.
        // The parent must arm every original channel before either child can
        // enter its main-model prelude. Merely publishing a pending input has
        // no source-lifetime effect. Repeated forwards reuse the same epoch;
        // failed or throwing forwards cancel the original group through RAII.
        std::optional<RetainedSourceAdmission::Attempt> source_attempt;
        if (effect == LocalTPConditionEffect::ForwardMainRow)
        {
            prepareRetainedSourceAdmission();
            source_attempt.emplace(*this,retained_source_admission_,__func__);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            tp_worker_pool_->setFailureCallback([this, operation]()
                                                {
                LOG_WARN("[TPWorkerPool] " << operation
                         << " failure detected - aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             child_operation,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return child_operation(
                    *device_runners_[participant],
                    participant);
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_device = result.worker_index;
                }
                continue;
            }
            if (!result.success)
                all_success = false;
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                operation.c_str(),
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR(
                "RankOrchestrator::" << operation
                << ": rethrowing participant exception from device "
                << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success &&
            effect == LocalTPConditionEffect::ForwardMainRow &&
            ownsMainForwardLogits() && logits_gatherer_ &&
            logits_gatherer_->needsGather(LogitsForwardPhase::Decode))
        {
            // Diagnostic host observation uses the original event-consuming
            // gather API and exactly one live vocabulary row. Device-owned
            // generation keeps decode gathering disabled and never copies here.
            if (!logits_gatherer_->gather(device_runners_, 1, vocab_size()))
                return fail("failed to publish current main-condition host logits");
            PerfStatsCollector::addCounter(
                "sampling",
                "rank_resident_condition_host_logits_publications",
                1.0, "decode", "rank",
                {{"operation", operation},
                 {"participants", std::to_string(device_runners_.size())},
                 {"rows", "1"}});
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_device_resident_main_condition_advances",
                1.0,
                "decode",
                "rank",
                {{"operation", operation},
                 {"participants", std::to_string(device_runners_.size())},
                 {"input_owner", "participant_device_logical_state"}});
        }
        if (source_attempt)
            source_attempt->keep=all_success;
        return all_success;
    }

    bool RankOrchestrator::dispatchMirroredLocalTPResidentMTPRequestBatch(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_batch,
        int first_draft_slot,
        int draft_slot_stride,
        const char *operation_name,
        const char *source_name,
        const std::function<bool(
            IInferenceRunner &,
            const DeviceResidentLogicalSequenceStateHandle &)> &child_operation)
    {
        const std::string operation =
            operation_name && operation_name[0] != '\0'
                ? operation_name
                : "resident_request_batch_mtp";
        const std::string source =
            source_name && source_name[0] != '\0'
                ? source_name
                : "unspecified";
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("RankOrchestrator::" << operation << ": " << reason);
            return false;
        };

        const int64_t last_draft_slot =
            static_cast<int64_t>(first_draft_slot) +
            static_cast<int64_t>(request_batch - 1) *
                static_cast<int64_t>(draft_slot_stride);
        if (request_batch <= 0 ||
            first_draft_slot < 0 ||
            draft_slot_stride <= 0 ||
            last_draft_slot < 0 ||
            last_draft_slot >= rank_stochastic_slot_capacity_)
        {
            return fail(
                "invalid destination-slot geometry: request_batch=" +
                std::to_string(request_batch) +
                " first_draft_slot=" + std::to_string(first_draft_slot) +
                " draft_slot_stride=" + std::to_string(draft_slot_stride));
        }
        if (!child_operation)
            return fail("missing grouped child operation");
        if (device_runners_.size() < 2 || !tp_ctx_)
            return fail("requires at least two LocalTP participants and a live collective context");
        if (tp_ctx_->backend() != CollectiveBackendType::NCCL &&
            tp_ctx_->backend() != CollectiveBackendType::RCCL)
        {
            return fail(
                std::string(
                    "requires NCCL/RCCL device-slot publication, got backend=") +
                collectiveBackendTypeToString(tp_ctx_->backend()));
        }
        if (!publishesFullVocabularyMTPLogits() ||
            !supportsMTPDeviceDraftTokenInput() ||
            !supportsDeviceStochasticMTPVerification())
        {
            return fail(
                "requires mirrored full-vocabulary MTP heads and device-resident draft/verifier support on every participant");
        }

        /*
         * The rank handle is deliberately opaque. Comparing it with a freshly
         * materialized aggregate checks every pointer, stream, event, epoch, and
         * request-count field without dereferencing child device memory.
         */
        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (!current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count != request_batch)
        {
            return fail(
                "received a stale, foreign, or shape-mismatched rank logical-state mailbox");
        }
        if (rank_resident_child_logical_state_handles_.size() !=
            device_runners_.size())
        {
            return fail("rank aggregate no longer has one child mailbox per participant");
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            const DeviceResidentLogicalSequenceStateHandle &child_state =
                rank_resident_child_logical_state_handles_[participant];
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits() ||
                !child->supportsMTPDeviceDraftTokenInput() ||
                !child->supportsDeviceStochasticMTPVerification())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " does not expose the mirrored device-resident MTP contract");
            }
            if (!child_state.valid() ||
                child_state.request_count != request_batch ||
                child_state.device != child->primaryDeviceId())
            {
                return fail(
                    "participant " + std::to_string(participant) +
                    " has a stale, foreign, or shape-mismatched child mailbox");
            }
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            tp_worker_pool_->setFailureCallback([this, operation]()
                                                {
                LOG_WARN("[TPWorkerPool] " << operation
                         << " failure detected - aborting collective backend");
                if (tp_ctx_)
                    tp_ctx_->requestAbort(); });
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();

        /*
         * The worker callback is participant-parallel and invokes exactly one
         * grouped child operation. Request rows remain a batch inside the child
         * graph; there is intentionally no rank-side loop over sidecar rows.
         */
        tp_worker_pool_->dispatch(
            [this,
             child_operation,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](size_t participant) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                const DeviceId device_id =
                    device_runners_[participant]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);
                return child_operation(
                    *device_runners_[participant],
                    rank_resident_child_logical_state_handles_[participant]);
            });

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_participant = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (auto &result : results)
        {
            if (!result.completed)
            {
                LOG_ERROR("RankOrchestrator::" << operation
                          << ": participant " << result.worker_index
                          << " did not complete");
                worker_timeout = true;
                all_success = false;
                continue;
            }
            if (result.exception)
            {
                all_success = false;
                if (!first_exception)
                {
                    first_exception = result.exception;
                    first_exception_participant = result.worker_index;
                }
                continue;
            }
            if (!result.success)
            {
                LOG_ERROR("RankOrchestrator::" << operation
                          << ": participant " << result.worker_index
                          << " rejected the grouped sidecar");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                operation.c_str(),
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::" << operation
                      << ": rethrowing primary exception from participant "
                      << first_exception_participant);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_request_batch_sidecars",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"source", source},
             {"implementation", "grouped_child_graphs"}});
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_resident_request_batch_draft_slot_publications",
            static_cast<double>(request_batch),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"requests", std::to_string(request_batch)},
             {"source", source},
             {"first_slot", std::to_string(first_draft_slot)},
             {"slot_stride", std::to_string(draft_slot_stride)},
             {"implementation", "participant_local_device_slots"},
             {"collective", "none"}});
        return true;
    }

    void RankOrchestrator::invalidateRankResidentLogicalStateAggregate(
        const char *lifecycle,
        const char *reason,
        int participant) const
    {
        const bool had_aggregate =
            !rank_resident_child_logical_state_handles_.empty();
        rank_resident_child_logical_state_handles_.clear();
        ++rank_resident_logical_state_epoch_;

        PerfStatsCollector::addCounterWithSequence(
            "mtp",
            "rank_resident_logical_state_aggregate_invalidations",
            1.0,
            {static_cast<uint64_t>(rank_resident_logical_state_epoch_)},
            "decode",
            "rank",
            {{"lifecycle", lifecycle && lifecycle[0] != '\0'
                               ? lifecycle
                               : "unspecified"},
             {"reason", reason && reason[0] != '\0'
                            ? reason
                            : "unspecified"},
             {"participant", std::to_string(participant)},
             {"had_aggregate", had_aggregate ? "true" : "false"}});
    }

    DeviceResidentLogicalSequenceStateHandle
    RankOrchestrator::deviceResidentLogicalSequenceState() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->deviceResidentLogicalSequenceState();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->deviceResidentLogicalSequenceState();
        }
        if (device_runners_.size() < 2 ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size())
        {
            return {};
        }

        int request_count = -1;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            const DeviceResidentLogicalSequenceStateHandle &child =
                rank_resident_child_logical_state_handles_[i];
            if (!device_runners_[i] || !child.valid())
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "cached_child_handle_invalid",
                    static_cast<int>(i));
                return {};
            }

            /*
             * A cached handle is only a rank-local ownership token. Ask the
             * child for its current mailbox before exposing the aggregate so a
             * prefix replacement, reset, workspace rebind, or publication epoch
             * change cannot leave a structurally valid but stale rank handle.
             * This compares pointers and events only; it performs no GPU work
             * and introduces no host synchronization.
             */
            const DeviceResidentLogicalSequenceStateHandle current_child =
                device_runners_[i]->deviceResidentLogicalSequenceState();
            if (!current_child.sameMailboxAs(child))
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "child_mailbox_no_longer_current",
                    static_cast<int>(i));
                return {};
            }
            if (request_count < 0)
                request_count = child.request_count;
            if (child.request_count != request_count)
            {
                invalidateRankResidentLogicalStateAggregate(
                    "aggregate_observation",
                    "child_request_count_mismatch",
                    static_cast<int>(i));
                return {};
            }
        }
        if (request_count <= 0)
            return {};

        rank_resident_logical_state_marker_[0] = request_count;
        DeviceResidentLogicalSequenceStateHandle handle;
        handle.target_positions_device =
            rank_resident_logical_state_marker_.data();
        handle.target_sequence_lengths_device =
            rank_resident_logical_state_marker_.data();
        handle.accepted_state_counts_device =
            rank_resident_logical_state_marker_.data();
        handle.next_condition_tokens_device =
            rank_resident_logical_state_marker_.data();
        handle.all_drafts_accepted_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.stopped_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.publication_ok_flags_device =
            rank_resident_logical_state_marker_.data();
        handle.request_count = request_count;
        handle.device = primaryDeviceId();
        handle.stream = &rank_resident_logical_state_stream_token_;
        handle.ready_event = &rank_resident_logical_state_ready_event_token_;
        handle.live_state_epoch = rank_resident_logical_state_epoch_;
        handle.publication_generation = rank_resident_logical_state_epoch_;
        return handle;
    }

    bool RankOrchestrator::observeDeviceResidentNextConditionTokens(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_count,
        int32_t *out_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->observeDeviceResidentNextConditionTokens(
                logical_state,
                request_count,
                out_tokens);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->observeDeviceResidentNextConditionTokens(
                logical_state,
                request_count,
                out_tokens);
        }

        const DeviceResidentLogicalSequenceStateHandle current_rank_state =
            deviceResidentLogicalSequenceState();
        if (request_count <= 0 ||
            !out_tokens ||
            !current_rank_state.valid() ||
            !logical_state.sameMailboxAs(current_rank_state) ||
            logical_state.request_count < request_count ||
            rank_resident_child_logical_state_handles_.size() !=
                device_runners_.size() ||
            device_runners_.empty() ||
            !device_runners_[0])
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident next-condition-token observation "
                "received a stale, foreign, or incomplete aggregate mailbox");
            return false;
        }

        const DeviceResidentLogicalSequenceStateHandle &primary_state =
            rank_resident_child_logical_state_handles_[0];
        if (!primary_state.valid() ||
            primary_state.request_count < request_count ||
            primary_state.device != device_runners_[0]->primaryDeviceId())
        {
            LOG_ERROR(
                "[RankOrchestrator] Resident next-condition-token observation "
                "has no current primary-child mailbox");
            return false;
        }

        /*
         * Mirrored LocalTP publishes child zero as the authoritative compact
         * token owner. The host sees only that final result row; every execution
         * consumer continues to use its own child-local device mailbox.
         */
        if (!device_runners_[0]->observeDeviceResidentNextConditionTokens(
                primary_state,
                request_count,
                out_tokens))
        {
            LOG_ERROR(
                "[RankOrchestrator] Primary child could not publish resident "
                "next-condition-token result rows");
            return false;
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_resident_next_condition_token_host_observations",
            1.0,
            "decode",
            "rank",
            {{"requests", std::to_string(request_count)},
             {"owner", "primary_child"},
             {"boundary", "host_result"}});
        return true;
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromLastForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens)
    {
        return commitMTPShiftedRowsFromPartialForward(
            tokens,
            token_count,
            already_appended_tokens,
            token_count);
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromPartialForward(
        const int32_t *tokens,
        int token_count,
        int already_appended_tokens,
        int main_forward_token_count,
        bool allow_speculative_discard,
        int position_offset_override,
        int already_appended_shifted_kv_tokens)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (token_count <= already_appended_tokens)
            return true;
        if (!tokens || token_count <= 0)
            return false;
        const int catchup_token_count = token_count - already_appended_tokens;
        const int hidden_source_row_start = already_appended_tokens - 1;
        const int hidden_source_row_end = hidden_source_row_start + catchup_token_count;
        if (main_forward_token_count <= 0 ||
            hidden_source_row_start < 0 ||
            hidden_source_row_end > main_forward_token_count)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromPartialForward: invalid main_forward_token_count="
                      << main_forward_token_count << " token_count=" << token_count
                      << " already_appended_tokens=" << already_appended_tokens
                      << " catchup_token_count=" << catchup_token_count
                      << " hidden_source_row_start=" << hidden_source_row_start
                      << " hidden_source_row_end=" << hidden_source_row_end);
            return false;
        }
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            /*
             * PP replay has already produced the accepted hidden rows on the
             * final stage.  Only that stage owns the shifted MTP KV cache, so
             * commit the accepted sidecar rows there instead of iterating every
             * pipeline participant.
             */
            return pp_sidecar->commitMTPShiftedRowsFromPartialForward(
                tokens,
                token_count,
                already_appended_tokens,
                main_forward_token_count,
                allow_speculative_discard,
                position_offset_override,
                already_appended_shifted_kv_tokens);
        }
        if (device_runners_.empty())
            return false;

        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowsFromPartialForward(
                       tokens,
                       token_count,
                       already_appended_tokens,
                       main_forward_token_count,
                       allow_speculative_discard,
                       position_offset_override,
                       already_appended_shifted_kv_tokens);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        const std::vector<int32_t> committed_tokens(tokens, tokens + token_count);
        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &committed_tokens, token_count, already_appended_tokens,
                 main_forward_token_count, allow_speculative_discard,
                 position_offset_override, already_appended_shifted_kv_tokens,
                 kernel_phase, rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
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
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token_count=" << token_count
                                  << " already_appended=" << already_appended_tokens
                                  << " main_forward_token_count=" << main_forward_token_count);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->commitMTPShiftedRowsFromPartialForward(
                                        committed_tokens.data(),
                                        token_count,
                                        already_appended_tokens,
                                        main_forward_token_count,
                                        allow_speculative_discard,
                                        position_offset_override,
                                        already_appended_shifted_kv_tokens);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_commit_leave"
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
                "rank_mtp_shifted_commit_collect",
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
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "commitMTPShiftedRowsFromLastForward",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::commitMTPShiftedRowsFromLastForward: requesting TP context abort after worker timeout");
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
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowsFromLastForward",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowsFromLastForward: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromDeviceTargetSample(
        int target_sample_slot,
        int already_appended_tokens,
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceTargetSample(
                target_sample_slot,
                already_appended_tokens,
                allow_speculative_discard);
        }
        if (device_runners_.empty())
        {
            return false;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromDeviceTargetSample(
                       target_sample_slot,
                       already_appended_tokens,
                       allow_speculative_discard);
        }

        /*
         * LocalTP resolves the first stochastic/greedy target token once at rank
         * scope and stages that token into the same device slot on every child.
         * The shifted MTP KV row must therefore be rebuilt on every participant
         * from that child-owned slot.  This is not a fallback: each child runs
         * its normal device-resident KV-only sidecar commit against its own MTP
         * cache, preserving the symmetric collective order used by the rest of
         * the grouped verifier transaction.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] device-target shifted commit failure detected - aborting collective backend");
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
             target_sample_slot,
             already_appended_tokens,
             allow_speculative_discard,
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

                if (i >= device_runners_.size() || !device_runners_[i])
                    return false;
                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                return device_runners_[i]->commitMTPShiftedRowFromDeviceTargetSample(
                           target_sample_slot,
                           already_appended_tokens,
                           allow_speculative_discard);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &r : results)
        {
            if (!r.completed || !r.success)
                all_success = false;
            if (r.exception && !first_exception)
            {
                first_exception = r.exception;
                first_exception_device = r.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device-target shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_device_target_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"target_slot", std::to_string(target_sample_slot)},
                 {"already_appended", std::to_string(already_appended_tokens)}});
        }
        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromDeviceResidentLogicalState(
        const DeviceResidentLogicalSequenceStateHandle &logical_state,
        int request_index,
        int already_appended_tokens,
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowFromDeviceResidentLogicalState(
                logical_state,
                request_index,
                already_appended_tokens,
                allow_speculative_discard);
        }
        if (device_runners_.size() != 1 || !device_runners_[0])
        {
            if (!logical_state.valid() ||
                logical_state.target_positions_device !=
                    rank_resident_logical_state_marker_.data() ||
                logical_state.live_state_epoch !=
                    rank_resident_logical_state_epoch_ ||
                rank_resident_child_logical_state_handles_.size() !=
                    device_runners_.size())
            {
                LOG_ERROR("[RankOrchestrator] Resident logical-state shifted MTP commit received a non-owned LocalTP mailbox");
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
                        LOG_WARN("[TPWorkerPool] resident logical-state shifted commit failure detected - aborting collective backend");
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
                 request_index,
                 already_appended_tokens,
                 allow_speculative_discard,
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

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    return device_runners_[i] &&
                           device_runners_[i]->commitMTPShiftedRowFromDeviceResidentLogicalState(
                               rank_resident_child_logical_state_handles_[i],
                               request_index,
                               already_appended_tokens,
                               allow_speculative_discard);
                });

            bool all_success = true;
            std::exception_ptr first_exception = nullptr;
            size_t first_exception_device = 0;
            auto results =
                tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
            for (auto &r : results)
            {
                if (!r.completed || !r.success)
                    all_success = false;
                if (r.exception && !first_exception)
                {
                    first_exception = r.exception;
                    first_exception_device = r.worker_index;
                    all_success = false;
                }
            }
            if (first_exception)
            {
                LOG_ERROR("[RankOrchestrator] Resident logical-state shifted commit rethrowing exception from participant "
                          << first_exception_device);
                std::rethrow_exception(first_exception);
            }
            if (all_success)
            {
                invalidateRankResidentLogicalStateAggregate(
                    "shifted_row_commit",
                    "child_mailboxes_consumed",
                    /*participant=*/-1);
                PerfStatsCollector::addCounter(
                    "mtp",
                    "rank_resident_logical_state_shifted_commits",
                    1.0,
                    "decode",
                    "rank",
                    {{"participants", std::to_string(device_runners_.size())},
                     {"request_index", std::to_string(request_index)}});
            }
            return all_success;
        }
        return device_runners_[0]->commitMTPShiftedRowFromDeviceResidentLogicalState(
            logical_state,
            request_index,
            already_appended_tokens,
            allow_speculative_discard);
    }

    bool RankOrchestrator::commitMTPShiftedRowsFromDeviceOutcome(
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        int already_appended_tokens,
        int max_state_commit_rows,
        int main_forward_token_count,
        bool allow_speculative_discard)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->commitMTPShiftedRowsFromDeviceOutcome(
                outcome,
                request_index,
                already_appended_tokens,
                max_state_commit_rows,
                main_forward_token_count,
                allow_speculative_discard);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->commitMTPShiftedRowsFromDeviceOutcome(
                outcome,
                request_index,
                already_appended_tokens,
                max_state_commit_rows,
                main_forward_token_count,
                allow_speculative_discard);
        }

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                outcome,
                "shifted MTP suffix commit",
                &outcome_error))
        {
            LOG_ERROR("[RankOrchestrator] " << outcome_error);
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
                    LOG_WARN("[TPWorkerPool] device-outcome shifted commit failure detected - aborting collective backend");
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
             request_index,
             already_appended_tokens,
             max_state_commit_rows,
             main_forward_token_count,
             allow_speculative_discard,
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

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                return device_runners_[i] &&
                       device_runners_[i]->commitMTPShiftedRowsFromDeviceOutcome(
                           rank_mirrored_child_outcomes_[i],
                           request_index,
                           already_appended_tokens,
                           max_state_commit_rows,
                           main_forward_token_count,
                           allow_speculative_discard);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &r : results)
        {
            if (!r.completed || !r.success)
                all_success = false;
            if (r.exception && !first_exception)
            {
                first_exception = r.exception;
                first_exception_device = r.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device-outcome shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_device_outcome_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)},
                 {"rows", std::to_string(std::max(0, max_state_commit_rows - already_appended_tokens))}});
        }
        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden(
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_sequential_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            if (already_appended_tokens < 0)
                return false;
            return pp_sidecar->commitMTPShiftedRowFromCurrentTerminalHidden(
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (already_appended_tokens < 0 || device_runners_.empty())
            return false;

        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromCurrentTerminalHidden(
                       token,
                       already_appended_tokens,
                       allow_speculative_discard,
                       position_offset_override);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] MTP sequential shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_sequential_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, token, already_appended_tokens, allow_speculative_discard,
                 position_offset_override, kernel_phase, rocm_phase, cuda_phase,
                 kv_phase, executor_phase](size_t i) -> bool
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
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_sequential_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << token
                                  << " already_appended=" << already_appended_tokens);
                    }

                    const bool ok = device_runners_[i] &&
                                    device_runners_[i]->commitMTPShiftedRowFromCurrentTerminalHidden(
                                        token,
                                        already_appended_tokens,
                                        allow_speculative_discard,
                                        position_offset_override);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_sequential_commit_leave"
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
                "rank_mtp_shifted_sequential_commit_collect",
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
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (collect_timeout_ms > 0)
                {
                    abortAfterTPWorkerTimeout(
                        "commitMTPShiftedRowFromCurrentTerminalHidden",
                        collect_timeout_ms,
                        tp_worker_pool_->completedCount(),
                        tp_worker_pool_->numWorkers());
                }
                if (tp_ctx_)
                {
                    LOG_WARN("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: requesting TP context abort after worker timeout");
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
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }

        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowFromCurrentTerminalHidden",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }

        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCurrentTerminalHidden: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        return all_success;
    }

    bool RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden(
        const PrefixStateSnapshot &checkpoint,
        int32_t token,
        int already_appended_tokens,
        bool allow_speculative_discard,
        int position_offset_override)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_shifted_checkpoint_terminal_hidden_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (!checkpoint.valid || already_appended_tokens < 0)
            return false;

        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const PrefixStateSnapshot *tail_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                tail_checkpoint = &checkpoint.participant_snapshots.back();
            return pp_sidecar->commitMTPShiftedRowFromCheckpointTerminalHidden(
                *tail_checkpoint,
                token,
                already_appended_tokens,
                allow_speculative_discard,
                position_offset_override);
        }
        if (device_runners_.empty())
            return false;

        const bool aggregate_checkpoint =
            !checkpoint.participant_snapshots.empty();
        const bool shared_logical_checkpoint =
            !aggregate_checkpoint && checkpoint.logical_checkpoint;
        if (aggregate_checkpoint &&
            checkpoint.participant_snapshots.size() != device_runners_.size())
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: participant snapshot count mismatch: snapshots="
                      << checkpoint.participant_snapshots.size()
                      << " runners=" << device_runners_.size());
            return false;
        }
        auto child_checkpoint = [&](size_t i) -> const PrefixStateSnapshot *
        {
            if (aggregate_checkpoint)
                return &checkpoint.participant_snapshots[i];
            if (device_runners_.size() == 1 || shared_logical_checkpoint)
                return &checkpoint;
            return nullptr;
        };

        if (device_runners_.size() == 1)
        {
            const PrefixStateSnapshot *child = child_checkpoint(0);
            return child &&
                   device_runners_[0] &&
                   device_runners_[0]->commitMTPShiftedRowFromCheckpointTerminalHidden(
                       *child,
                       token,
                       already_appended_tokens,
                       allow_speculative_discard,
                       position_offset_override);
        }

        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this]()
                                                    {
                    LOG_WARN("[TPWorkerPool] checkpoint-terminal-hidden shifted-row commit failure detected - aborting collective backend");
                    tp_ctx_->requestAbort(); });
            }
        }

        auto kernel_phase = KernelProfiler::getCurrentPhase();
        auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        auto kv_phase = KVCacheProfiler::getCurrentPhase();
        auto executor_phase = GraphExecutorStats::currentPhase();

        {
            PerfStatsCollector::ScopedTimer dispatch_timer(
                "mtp",
                "rank_mtp_shifted_checkpoint_terminal_hidden_commit_dispatch",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            tp_worker_pool_->dispatch(
                [this, &checkpoint, &child_checkpoint, token, already_appended_tokens,
                 allow_speculative_discard, position_offset_override,
                 kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t i) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);

                    if (i >= device_runners_.size() ||
                        !device_runners_[i])
                    {
                        return false;
                    }
                    const PrefixStateSnapshot *child = child_checkpoint(i);
                    if (!child)
                    {
                        LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                                  << i
                                  << " has no participant checkpoint for non-logical multi-device commit");
                        return false;
                    }

                    auto device_id = device_runners_[i]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_checkpoint_terminal_hidden_commit_enter"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " token=" << token
                                  << " already_appended=" << already_appended_tokens);
                    }

                    const bool ok =
                        device_runners_[i]->commitMTPShiftedRowFromCheckpointTerminalHidden(
                            *child,
                            token,
                            already_appended_tokens,
                            allow_speculative_discard,
                            position_offset_override);

                    if (debugEnv().tp_collective_contract_trace)
                    {
                        LOG_DEBUG("[TP_WORKER_CONTRACT] event=mtp_shifted_checkpoint_terminal_hidden_commit_leave"
                                  << " worker=" << i
                                  << " device=" << device_id.toString()
                                  << " success=" << (ok ? 1 : 0));
                    }
                    return ok;
                });
        }

        std::vector<TPWorkerPool::WorkerResult> results;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        {
            PerfStatsCollector::ScopedTimer collect_timer(
                "mtp",
                "rank_mtp_shifted_checkpoint_terminal_hidden_commit_collect",
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())}});
            results = tp_worker_pool_->collectAll(collect_timeout_ms);
        }

        bool all_success = true;
        bool worker_timeout = false;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        for (auto &r : results)
        {
            if (!r.completed)
            {
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                          << r.worker_index << " did not complete (stuck)");
                worker_timeout = true;
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
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
                LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Device "
                          << r.worker_index << " shifted-row commit failed");
                all_success = false;
            }
        }
        if (worker_timeout && collect_timeout_ms > 0)
        {
            abortAfterTPWorkerTimeout(
                "commitMTPShiftedRowFromCheckpointTerminalHidden",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::commitMTPShiftedRowFromCheckpointTerminalHidden: Re-throwing primary exception from device "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        return all_success;
    }

    bool RankOrchestrator::commitMTPInitialShiftedRowFromDeviceOutcome(
        const PrefixStateSnapshot &checkpoint,
        const DeviceSpeculativeOutcomeHandle &outcome,
        int request_index,
        bool allow_speculative_discard)
    {
        PerfStatsCollector::ScopedTimer total_timer(
            "mtp",
            "rank_mtp_initial_shifted_device_outcome_commit_total",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())}});
        if (!checkpoint.valid || request_index < 0)
            return false;

        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const PrefixStateSnapshot *tail_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                tail_checkpoint = &checkpoint.participant_snapshots.back();
            return pp_sidecar->commitMTPInitialShiftedRowFromDeviceOutcome(
                *tail_checkpoint,
                outcome,
                request_index,
                allow_speculative_discard);
        }
        if (device_runners_.empty())
            return false;
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            const PrefixStateSnapshot *child_checkpoint = &checkpoint;
            if (!checkpoint.participant_snapshots.empty())
                child_checkpoint = &checkpoint.participant_snapshots.front();
            return device_runners_[0]->commitMTPInitialShiftedRowFromDeviceOutcome(
                *child_checkpoint,
                outcome,
                request_index,
                allow_speculative_discard);
        }

        const bool aggregate_checkpoint =
            !checkpoint.participant_snapshots.empty();
        const bool shared_logical_checkpoint =
            !aggregate_checkpoint && checkpoint.logical_checkpoint;
        if (aggregate_checkpoint &&
            checkpoint.participant_snapshots.size() != device_runners_.size())
        {
            LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted MTP commit checkpoint count mismatch: snapshots="
                      << checkpoint.participant_snapshots.size()
                      << " runners=" << device_runners_.size());
            return false;
        }
        auto child_checkpoint = [&](size_t i) -> const PrefixStateSnapshot *
        {
            if (aggregate_checkpoint)
                return &checkpoint.participant_snapshots[i];
            if (shared_logical_checkpoint)
                return &checkpoint;
            return nullptr;
        };

        std::string outcome_error;
        if (!validateCurrentMirroredLocalTPOutcome(
                outcome,
                "initial shifted MTP row commit",
                &outcome_error))
        {
            LOG_ERROR("[RankOrchestrator] " << outcome_error);
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
                    LOG_WARN("[TPWorkerPool] device-outcome initial shifted commit failure detected - aborting collective backend");
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
             &child_checkpoint,
             request_index,
             allow_speculative_discard,
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

                if (i >= device_runners_.size() ||
                    !device_runners_[i])
                {
                    return false;
                }
                const PrefixStateSnapshot *child = child_checkpoint(i);
                if (!child)
                {
                    LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted MTP commit has no participant checkpoint for child "
                              << i);
                    return false;
                }

                auto device_id = device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device_id.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device_id.ordinal);

                return device_runners_[i]->commitMTPInitialShiftedRowFromDeviceOutcome(
                    *child,
                    rank_mirrored_child_outcomes_[i],
                    request_index,
                    allow_speculative_discard);
            });

        bool all_success = true;
        std::exception_ptr first_exception = nullptr;
        size_t first_exception_device = 0;
        auto results =
            tp_worker_pool_->collectAll(effectiveTPWorkerJoinTimeoutMs());
        for (auto &r : results)
        {
            if (!r.completed || !r.success)
                all_success = false;
            if (r.exception && !first_exception)
            {
                first_exception = r.exception;
                first_exception_device = r.worker_index;
                all_success = false;
            }
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device-outcome initial shifted commit rethrowing exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (all_success)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_device_outcome_initial_shifted_commits",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"request_index", std::to_string(request_index)}});
        }
        return all_success;
    }

    bool RankOrchestrator::flushPendingMTPWork()
    {
        if (!pp_stage_runners_.empty())
        {
            bool ok = true;
            for (const auto &runner : pp_stage_runners_)
            {
                ok = runner && runner->flushPendingMTPWork() && ok;
            }
            return ok;
        }

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner && runner->flushPendingMTPWork() && ok;
        }
        return ok;
    }

    bool RankOrchestrator::ensureMTPCheckpointTerminalHidden()
    {
        if (!pp_stage_runners_.empty())
        {
            /*
             * This helper materializes the terminal hidden row consumed by the
             * MTP sidecar. In LocalPP, only the pipeline tail owns the final
             * hidden state, final norm, LM head, and MTP sidecar. Earlier stages
             * own local KV/GDN replay state, but their activation tensors are
             * transferred onward after forwardPP(); asking them to row-select a
             * "terminal" hidden row can race stale producer-side device buffers.
             */
            IInferenceRunner *pp_sidecar = finalPPSidecarRunner();
            return pp_sidecar && pp_sidecar->ensureMTPCheckpointTerminalHidden();
        }

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner && runner->ensureMTPCheckpointTerminalHidden() && ok;
        }
        return ok;
    }

    const float *RankOrchestrator::mtpLogits() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->mtpLogits();
        }

        if (device_runners_.empty())
        {
            return nullptr;
        }

        bool has_local_mtp_logits = false;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            has_local_mtp_logits = has_local_mtp_logits || runner->hasMTPLogitsLocal();
        }

        if (has_local_mtp_logits)
        {
            std::vector<LogitsLocalInfo> local_infos;
            local_infos.reserve(device_runners_.size());
            for (const auto &runner : device_runners_)
            {
                if (!runner->hasMTPLogitsLocal())
                {
                    LOG_ERROR("RankOrchestrator::mtpLogits: mixed local and replicated MTP logits are invalid");
                    return nullptr;
                }

                LogitsLocalInfo info =
                    runner->consumeMTPLogitsLocalInfoForHostGather();
                if (!info)
                    return nullptr;
                local_infos.push_back(info);
            }

            const int full_vocab = vocab_size();
            if (full_vocab <= 0)
                return nullptr;

            if (!mtp_logits_gatherer_ ||
                mtp_logits_gatherer_->bufferNumel() < static_cast<size_t>(full_vocab))
            {
                mtp_logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    full_vocab,
                    1,
                    logits_backend_resolver_);
            }

            if (!mtp_logits_gatherer_ ||
                !mtp_logits_gatherer_->gatherLocalInfos(local_infos, 1, full_vocab))
            {
                return nullptr;
            }
            return mtp_logits_gatherer_->data();
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        const float *primary_logits = nullptr;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;

            const float *child_logits = runner->mtpLogits();
            if (!child_logits)
                return nullptr;

            if (!primary_logits)
            {
                primary_logits = child_logits;
                continue;
            }

            if (!std::equal(primary_logits, primary_logits + compare_vocab, child_logits))
            {
                LOG_WARN("RankOrchestrator::mtpLogits: child MTP logits diverged in replicated TP path");
                return nullptr;
            }
        }
        return primary_logits;
    }

    int RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice()
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMTPLogitsOnDevice();
        }

        if (device_runners_.empty())
        {
            return -1;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] ? device_runners_[0]->sampleGreedyFromMTPLogitsOnDevice() : -1;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasMTPLogitsLocal())
            {
                LOG_DEBUG("RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                          "participant missing local MTP logits");
                return -1;
            }

            LogitsLocalInfo info = runner->consumeMTPLogitsLocalInfoForSampling();
            if (!info)
            {
                LOG_DEBUG("RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                          "participant returned empty local MTP logits info");
                return -1;
            }
            local_infos.push_back(info);
        }

        const int token = DeviceSampler::sampleGreedyFromLocalInfos(local_infos, 0);
        if (token < 0)
        {
            std::ostringstream oss;
            oss << "RankOrchestrator::sampleGreedyFromMTPLogitsOnDevice: "
                   "sharded MTP sampling failed for "
                << local_infos.size() << " participants";
            for (size_t i = 0; i < local_infos.size(); ++i)
            {
                const auto &info = local_infos[i];
                oss << " [idx=" << i
                    << " tensor=" << (info.tensor ? "yes" : "no")
                    << " gpu_ptr=" << info.gpu_ptr
                    << " device="
                    << (info.device.has_value() ? info.device->toString() : std::string("none"))
                    << " stream=" << info.stream
                    << " vocab_local=" << info.vocab_local
                    << " argmax_capacity=" << info.argmax_partial_capacity
                    << "]";
            }
            LOG_DEBUG(oss.str());
        }
        return token;
    }

    bool RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot(
        int draft_sample_slot,
        int32_t *out_token)
    {
        if (out_token)
            *out_token = -1;
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                draft_sample_slot,
                out_token);
        }
        if (publishesFullVocabularyMTPLogits())
        {
            /*
             * Mirrored LocalTP MTP heads expose full-vocabulary sidecar logits
             * on every participant.  Sampling each child directly keeps the
             * draft-token slots device-resident and avoids a tiny rank-level
             * allreduce/gather.  When the caller still asks for a host shadow,
             * use it only as a consistency check and response mirror.
             */
            bool have_shadow = false;
            int32_t rank_token = -1;
            for (size_t child = 0; child < device_runners_.size(); ++child)
            {
                IInferenceRunner *runner = device_runners_[child].get();
                if (!runner)
                    return false;

                int32_t child_token = -1;
                int32_t *child_shadow = out_token ? &child_token : nullptr;
                if (!runner->sampleGreedyFromMTPLogitsToDeviceDraftSlot(
                        draft_sample_slot,
                        child_shadow))
                {
                    LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                              "mirrored child " << child
                                                << " failed device draft-slot sampling");
                    return false;
                }

                if (out_token)
                {
                    if (child_token < 0)
                    {
                        LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                                  "mirrored child " << child
                                                    << " returned an invalid draft token");
                        return false;
                    }
                    if (!have_shadow)
                    {
                        rank_token = child_token;
                        have_shadow = true;
                    }
                    else if (child_token != rank_token)
                    {
                        std::vector<std::vector<MTPMirroredTensorDigest>>
                            participant_digests;
                        participant_digests.reserve(device_runners_.size());
                        for (const auto &participant : device_runners_)
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

                        LOG_ERROR("RankOrchestrator::sampleGreedyFromMTPLogitsToDeviceDraftSlot: "
                                  "mirrored children sampled different draft tokens "
                                  << rank_token << " and " << child_token
                                  << describeMirroredDigestMismatch(
                                         participant_digests));
                        return false;
                    }
                }
            }

            if (out_token)
                *out_token = rank_token;

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_mtp_draft_slot_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(draft_sample_slot)},
                 {"host_shadow", out_token ? "true" : "false"}});
            return true;
        }
        return sampleRankGreedyMTPLogitsToLocalTPDraftSlot(
            draft_sample_slot,
            out_token);
    }

    bool RankOrchestrator::sampleGreedyFromMainLogitsToDeviceTargetSlot(
        int target_sample_slot,
        int32_t *out_token)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                target_sample_slot,
                out_token);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                target_sample_slot,
                out_token);
        }

        /*
         * The main LM head and the MTP verifier head have independent
         * distribution policies.  In particular, LocalTP may shard the main
         * LM head while mirroring the much smaller MTP head.  Never infer the
         * former from publishesFullVocabularyMTPLogits(): doing so would
         * sample only participant zero's main-logits shard and then incorrectly
         * retire participant one as though it owned a redundant full-vocabulary
         * row.
         */
        bool any_local_main_logits = false;
        bool all_local_main_logits = true;
        for (const auto &runner : device_runners_)
        {
            const bool has_local = runner && runner->hasLogitsLocal();
            any_local_main_logits = any_local_main_logits || has_local;
            all_local_main_logits = all_local_main_logits && has_local;
        }
        if (any_local_main_logits != all_local_main_logits)
        {
            LOG_ERROR("[RankOrchestrator] LocalTP main-target argmax has mixed "
                      "sharded and replicated main-logits declarations");
            return false;
        }
        if (all_local_main_logits)
        {
            return sampleRankGreedyMainLogitsToLocalTPTargetSlot(
                target_sample_slot,
                out_token);
        }

        if (publishesFullVocabularyMTPLogits())
        {
            if (target_sample_slot < 0 ||
                target_sample_slot >= rank_stochastic_slot_capacity_ ||
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
                    LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax requires a full-head device sampler on participant "
                              << child);
                    return false;
                }
            }

            /*
             * The mirrored rows are identical, so one argmax is sufficient.
             * Keep the normal serving path entirely device-owned by requesting
             * no host shadow from child zero, then broadcast its target mailbox
             * slot through the configured LocalTP collective. A host value is
             * materialized only for callers that explicitly request one.
             */
            int32_t primary_shadow = -1;
            if (!device_runners_.front()
                     ->sampleGreedyFromMainLogitsToDeviceTargetSlot(
                         target_sample_slot,
                         out_token ? &primary_shadow : nullptr))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax failed on primary participant");
                return false;
            }
            if (!broadcastPrimaryLocalTPMainTargetSlotToChildren(
                    target_sample_slot))
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP main-target argmax could not broadcast the primary device slot");
                return false;
            }
            if (out_token)
                *out_token = primary_shadow;

            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_main_target_argmax_samples",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"slot", std::to_string(target_sample_slot)},
                 {"host_shadow", out_token ? "true" : "false"},
                 {"implementation", "primary_child_device_slot_broadcast"}});
            return true;
        }
        return sampleRankGreedyMainLogitsToLocalTPTargetSlot(
            target_sample_slot,
            out_token);
    }

    int RankOrchestrator::sampleGreedyFromAllPositionLogitsOnDevice(int row)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromAllPositionLogitsOnDevice(row);
        }

        if (device_runners_.empty() || row < 0)
        {
            return -1;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0]
                       ? device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDevice(row)
                       : -1;
        }

        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner || !runner->hasAllPositionLogitsLocal())
            {
                return -1;
            }

            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
            if (!info)
            {
                return -1;
            }
            local_infos.push_back(info);
        }

        return DeviceSampler::sampleGreedyFromLocalInfos(local_infos, row);
    }

    bool RankOrchestrator::sampleGreedyFromAllPositionLogitsOnDeviceRows(
        int start_row,
        int row_count,
        int32_t *out_tokens)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row,
                row_count,
                out_tokens);
        }

        if (device_runners_.empty() ||
            start_row < 0 ||
            row_count <= 0 ||
            !out_tokens)
        {
            return false;
        }
        if (device_runners_.size() == 1)
        {
            return device_runners_[0] &&
                   device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                       start_row,
                       row_count,
                       out_tokens);
        }

        bool any_local_logits = false;
        bool all_local_logits = true;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return false;
            const bool has_local = runner->hasAllPositionLogitsLocal();
            any_local_logits = any_local_logits || has_local;
            all_local_logits = all_local_logits && has_local;
        }

        if (!any_local_logits)
        {
            /*
             * Some LocalTP tests and replicated-logits topologies expose a full
             * verifier-logits tensor on each child instead of vocab shards. One
             * representative child can sample those rows directly.
             */
            return device_runners_[0]->sampleGreedyFromAllPositionLogitsOnDeviceRows(
                start_row,
                row_count,
                out_tokens);
        }

        if (!all_local_logits)
        {
            return false;
        }

        /*
         * Column-parallel LocalTP all-position verifier rows are logits shards.
         * Sampling is the verifier replay consumer, so collect one consuming
         * LogitsLocalInfo per child and reuse those exact streams for every
         * row in this batch.  Using the non-consuming getter here would sample
         * on child default streams and can race graph-captured verifier replay.
         */
        std::vector<LogitsLocalInfo> local_infos;
        local_infos.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            LogitsLocalInfo info =
                runner->consumeAllPositionLogitsLocalInfoForSampling();
            if (!info)
                return false;
            local_infos.push_back(info);
        }

        if (DeviceSampler::sampleGreedyRowsFromLocalInfos(
                local_infos,
                start_row,
                row_count,
                out_tokens))
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_all_position_verifier_token_batch_samples",
                static_cast<double>(row_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"start_row", std::to_string(start_row)},
                 {"implementation", "batched_cross_shard_argmax"}});
            return true;
        }

        for (int i = 0; i < row_count; ++i)
        {
            const int token = DeviceSampler::sampleGreedyFromLocalInfos(
                local_infos,
                start_row + i);
            if (token < 0)
                return false;
            out_tokens[i] = static_cast<int32_t>(token);
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_all_position_verifier_token_batch_samples",
            static_cast<double>(row_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"start_row", std::to_string(start_row)},
             {"implementation", "serial_cross_shard_argmax"}});
        return true;
    }

    bool RankOrchestrator::resolveRankGreedyOutcomeTokensForLocalTP(
        const int32_t *draft_tokens,
        int draft_token_count,
        std::vector<int32_t> *out_tokens) const
    {
        if (!draft_tokens ||
            draft_token_count <= 0 ||
            !out_tokens)
        {
            return false;
        }

        out_tokens->clear();
        out_tokens->reserve(static_cast<size_t>(draft_token_count));

        int resolved_shadow_count = 0;
        for (int i = 0; i < draft_token_count; ++i)
        {
            int32_t token = draft_tokens[i];
            if (token < 0)
            {
                /*
                 * OrchestrationRunner uses negative shadows only after it has
                 * staged the corresponding compact token into rank-owned slots.
                 * Slot zero is the fixed first-token target slot for the current
                 * single-request transaction; draft entries map one-to-one to
                 * draft slots starting at zero.
                 */
                token = (i == 0)
                            ? rankStochasticTargetSampleToken(/*slot=*/0)
                            : rankStochasticDraftSampleToken(/*slot=*/i - 1);
                ++resolved_shadow_count;
            }
            if (token < 0)
            {
                out_tokens->clear();
                return false;
            }
            out_tokens->push_back(token);
        }

        if (resolved_shadow_count > 0)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_greedy_outcome_deferred_token_resolutions",
                static_cast<double>(resolved_shadow_count),
                "decode",
                "rank",
                {{"tokens", std::to_string(draft_token_count)}});
        }
        return true;
    }

    bool RankOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDevice(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeVerifyBatchOutcome *out)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionBatchOutcomeOnDevice(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionBatchOutcomeOnDevice(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out);
        }

        using namespace sampling_math;
        if (!out)
            return false;
        *out = DeviceSpeculativeVerifyBatchOutcome{};

        const int compare_rows = draft_token_count - 1;
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > rank_max_verifier_rows_ ||
            compare_rows < 0 ||
            compare_rows > rank_max_draft_depth_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }

        std::vector<int32_t> resolved_draft_tokens;
        if (!resolveRankGreedyOutcomeTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                &resolved_draft_tokens))
        {
            return false;
        }

        /*
         * LocalTP all-position verifier logits are vocab shards.  Reduce the
         * compact verifier row batch across child-local shards, then run the
         * same SamplingMath summary used by single-device GPU reducers.  This
         * avoids gathering full logits and keeps accepted-count/ready-token
         * semantics identical to serial greedy decode.
         */
        std::vector<int32_t> verifier_tokens_i32(
            static_cast<size_t>(draft_token_count),
            -1);
        if (!sampleGreedyFromAllPositionLogitsOnDeviceRows(
                /*start_row=*/0,
                draft_token_count,
                verifier_tokens_i32.data()))
        {
            return false;
        }

        std::vector<int> verifier_tokens(
            static_cast<size_t>(draft_token_count),
            -1);
        std::vector<int> packed_draft_tokens(
            static_cast<size_t>(draft_token_count),
            -1);
        for (int row = 0; row < draft_token_count; ++row)
        {
            verifier_tokens[static_cast<size_t>(row)] =
                static_cast<int>(verifier_tokens_i32[static_cast<size_t>(row)]);
            packed_draft_tokens[static_cast<size_t>(row)] =
                static_cast<int>(resolved_draft_tokens[static_cast<size_t>(row)]);
        }

        std::array<int, kSpeculativeBatchMaxStopTokens> packed_stop_tokens{};
        packed_stop_tokens.fill(-1);
        for (int i = 0; i < stop_token_count; ++i)
            packed_stop_tokens[static_cast<size_t>(i)] = stop_tokens[i];

        std::vector<int> output_tokens_int(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        std::array<int, kSpeculativeBatchMetaCount> meta{};
        meta.fill(0);
        summarize_greedy_speculative_verify_batch(
            static_cast<int>(resolved_draft_tokens[0]),
            verifier_tokens.data(),
            packed_draft_tokens.data(),
            compare_rows,
            packed_stop_tokens.data(),
            stop_token_count,
            output_tokens_int.data(),
            static_cast<int>(output_tokens_int.size()),
            meta.data());

        std::vector<int32_t> output_tokens(
            static_cast<size_t>(rank_compact_output_token_stride_),
            -1);
        for (size_t i = 0; i < output_tokens.size(); ++i)
            output_tokens[i] = static_cast<int32_t>(output_tokens_int[i]);

        if (!fillRankSpeculativeVerifyOutcomeFromMeta(output_tokens, meta, out))
            return false;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_all_position_greedy_compact_outcomes",
            static_cast<double>(draft_token_count),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"compare_rows", std::to_string(compare_rows)},
             {"implementation", "cross_shard_argmax"}});
        return true;
    }

    bool RankOrchestrator::prepareGreedyAllPositionBatchOutcomeGraph(
        int verifier_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->prepareGreedyAllPositionBatchOutcomeGraph(
                verifier_token_count,
                stop_tokens,
                stop_token_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->prepareGreedyAllPositionBatchOutcomeGraph(
                    verifier_token_count,
                    stop_tokens,
                    stop_token_count,
                    penalty_policy);
        }
        if (device_runners_.size() < 2 ||
            !publishesFullVocabularyMTPLogits())
        {
            LOG_ERROR("[RankOrchestrator] Multi-device graph-owned greedy outcomes require a mirrored LocalTP MTP head");
            return false;
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            const auto &child = device_runners_[i];
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP graph-owned greedy outcome participant "
                          << i << " is not eligible");
                return false;
            }
        }

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i]
                     ->prepareGreedyAllPositionBatchOutcomeGraph(
                         verifier_token_count,
                         stop_tokens,
                         stop_token_count,
                         penalty_policy))
            {
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                throw std::runtime_error(
                    "Failed to arm every mirrored LocalTP graph-owned greedy "
                    "outcome participant");
            }
        }
        PerfStatsCollector::addCounter(
            "mtp",
            "rank_graph_owned_greedy_outcome_transactions_armed",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"rows", std::to_string(verifier_token_count)}});
        return true;
    }

    bool RankOrchestrator::configureMTPRequestStopTokens(
        const std::vector<int32_t> &stop_tokens)
    {
        auto configure_runners =
            [&stop_tokens](
                std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                const char *group)
        {
            for (size_t i = 0; i < runners.size(); ++i)
            {
                if (!runners[i])
                {
                    throw std::logic_error(
                        std::string{"Cannot configure MTP request stop tokens "
                                    "on a null "} +
                        group + " participant");
                }
                if (!runners[i]->configureMTPRequestStopTokens(stop_tokens))
                {
                    throw std::runtime_error(
                        std::string{"MTP request stop-token configuration was "
                                    "rejected by "} +
                        group + " participant " + std::to_string(i));
                }
            }
        };

        configure_runners(device_runners_, "device");
        configure_runners(pp_stage_runners_, "pipeline");
        return true;
    }

    bool RankOrchestrator::initializePromptRepetitionHistory(
        std::span<const int32_t> unique_tokens, int request_index)
    {
        for (auto *group : {&device_runners_, &pp_stage_runners_})
            for (const auto &runner : *group)
                if (!runner || !runner->initializePromptRepetitionHistory(unique_tokens, request_index))
                    return false;
        return true;
    }

    bool RankOrchestrator::configureMTPRequestPenaltyPolicy(
        const MTPRequestPenaltyPolicy &policy)
    {
        auto configure_runners =
            [&policy](
                std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                const char *group)
        {
            for (size_t i = 0; i < runners.size(); ++i)
            {
                if (!runners[i])
                {
                    throw std::logic_error(
                        std::string{"Cannot configure MTP request penalty policy "
                                    "on a null "} +
                        group + " participant");
                }
                if (!runners[i]->configureMTPRequestPenaltyPolicy(policy))
                {
                    throw std::runtime_error(
                        std::string{"MTP request penalty-policy configuration "
                                    "was rejected by "} +
                        group + " participant " + std::to_string(i));
                }
            }
        };

        configure_runners(device_runners_, "device");
        configure_runners(pp_stage_runners_, "pipeline");
        return true;
    }

    bool RankOrchestrator::verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }

        if (!out_handle)
            return false;
        *out_handle = DeviceSpeculativeOutcomeHandle{};
        rank_compact_outcome_valid_ = false;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_resident_child_logical_state_handles_.clear();

        if (publishesFullVocabularyMTPLogits())
        {
            return verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(
                draft_tokens,
                draft_token_count,
                stop_tokens,
                stop_token_count,
                out_handle);
        }

        if (primaryDeviceId().is_gpu())
        {
            LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident greedy verification requires mirrored full-head child verifier outcomes");
            return false;
        }

        std::vector<int32_t> resolved_draft_tokens;
        if (!resolveRankGreedyOutcomeTokensForLocalTP(
                draft_tokens,
                draft_token_count,
                &resolved_draft_tokens))
        {
            return false;
        }

        DeviceSpeculativeVerifyBatchOutcome outcome;
        if (!verifyGreedyAllPositionBatchOutcomeOnDevice(
                resolved_draft_tokens.data(),
                draft_token_count,
                stop_tokens,
                stop_token_count,
                &outcome))
        {
            return false;
        }

        /*
         * Stage the already-reduced SamplingMath outcome into rank-owned
         * compact arrays.  The handle points at these arrays so publication and
         * response materialization consume one identical summary.  No child
         * runner is re-entered here, and no verifier row is replayed.
         */
        using namespace sampling_math;
        if (outcome.output_tokens.size() > rank_compact_output_tokens_.size())
            return false;
        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
        std::copy(
            outcome.output_tokens.begin(),
            outcome.output_tokens.end(),
            rank_compact_output_tokens_.begin());
        rank_compact_output_meta_.fill(0);
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::Greedy;
        rank_compact_output_meta_[kSpecBatchMetaOk] = outcome.ok ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaOutputCount] =
            outcome.output_token_count;
        rank_compact_output_meta_[kSpecBatchMetaAcceptedSpeculativePrefix] =
            outcome.accepted_speculative_prefix;
        rank_compact_output_meta_
            [kSpecBatchMetaTargetVerifierStateCommitCount] =
                outcome.target_verifier_state_commit_count;
        rank_compact_output_meta_[kSpecBatchMetaReadyToken] =
            outcome.ready_token;
        rank_compact_output_meta_[kSpecBatchMetaRejectedVerifiedToken] =
            outcome.rejected_verified_token;
        rank_compact_output_meta_[kSpecBatchMetaStoppedOnOutput] =
            outcome.stopped_on_output ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaAllSpeculativeAccepted] =
            outcome.all_speculative_accepted ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaConsumedVerifierRows] =
            outcome.consumed_verifier_rows;
        rank_compact_output_meta_[kSpecBatchMetaSampledTerminal] =
            outcome.sampled_terminal ? 1 : 0;
        rank_compact_output_meta_[kSpecBatchMetaCommitBoundaryClipped] =
            outcome.commit_boundary_clipped ? 1 : 0;

        rank_compact_last_draft_tokens_.assign(
            resolved_draft_tokens.begin(),
            resolved_draft_tokens.end());
        rank_compact_last_stop_tokens_.clear();
        if (stop_token_count > 0)
        {
            rank_compact_last_stop_tokens_.assign(
                stop_tokens,
                stop_tokens + stop_token_count);
        }

        out_handle->output_tokens_device = rank_compact_output_tokens_.data();
        out_handle->meta_device = rank_compact_output_meta_.data();
        out_handle->request_count = 1;
        out_handle->logical_verifier_rows_per_request = draft_token_count;
        out_handle->physical_verifier_rows_per_request = draft_token_count;
        out_handle->output_token_stride = rank_compact_output_token_stride_;
        out_handle->meta_stride = kSpeculativeBatchMetaCount;
        out_handle->device = primaryDeviceId();
        out_handle->stream = &rank_compact_outcome_stream_token_;
        out_handle->response_ready_event =
            std::shared_ptr<void>(
                &rank_compact_outcome_ready_event_token_,
                [](void *) {});
        /*
         * This is the rank-owned fallback only for non-GPU LocalTP.  It reduces
         * the greedy outcome directly into the rank's compact mailbox, so the
         * handoff must explicitly identify the greedy graph family just as a
         * device-owned producer does.  Leaving the tag unspecified would make
         * an otherwise complete mailbox intentionally invalid.
         */
        out_handle->sampling_mode = DeviceGenerationSamplingMode::Greedy;

        rank_compact_outcome_valid_ = out_handle->valid();
        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_all_position_greedy_resident_compatible_outcomes",
                static_cast<double>(draft_token_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "rank_owned_compact_sampling_math"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::verifyGreedyMirroredLocalTPBatchOutcomeOnDeviceResident(
        const int32_t *draft_tokens,
        int draft_token_count,
        const int32_t *stop_tokens,
        int stop_token_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        using namespace sampling_math;
        if (!out_handle)
            return false;

        const int compare_rows = draft_token_count - 1;
        if (device_runners_.size() < 2 ||
            !draft_tokens ||
            draft_token_count <= 0 ||
            draft_token_count > rank_max_verifier_rows_ ||
            compare_rows < 0 ||
            compare_rows > rank_max_draft_depth_ ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens))
        {
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy MTP requires every child to expose a GPU mirrored-head resident publisher; participant "
                          << i << " is not ready");
                return false;
            }

            /*
             * Pass the verifier token shadows through unchanged.  If the first
             * token or drafts are device-resident, the child reducer reads the
             * already-materialized verifier input row from its own arena buffer;
             * resolving the shadows here would reintroduce a host ownership edge.
             */
            if (!child->verifyGreedyAllPositionBatchOutcomeOnDeviceResident(
                    draft_tokens,
                    draft_token_count,
                    stop_tokens,
                    stop_token_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid() ||
                child_outcomes[i].sampling_mode !=
                    DeviceGenerationSamplingMode::Greedy)
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy MTP child "
                          << i << " failed resident greedy verifier reduction");
                return false;
            }
        }
        const bool every_child_published_in_graph =
            std::all_of(
                child_outcomes.begin(),
                child_outcomes.end(),
                [](const DeviceSpeculativeOutcomeHandle &outcome)
                {
                    return outcome
                        .local_tp_participant_complete;
                });
        if (!every_child_published_in_graph)
        {
            if (tp_ctx_)
                tp_ctx_->requestAbort();
            throw std::runtime_error(
                "Mirrored LocalTP greedy outcomes were not produced locally "
                "by every participant");
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredGreedy;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_greedy_resident_outcomes",
                static_cast<double>(draft_token_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"compare_rows", std::to_string(compare_rows)},
                 {"implementation",
                  "graph_captured_participant_local_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
        const DeviceGreedyBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                requests,
                request_count,
                out_handle);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
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

        if (!out_handle ||
            !requests ||
            request_count <= 0 ||
            device_runners_.size() < 2)
        {
            return false;
        }
        if (!publishesFullVocabularyMTPLogits())
        {
            if (primaryDeviceId().is_gpu())
            {
                LOG_ERROR("[RankOrchestrator] GPU LocalTP MTP resident greedy request batching requires mirrored full-head child verifier outcomes");
            }
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy request-batch MTP requires every child to expose a GPU mirrored-head resident publisher; participant "
                          << i << " is not ready");
                return false;
            }
            if (!child->verifyGreedyAllPositionRequestBatchOutcomesOnDeviceResident(
                    requests,
                    request_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid() ||
                child_outcomes[i].sampling_mode !=
                    DeviceGenerationSamplingMode::Greedy)
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP greedy request-batch MTP child "
                          << i << " failed resident greedy verifier reduction");
                return false;
            }
        }
        if (!validateMirroredLocalTPChildOutcomesComplete(
                child_outcomes,
                "rank_mirrored_localtp_greedy_request_batch_local_outcomes"))
        {
            return false;
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredGreedy;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_greedy_request_batch_resident_outcomes",
                static_cast<double>(request_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "participant_local_device_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::verifyStochasticMirroredLocalTPRequestBatchOutcomesOnDeviceResident(
        const DeviceStochasticBatchOutcomeRequest *requests,
        int request_count,
        DeviceSpeculativeOutcomeHandle *out_handle)
    {
        if (!out_handle)
            return false;
        *out_handle = DeviceSpeculativeOutcomeHandle{};

        if (!requests ||
            request_count <= 0 ||
            device_runners_.size() < 2)
        {
            return false;
        }

        std::vector<DeviceSpeculativeOutcomeHandle> child_outcomes(
            device_runners_.size());
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            IInferenceRunner *child = device_runners_[i].get();
            if (!child ||
                !child->primaryDeviceId().is_gpu() ||
                !child->publishesFullVocabularyMTPLogits() ||
                !child->supportsDeviceStochasticMTPVerification() ||
                !child->supportsDeviceResidentMTPSpecStatePublication())
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic MTP requires every child to expose a GPU mirrored-head resident stochastic publisher; participant "
                          << i << " is not ready");
                return false;
            }

            /*
             * The descriptors are value-owned and already name child-local target
             * distribution slots, draft-token slots, first-token slots, and RNG
             * draws. Forward the exact same descriptor list to every mirrored
             * child so each participant has a local compact mailbox; the
             * device-side broadcast below then replaces those mailbox contents
             * with the primary child's authoritative rank outcome.
             */
            if (!child->verifyStochasticDistributionsRequestBatchOutcomesOnDeviceResident(
                    requests,
                    request_count,
                    &child_outcomes[i]) ||
                !child_outcomes[i].valid() ||
                child_outcomes[i].sampling_mode !=
                    DeviceGenerationSamplingMode::Stochastic)
            {
                LOG_ERROR("[RankOrchestrator] Mirrored LocalTP stochastic MTP child "
                          << i << " failed resident stochastic verifier reduction");
                return false;
            }
        }
        if (!validateMirroredLocalTPChildOutcomesComplete(
                child_outcomes,
                "rank_mirrored_localtp_stochastic_local_outcomes"))
        {
            return false;
        }

        rank_mirrored_child_outcomes_ = std::move(child_outcomes);
        rank_mirrored_primary_outcome_ =
            rank_mirrored_child_outcomes_.front();
        *out_handle = rank_mirrored_primary_outcome_;
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::MirroredStochastic;
        rank_compact_outcome_valid_ = out_handle->valid();
        rank_mirrored_child_outcomes_valid_ = rank_compact_outcome_valid_;

        if (rank_compact_outcome_valid_)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_mirrored_localtp_stochastic_resident_outcomes",
                static_cast<double>(request_count),
                "decode",
                "rank",
                {{"participants", std::to_string(device_runners_.size())},
                 {"implementation", "participant_local_device_outcomes"},
                 {"collective", "none"}});
        }
        return rank_compact_outcome_valid_;
    }

    bool RankOrchestrator::validateMirroredLocalTPChildOutcomesComplete(
        const std::vector<DeviceSpeculativeOutcomeHandle> &child_outcomes,
        const char *context_name)
    {
        const std::string operation_name =
            context_name && context_name[0] != '\0'
                ? std::string(context_name)
                : std::string("rank_mirrored_localtp_local_outcomes");
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("[RankOrchestrator] " << operation_name << ": " << reason);
            return false;
        };

        if (device_runners_.size() < 2 ||
            child_outcomes.size() != device_runners_.size())
        {
            return fail(
                "participant-local outcome validation requires one child handle per LocalTP participant");
        }

        const DeviceSpeculativeOutcomeHandle &primary = child_outcomes.front();
        if (!primary.valid())
            return fail("primary mirrored child outcome is invalid");

        /*
         * Production never copies these compact rows to compare them. The
         * grouped verifier parity suites prove byte identity across the same
         * deterministic local kernels. Runtime validates only the complete
         * ownership contract and fails hard if any participant did not execute
         * the participant-local path.
         */
        for (size_t i = 0; i < child_outcomes.size(); ++i)
        {
            const DeviceSpeculativeOutcomeHandle &child = child_outcomes[i];
            IInferenceRunner *runner = device_runners_[i].get();
            if (!runner)
            {
                return fail("missing mirrored LocalTP participant " +
                            std::to_string(i));
            }
            if (!child.valid())
            {
                return fail("invalid mirrored child outcome for participant " +
                            std::to_string(i));
            }
            if (child.request_count != primary.request_count ||
                child.logical_verifier_rows_per_request !=
                    primary.logical_verifier_rows_per_request ||
                child.physical_verifier_rows_per_request !=
                    primary.physical_verifier_rows_per_request ||
                child.output_token_stride != primary.output_token_stride ||
                child.meta_stride != primary.meta_stride)
            {
                return fail(
                    "mirrored child outcome shape mismatch for participant " +
                    std::to_string(i));
            }
            if (child.sampling_mode != primary.sampling_mode)
            {
                return fail(
                    "mirrored child outcome sampling-mode mismatch for participant " +
                    std::to_string(i));
            }
            if (child.device != runner->primaryDeviceId())
            {
                return fail(
                    "mirrored child outcome device does not match participant " +
                    std::to_string(i));
            }
            if (!child.device.is_gpu())
            {
                return fail(
                    "mirrored participant-local outcomes are GPU-only; participant " +
                    std::to_string(i) + " is " + child.device.toString());
            }
            if (!child.local_tp_participant_complete)
                return fail(
                    "participant " +
                    std::to_string(i) +
                    " did not publish a complete local mirrored outcome");
            if (!child.stream || !child.response_ready_event)
            {
                return fail(
                    "participant " +
                    std::to_string(i) +
                    " did not retain its exact outcome producer stream/event");
            }
        }

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_mirrored_localtp_participant_local_outcomes",
            static_cast<double>(child_outcomes.size()),
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"request_count", std::to_string(primary.request_count)},
             {"implementation", "independent_identical_device_outcomes"},
             {"collective", "none"},
             {"context", operation_name}});
        return true;
    }

    bool RankOrchestrator::buildRankStochasticDistributionFromLocalTP(
        DeviceLogitsSource source,
        int row,
        DeviceDistributionBuffer buffer,
        int slot,
        const SamplingParams &params,
        int vocab_size)
    {
        if (buffer != DeviceDistributionBuffer::Target ||
            device_runners_.size() < 2 ||
            row < 0 ||
            slot < 0 ||
            slot >= rank_stochastic_slot_capacity_ ||
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
            if (!runner)
                return false;

            LogitsLocalInfo info;
            switch (source)
            {
            case DeviceLogitsSource::Main:
                if (!runner->hasLogitsLocal())
                    return false;
                info = runner->consumeLogitsLocalInfoForSampling();
                break;
            case DeviceLogitsSource::MTP:
                if (!runner->hasMTPLogitsLocal())
                    return false;
                info = runner->consumeMTPLogitsLocalInfoForSampling();
                break;
            case DeviceLogitsSource::AllPosition:
                if (!runner->hasAllPositionLogitsLocal())
                    return false;
                info = runner->consumeAllPositionLogitsLocalInfoForSampling();
                break;
            case DeviceLogitsSource::MainRequestBatch:
                // Live request-batch condition rows are mirrored and child-owned.
                // They must never enter the legacy rank-side shard reducer.
                return false;
            }
            if (!info || info.vocab_local == 0)
                return false;
            local_infos.push_back(info);
        }

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
            if (!info.tensor || info.vocab_local == 0)
                return false;

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
            if (local_k <= 0)
                return false;

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
                const int local_index = top_indices[static_cast<size_t>(i)];
                if (local_index < 0)
                    continue;
                const int token =
                    static_cast<int>(vocab_offset) + local_index;
                if (token < 0 || token >= vocab_size)
                    continue;
                candidates.push_back(
                    Candidate{top_values[static_cast<size_t>(i)], token});
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
        if (selected <= 0)
            return false;

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
        rank_stochastic_target_top_k_[static_cast<size_t>(slot)] = selected;
        rank_stochastic_target_sample_tokens_[static_cast<size_t>(slot)] = -1;

        PerfStatsCollector::addCounter(
            "mtp",
            "rank_stochastic_compact_distribution_builds",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"slot", std::to_string(slot)},
             {"row", std::to_string(row)},
             {"top_k", std::to_string(params.top_k)},
             {"implementation", "cross_shard_topk_candidate_merge"}});
        return true;
    }
} // namespace llaminar2
