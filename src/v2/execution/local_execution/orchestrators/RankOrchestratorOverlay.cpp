/**
 * @file RankOrchestratorOverlay.cpp
 * @brief RankOrchestrator MTP publication, prefix cache, overlay maintenance, and device sync.
 */
#include "RankOrchestratorInternal.h"

namespace llaminar2
{

    /**
     * @brief Publish grouped decode-equivalent accepted rows across the rank.
     *
     * The ordinary batch publisher already owns the hard parts: per-request
     * common-prefix clamping, PP shifted-KV ownership, TP worker-pool fan-out,
     * timeout handling, and participant error aggregation.  This method wraps
     * that machinery in a scope that swaps child calls to the grouped API, so
     * the new lane inherits the same correctness coordination without copying a
     * second publication implementation.
     */
    bool RankOrchestrator::publishGroupedDecodeEquivalentMTPSpecStateBatch(
        const MTPSpecStepPlanBatch &plans,
        std::string *error)
    {
        /*
         * The scoped flag is the only difference from a direct batch
         * publication.  It makes publishAcceptedMTPSpecStateBatch() ask each
         * participant's grouped publisher directly, then restores normal
         * direct-publish behavior before returning to the caller.
         */
        struct ScopedGroupedDecodeEquivalentPublication
        {
            bool &slot;
            bool previous;

            explicit ScopedGroupedDecodeEquivalentPublication(bool &slot_in)
                : slot(slot_in), previous(slot_in)
            {
                slot = true;
            }

            ~ScopedGroupedDecodeEquivalentPublication()
            {
                slot = previous;
            }
        } scope(grouped_decode_equivalent_spec_publication_scope_);

        const bool ok = publishAcceptedMTPSpecStateBatch(plans, error);
        if (ok)
        {
            const size_t participant_count =
                !pp_stage_runners_.empty()
                    ? pp_stage_runners_.size()
                    : device_runners_.size();
            PerfStatsCollector::addCounter(
                "mtp",
                "rank_grouped_decode_equivalent_spec_state_batch_publications",
                1.0,
                "decode",
                "rank",
                {{"participants", std::to_string(participant_count)},
                 {"request_count", std::to_string(plans.request_count)}});
        }
        return ok;
    }

    const float *RankOrchestrator::getAllPositionLogits() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->getAllPositionLogits();
        }

        if (device_runners_.empty())
        {
            return nullptr;
        }

        const size_t rows = std::max<size_t>(
            1,
            static_cast<size_t>(
                current_all_position_logit_rows_ > 0
                    ? current_all_position_logit_rows_
                    : std::max(0, current_padded_seq_len_)));

        bool has_local_all_position_logits = false;
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            has_local_all_position_logits =
                has_local_all_position_logits || runner->hasAllPositionLogitsLocal();
        }

        if (has_local_all_position_logits)
        {
            std::vector<LogitsLocalInfo> local_infos;
            local_infos.reserve(device_runners_.size());
            for (const auto &runner : device_runners_)
            {
                if (!runner->hasAllPositionLogitsLocal())
                {
                    LOG_ERROR("RankOrchestrator::getAllPositionLogits: mixed local and replicated verifier logits are invalid");
                    return nullptr;
                }

                LogitsLocalInfo info =
                    runner->consumeAllPositionLogitsLocalInfoForHostGather();
                if (!info)
                    return nullptr;
                local_infos.push_back(info);
            }

            const int full_vocab = vocab_size();
            if (full_vocab <= 0)
                return nullptr;

            const size_t required_elements = rows * static_cast<size_t>(full_vocab);
            if (!all_position_logits_gatherer_ ||
                all_position_logits_gatherer_->bufferNumel() < required_elements)
            {
                all_position_logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    full_vocab,
                    rows,
                    logits_backend_resolver_);
            }

            if (!all_position_logits_gatherer_ ||
                !all_position_logits_gatherer_->gatherLocalInfos(local_infos, rows, full_vocab))
            {
                return nullptr;
            }
            return all_position_logits_gatherer_->data();
        }

        const int compare_vocab = device_runners_[0] ? device_runners_[0]->vocab_size() : 0;
        if (compare_vocab <= 0)
        {
            return nullptr;
        }

        /*
         * Replicated full-vocab verifier logits follow the same public
         * contract as ordinary serial TP decode: child 0 is the authoritative
         * host-visible logits tensor.  Requiring bitwise equality across every
         * child here is stricter than serial decode and makes decode-equivalent
         * verifier rows fail before the parity harness can compare them with
         * the serial path that production sampling actually observes.
         */
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                return nullptr;
            if (runner->vocab_size() != compare_vocab)
                return nullptr;
        }
        return device_runners_[0]->getAllPositionLogits();
    }

    std::string RankOrchestrator::mtpDecodeUnsupportedReason() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            const std::string child_reason =
                pp_sidecar->mtpDecodeUnsupportedReason();
            if (!child_reason.empty())
                return child_reason;
            return {};
        }
        if (device_runners_.size() > 1)
        {
            for (const auto &runner : device_runners_)
            {
                if (!runner)
                    return "MTP decode requires every TP participant to be available";

                const std::string child_reason = runner->mtpDecodeUnsupportedReason();
                if (!child_reason.empty())
                    return child_reason;
            }
        }
        return {};
    }

    bool RankOrchestrator::supportsMTPSidecarLogitsStreamHandoff() const
    {
        if (const IInferenceRunner *tail = finalPPSidecarRunner())
            return pp_device_generation_ && pp_device_generation_->prefillPrepared() &&
                tail->supportsMTPSidecarLogitsStreamHandoff();
        if (device_runners_.empty())
            return false;
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPSidecarLogitsStreamHandoff();
            });
    }

    bool RankOrchestrator::supportsMTPDeviceDraftTokenInput() const
    {
        if (const IInferenceRunner *tail = finalPPSidecarRunner())
            return pp_device_generation_ && pp_device_generation_->prefillPrepared() &&
                tail->supportsMTPDeviceDraftTokenInput();
        if (device_runners_.empty())
            return false;
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPDeviceDraftTokenInput();
            });
    }

    bool RankOrchestrator::supportsMTPSidecarPreservesMainState() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsMTPSidecarPreservesMainState();
        }
        if (device_runners_.empty())
            return false;

        /*
         * LocalTP runs the MTP sidecar on every participant in lockstep.  The
         * rank-level capability is therefore a domain-wide AND: a single child
         * that cannot preserve main state makes the whole TP sidecar unsafe for
         * verifier-base reuse.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPSidecarPreservesMainState();
            });
    }

    bool RankOrchestrator::supportsMTPShiftedRowReuseFromSidecar() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsMTPShiftedRowReuseFromSidecar();
        }
        if (device_runners_.empty())
            return false;

        /*
         * Reusing the first shifted MTP KV row is only correct when every TP
         * shard can keep its sidecar row as the accepted row-zero publication
         * boundary.  Treat it as a rank-wide all-participant capability rather
         * than a property of participant 0.
         */
        return std::all_of(
            device_runners_.begin(),
            device_runners_.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner->supportsMTPShiftedRowReuseFromSidecar();
            });
    }

    bool RankOrchestrator::applyPenaltiesOnDevice(
        const LogitPenaltyBatch &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesOnDevice(penalties, vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesOnDevice(
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        /*
         * Column-parallel logits are rank-wide state.  Each child runner
         * applies the same sparse global-token penalty map and internally
         * ignores tokens outside its vocab shard.  Returning false here for
         * non-empty maps made temperature-zero LocalTP MTP fail as soon as
         * model defaults enabled repetition penalties.
         */
        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesOnDevice(penalties, vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::applyPenaltiesToMTPLogitsOnDevice(
        const LogitPenaltyBatch &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesToMTPLogitsOnDevice(
                penalties,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesToMTPLogitsOnDevice(
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        /*
         * MTP sidecar logits are sharded exactly like the main LM head in
         * LocalTP.  Fan out the global sparse penalties to every participant so
         * the next rank-level sampler sees a coherent penalty-mutated row.
         */
        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesToMTPLogitsOnDevice(
                     penalties,
                     vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::applyPenaltiesToAllPositionLogitsOnDeviceRow(
        int row,
        const LogitPenaltyBatch &penalties,
        int vocab_size)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                row,
                penalties,
                vocab_size);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                row,
                penalties,
                vocab_size);
        }
        if (penalties.empty())
            return true;

        bool ok = true;
        for (const auto &runner : device_runners_)
        {
            ok = runner &&
                 runner->applyPenaltiesToAllPositionLogitsOnDeviceRow(
                     row,
                     penalties,
                     vocab_size) &&
                 ok;
        }
        return ok;
    }

    bool RankOrchestrator::applyDeviceOwnedMTPPenaltiesToLogitRows(
        DeviceLogitsSource source,
        int row_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyDeviceOwnedMTPPenaltiesToLogitRows(
                source,
                row_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->applyDeviceOwnedMTPPenaltiesToLogitRows(
                    source,
                    row_count,
                    penalty_policy);
        }
        if (device_runners_.empty() || row_count <= 0)
            return false;

        if (source == DeviceLogitsSource::Main)
        {
            IInferenceRunner *primary = device_runners_[0].get();
            return primary &&
                   primary->applyDeviceOwnedMTPPenaltiesToLogitRows(
                       source,
                       row_count,
                       penalty_policy);
        }
        if (source != DeviceLogitsSource::AllPosition ||
            !publishesFullVocabularyMTPLogits())
        {
            LOG_ERROR("[RankOrchestrator] Device-owned stochastic verifier penalties require mirrored full-vocabulary LocalTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->applyDeviceOwnedMTPPenaltiesToLogitRows(
                    source,
                    row_count,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Device-owned stochastic verifier penalty transform failed on participant "
                          << participant);
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::applyDeviceOwnedMTPBranchPenaltiesToLogits(
        int prior_draft_count,
        const MTPRequestPenaltyPolicy &penalty_policy)
    {
        if (IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                prior_draft_count,
                penalty_policy);
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]
                ->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                    prior_draft_count,
                    penalty_policy);
        }
        if (device_runners_.empty() || prior_draft_count < 0 ||
            !publishesFullVocabularyMTPLogits())
        {
            LOG_ERROR("[RankOrchestrator] Device-owned MTP branch penalties require mirrored full-vocabulary LocalTP heads");
            return false;
        }

        for (size_t participant = 0;
             participant < device_runners_.size();
             ++participant)
        {
            IInferenceRunner *child = device_runners_[participant].get();
            if (!child ||
                !child->applyDeviceOwnedMTPBranchPenaltiesToLogits(
                    prior_draft_count,
                    penalty_policy))
            {
                LOG_ERROR("[RankOrchestrator] Device-owned MTP branch penalty transform failed on participant "
                          << participant);
                return false;
            }
        }
        return true;
    }

    bool RankOrchestrator::supportsRowLocalAllPositionPenaltyApplication() const
    {
        if (const IInferenceRunner *pp_sidecar = finalPPSidecarRunner())
        {
            return pp_sidecar->supportsRowLocalAllPositionPenaltyApplication();
        }
        if (device_runners_.size() == 1 && device_runners_[0])
        {
            return device_runners_[0]->supportsRowLocalAllPositionPenaltyApplication();
        }

        /*
         * Multi-child TP/PP can promote penalty-greedy compact verification
         * only after every participant can mutate its verifier shard with the
         * same branch-local sampler history.  The compact reducer is a separate
         * capability, so callers still need to check that too.
         */
        if (device_runners_.empty())
            return false;
        for (const auto &runner : device_runners_)
        {
            if (!runner ||
                !runner->supportsRowLocalAllPositionPenaltyApplication())
            {
                return false;
            }
        }
        return true;
    }

    void RankOrchestrator::setSkipLogitsGatherDecode(bool skip)
    {
        skip_logits_gather_decode_ = skip;
        applyLogitsGatherSkipFlags();
    }

    void RankOrchestrator::setSkipLogitsGatherPrefill(bool skip)
    {
        skip_logits_gather_prefill_ = skip;
        applyLogitsGatherSkipFlags();
    }

    bool RankOrchestrator::waitForLastInferenceCompletionForBenchmark()
    {
        const auto &participants =
            !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
        {
            LOG_ERROR("[RankOrchestrator] Benchmark completion boundary has no participants");
            return false;
        }

        /*
         * forwardTP() submits every participant before returning. Waiting here
         * is therefore safe for captured NCCL/RCCL graphs: no rank can be held
         * before its peer graph is launched. Each child waits on its own exact
         * terminal event, so this does not widen into a device synchronization.
         */
        for (size_t participant = 0; participant < participants.size(); ++participant)
        {
            if (!participants[participant] ||
                !participants[participant]
                     ->waitForLastInferenceCompletionForBenchmark())
            {
                LOG_ERROR("[RankOrchestrator] Benchmark terminal-event wait failed for participant "
                          << participant);
                return false;
            }
        }
        return true;
    }

    void RankOrchestrator::setMTPAllPositionVerifierSyncDeferralEnabled(bool enabled)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setMTPAllPositionVerifierSyncDeferralEnabled(enabled);
        }
    }

    void RankOrchestrator::setMTPMainDecodeSyncDeferralEnabled(bool enabled)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setMTPMainDecodeSyncDeferralEnabled(enabled);
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setMTPMainDecodeSyncDeferralEnabled(enabled);
        }
    }

    void RankOrchestrator::applyLogitsGatherSkipFlags()
    {
        if (logits_gatherer_)
        {
            logits_gatherer_->setSkipDecode(skip_logits_gather_decode_);
            logits_gatherer_->setSkipPrefill(skip_logits_gather_prefill_);
        }

        for (auto &runner : device_runners_)
        {
            if (!runner)
                continue;
            runner->setSkipLogitsGatherDecode(skip_logits_gather_decode_);
            runner->setSkipLogitsGatherPrefill(skip_logits_gather_prefill_);
        }

        for (auto &runner : pp_stage_runners_)
        {
            if (!runner)
                continue;
            runner->setSkipLogitsGatherDecode(skip_logits_gather_decode_);
            runner->setSkipLogitsGatherPrefill(skip_logits_gather_prefill_);
        }
    }

    void RankOrchestrator::wireLocalTPMoERuntimeHistogramSyncs()
    {
        if (local_tp_moe_histogram_syncs_wired_)
            return;
        local_tp_moe_histogram_syncs_wired_ = true;

        if (!tp_ctx_ || tp_ctx_->degree() <= 1 || device_runners_.size() <= 1)
            return;
        if (moeOverlayAuthorityExecution() ==
            MoEOverlayAuthorityExecutionKind::
                DeviceResident)
        {
            LOG_DEBUG("RankOrchestrator: skipping host MoE runtime histogram bridge; "
                      "device-resident ExpertOverlay authority owns histogram allgather");
            return;
        }

        std::vector<MoERebalanceController *> controllers;
        controllers.reserve(device_runners_.size());
        for (const auto &runner : device_runners_)
        {
            if (!runner)
                continue;
            for (auto *controller : runner->moeRebalanceControllers())
            {
                if (controller &&
                    std::find(controllers.begin(), controllers.end(), controller) == controllers.end())
                {
                    controllers.push_back(controller);
                }
            }
        }

        auto *active = selectActiveMoERebalanceController(controllers);
        if (!active || !active->histogram())
            return;

        DecodeExpertHistogram *active_histogram = active->histogram();
        const std::string domain_id = active->domainId();
        const int num_layers = active->numLayers();
        const int num_experts = active->numExperts();
        int sibling_count = 0;

        for (auto *source_controller : controllers)
        {
            if (!source_controller || source_controller == active)
                continue;
            if (source_controller->domainId() != domain_id)
                continue;
            if (source_controller->numLayers() != num_layers ||
                source_controller->numExperts() != num_experts)
            {
                LOG_WARN("RankOrchestrator: skipping LocalTP MoE runtime histogram bridge for domain "
                         << domain_id << " because controller dimensions differ");
                continue;
            }

            DecodeExpertHistogram *source_histogram = source_controller->histogram();
            if (!source_histogram || source_histogram == active_histogram)
                continue;

            const int source_participant = ++sibling_count;
            active_histogram->registerRuntimeHistogramSync(
                [active_histogram,
                 source_histogram,
                 domain_id,
                 num_layers,
                 num_experts,
                 source_participant]() -> bool
                {
                    if (!source_histogram->syncRuntimeHistograms())
                        return false;

                    uint64_t activations_merged = 0;
                    for (int layer = 0; layer < num_layers; ++layer)
                    {
                        auto counts = source_histogram->layerHistogram(layer);
                        for (uint64_t count : counts)
                            activations_merged += count;
                        active_histogram->mergeLayerCounts(
                            layer,
                            counts.data(),
                            num_experts,
                            /*count_window_tokens=*/false);
                    }

                    source_histogram->resetWindow();

                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_runtime_histogram_sibling_syncs",
                        1.0,
                        "rebalance",
                        {},
                        {{"domain_id", domain_id},
                         {"source_participant", std::to_string(source_participant)}});
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_runtime_histogram_sibling_activations",
                        static_cast<double>(activations_merged),
                        "rebalance",
                        {},
                        {{"domain_id", domain_id},
                         {"source_participant", std::to_string(source_participant)}});
                    return true;
                });

            PerfStatsCollector::addCounter(
                "moe_rebalance",
                "rank_runtime_histogram_sibling_sync_registrations",
                1.0,
                "rebalance",
                primaryDeviceId().toString(),
                {{"domain_id", domain_id},
                 {"source_participant", std::to_string(source_participant)}});
        }

        if (sibling_count > 0)
        {
            LOG_DEBUG("RankOrchestrator: wired " << sibling_count
                                                 << " LocalTP sibling MoE runtime histogram sync(s) for domain "
                                                 << domain_id);
        }
    }

    bool RankOrchestrator::forward_batch(const std::vector<std::vector<int>> &token_batches)
    {
        return forwardHostTokenBatchAcrossDevices(
            token_batches,
            &IInferenceRunner::forward_batch,
            "forward_batch");
    }

    bool RankOrchestrator::forwardHostTokenBatchAcrossDevices(
        const std::vector<std::vector<int>> &token_batches,
        bool (IInferenceRunner::*entrypoint)(
            const std::vector<std::vector<int>> &),
        const char *operation)
    {
        if (device_runners_.empty())
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                            << ": No device runners available");
            return false;
        }
        if (!entrypoint || !operation)
        {
            LOG_ERROR(
                "RankOrchestrator::forwardHostTokenBatchAcrossDevices: missing "
                "typed child entrypoint or operation identity");
            return false;
        }

        prepareRetainedSourceAdmission();
        RetainedSourceAdmission::Attempt source_attempt{*this,retained_source_admission_,__func__};

        LOG_DEBUG("RankOrchestrator::" << operation
                                        << ": batch_size=" << token_batches.size()
                                        << ", devices=" << device_runners_.size());

        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
            {
                LOG_ERROR("RankOrchestrator::" << operation
                                                << ": Missing device runner "
                                                << i);
                return false;
            }
        }

        /*
         * Host-token batch calls are request-hot for CPU generation and
         * verifier paths. They use the persistent LocalTP participant threads;
         * per-request std::async fan-out grows allocator arenas and loses the
         * exact profiler/device thread binding established by ordinary replay.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ =
                makeParticipantWorkerPool(device_runners_);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback([this, operation]()
                {
                    LOG_WARN("[TPWorkerPool] " << operation
                                                << " failure detected - aborting collective backend");
                    tp_ctx_->requestAbort();
                });
            }
        }
        if (tp_worker_pool_->numWorkers() != device_runners_.size())
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                            << ": Worker topology does not match "
                                               "the participant count");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this,
             &token_batches,
             entrypoint,
             kernel_phase,
             rocm_phase,
             cuda_phase,
             kv_phase,
             executor_phase](std::size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);
                const DeviceId device =
                    device_runners_[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return (device_runners_[i].get()->*entrypoint)(token_batches);
            });

        bool all_success = true;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        const int collect_timeout_ms = effectiveTPWorkerJoinTimeoutMs();
        const auto results = tp_worker_pool_->collectAll(collect_timeout_ms);
        for (const auto &result : results)
        {
            if (!result.completed)
            {
                all_success = false;
                if (tp_ctx_)
                    tp_ctx_->requestAbort();
                continue;
            }
            if (result.exception && !first_exception)
            {
                all_success = false;
                first_exception = result.exception;
                first_exception_device = result.worker_index;
            }
            if (!result.success)
            {
                LOG_ERROR("RankOrchestrator::" << operation
                                                << ": Device "
                                                << result.worker_index
                                                << " forward failed");
                all_success = false;
            }
        }
        if (!all_success && collect_timeout_ms > 0 &&
            tp_worker_pool_->completedCount() != tp_worker_pool_->numWorkers())
        {
            abortAfterTPWorkerTimeout(
                operation,
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("RankOrchestrator::" << operation
                                            << ": Re-throwing primary exception from device "
                                            << first_exception_device);
            std::rethrow_exception(first_exception);
        }

        if (all_success)
        {
            // Update batch tracking from primary device
            if (!device_runners_.empty() && device_runners_[0])
            {
                current_batch_size_ = device_runners_[0]->batch_size();
                current_padded_seq_len_ = device_runners_[0]->padded_seq_len();
                current_sequence_lengths_ = device_runners_[0]->sequence_lengths();
            }
            stats_dirty_ = true;
        }

        source_attempt.keep=all_success;
        return all_success;
    }

    const float *RankOrchestrator::getLogits(int seq_idx) const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getLogits(seq_idx);
        }
        return nullptr;
    }

    int RankOrchestrator::batch_size() const
    {
        return current_batch_size_;
    }

    int RankOrchestrator::padded_seq_len() const
    {
        return current_padded_seq_len_;
    }

    const std::vector<int> &RankOrchestrator::sequence_lengths() const
    {
        return current_sequence_lengths_;
    }

    int RankOrchestrator::vocab_size() const
    {
        // For tensor-parallel setups, the LM head may be column-sharded across devices.
        // In that case, each device has logits for vocab_size/tp_degree tokens.
        // We should return the FULL vocab size (sum of all devices), not just device 0's.
        //
        // The model_ctx_ always has the true total vocab size from the model metadata.
        if (model_ctx_)
        {
            return static_cast<int>(model_ctx_->vocabSize());
        }

        // Fallback: if no model context, use device 0's vocab (may be wrong for TP)
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->vocab_size();
        }
        return 0;
    }

    void RankOrchestrator::resetInferenceState(const InferenceStateResetRequest &request)
    {
        if (request.boundary != InferenceStateResetRequest::Boundary::Request ||
            !request.resetsAllLiveRequestOwners() ||
            !request.reset_model_runtime ||
            !request.preserve_replay_safe_graphs)
        {
            throw std::invalid_argument(
                "RankOrchestrator::resetInferenceState currently supports only full "
                "request-boundary resets that preserve replay-safe graph captures");
        }
        LOG_DEBUG("RankOrchestrator::resetInferenceState: resetting request-owned state on "
                  << device_runners_.size() << " TP devices and "
                  << pp_stage_runners_.size() << " PP stages"
                  << " reason=" << (request.reason ? request.reason : "request-boundary"));

        if (retained_source_admission_ && retained_source_admission_->retired_failure)
            std::rethrow_exception(retained_source_admission_->retired_failure);

        // Close publication is a separate native-proof phase. A missing proof
        // still fails before any participant's data reset can start.
        if(retained_source_admission_){
            try{
                retireOriginalStorageParticipants(device_runners_,[&](const auto& runner){
                    if(auto* device=dynamic_cast<DeviceGraphOrchestrator*>(runner.get()))
                        device->prepareGroupedRetainedStorageReset(request);
                },[]{},[]{});
            }catch(...){
                const auto original=std::current_exception();
                // Missing producer proof cannot release source/native owners or
                // make this request reusable. Withdraw the collected TP callback
                // before any future teardown, retaining this exact failed borrow
                // and the original proof error even if callback retirement fails.
                retained_source_admission_->failed=true;
                try { retireRetainedSourceFailureCallback(); }
                catch(const std::exception& error) {
                    LOG_ERROR("[RankOrchestrator] Failed reset could not retire its original callback: " << error.what());
                }
                catch(...) { LOG_ERROR("[RankOrchestrator] Failed reset could not retire its original callback"); }
                std::rethrow_exception(original);
            }
            std::exception_ptr cancellationFailure;
            if (retained_source_admission_->failed) {
                try { retained_source_admission_->original->cancelStorageChannelsAfterSourceProof(); }
                catch (...) { cancellationFailure = std::current_exception(); }
            }
            // Drain every adopted right before full DGO reset: its generic
            // reset transaction deliberately terminates on missing prior work.
            // A recoverable service failure must not enter that data-clearing
            // transaction or prevent an original peer from retiring its right.
            awaitRetainedSourceAdmission(cancellationFailure);
            // Only successful original IO retirement permits model-state clear.
            retireOriginalStorageParticipants(device_runners_,[&](const auto& runner){
                if(runner)runner->resetInferenceState(request);
            },[]{},[]{});
        }else{
            for(auto& runner:device_runners_)if(runner)runner->resetInferenceState(request);
        }

        // Reset PP stage runners (each has its own KV/GDN owner).
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->resetInferenceState(request);
            }
        }

        if (rank_local_expert_endpoints_)
            rank_local_expert_endpoints_->resetInferenceState(request);

        current_position_ = 0;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            0);
        std::fill(
            rank_stochastic_target_token_ids_.begin(),
            rank_stochastic_target_token_ids_.end(),
            -1);
        std::fill(
            rank_stochastic_target_probs_.begin(),
            rank_stochastic_target_probs_.end(),
            0.0f);
        std::fill(
            rank_stochastic_target_top_k_.begin(),
            rank_stochastic_target_top_k_.end(),
            0);
        std::fill(
            rank_stochastic_target_sample_tokens_.begin(),
            rank_stochastic_target_sample_tokens_.end(),
            -1);
        std::fill(
            rank_stochastic_draft_sample_tokens_.begin(),
            rank_stochastic_draft_sample_tokens_.end(),
            -1);
        std::fill(
            rank_mirrored_target_distribution_ready_.begin(),
            rank_mirrored_target_distribution_ready_.end(),
            false);
        rank_stochastic_staged_draft_tokens_.clear();
        std::fill(
            rank_compact_output_tokens_.begin(),
            rank_compact_output_tokens_.end(),
            -1);
        rank_compact_output_meta_.fill(0);
        rank_compact_outcome_kind_ = RankCompactOutcomeKind::None;
        rank_compact_last_draft_tokens_.clear();
        rank_compact_last_stop_tokens_.clear();
        rank_mtp_verifier_child_token_inputs_.clear();
        rank_mtp_verifier_child_token_count_ = 0;
        rank_compact_outcome_valid_ = false;
        rank_mirrored_child_outcomes_.clear();
        rank_mirrored_primary_outcome_ = DeviceSpeculativeOutcomeHandle{};
        rank_mirrored_child_outcomes_valid_ = false;
        rank_hosted_device_moe_rebalance_tickets_.fill(
            DeviceMoERebalanceDispatchTicket{});
        rank_hosted_device_moe_rebalance_ticket_count_ = 0;
        rank_hosted_device_moe_rebalance_observation_schedule_ = {};
        rank_hosted_device_moe_rounds_until_observation_ = 0u;
        rank_hosted_device_moe_observation_schedule_initialized_ = false;
        invalidateRankResidentLogicalStateAggregate(
            "request_reset",
            request.reason ? request.reason : "request_boundary");
        stats_dirty_ = true;
    }

    void RankOrchestrator::clear_cache()
    {
        resetInferenceState(
            InferenceStateResetRequest::requestBoundary("clear_cache"));
    }

    bool RankOrchestrator::purgePrefixCache()
    {
        /* Every LocalTP/PP child owns a shard of the same logical archive. */
        for (auto &runner : device_runners_)
        {
            if (runner && !runner->purgePrefixCache())
                return false;
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner && !runner->purgePrefixCache())
                return false;
        }
        return true;
    }

    DeviceMoERebalanceMaintenanceExecutionPolicy
    RankOrchestrator::deviceMoERebalanceMaintenanceExecutionPolicy()
        const noexcept
    {
        /* Pipeline stages are independent maintenance domains. The outer rank
         * fans their complete boundary calls out concurrently and never tries
         * to compare participant IDs across unrelated LocalTP domains. */
        if (!pp_stage_runners_.empty())
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;
        if (device_runners_.empty())
            return DeviceMoERebalanceMaintenanceExecutionPolicy::Inactive;

        DeviceMoERebalanceMaintenanceExecutionPolicy policy =
            device_runners_.front()
                ? device_runners_.front()
                      ->deviceMoERebalanceMaintenanceExecutionPolicy()
                : DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            const auto participant_policy =
                device_runners_[index]
                    ? device_runners_[index]
                          ->deviceMoERebalanceMaintenanceExecutionPolicy()
                    : DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
            if (participant_policy != policy)
                return DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported;
        }
        return policy;
    }

    DeviceMoERebalanceHostedObservationSchedule
    RankOrchestrator::deviceMoERebalanceHostedObservationSchedule()
        const noexcept
    {
        if (!pp_stage_runners_.empty() || device_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !device_runners_.front())
        {
            return {};
        }

        const auto schedule = device_runners_.front()
                                  ->deviceMoERebalanceHostedObservationSchedule();
        if (!schedule.valid())
            return {};
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            if (!device_runners_[index] ||
                device_runners_[index]
                        ->deviceMoERebalanceHostedObservationSchedule() !=
                    schedule)
            {
                return {};
            }
        }
        return schedule;
    }

    std::vector<MoEOverlayDeviceControllerRuntimeBinding>
    RankOrchestrator::moeOverlayDeviceControllerRuntimeBindings() const
    {
        std::vector<MoEOverlayDeviceControllerRuntimeBinding> result;
        const auto append = [&result](const auto &runners)
        {
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;
                auto child =
                    runner->moeOverlayDeviceControllerRuntimeBindings();
                result.insert(
                    result.end(), child.begin(), child.end());
            }
        };
        append(device_runners_);
        append(pp_stage_runners_);
        if (rank_local_expert_endpoints_)
        {
            auto endpoints = rank_local_expert_endpoints_->moeOverlayDeviceControllerRuntimeBindings();
            result.insert(result.end(), endpoints.begin(), endpoints.end());
        }
        std::sort(
            result.begin(),
            result.end(),
            [](const auto &left, const auto &right)
            {
                return left.overlay_participant_id <
                       right.overlay_participant_id;
            });
        for (std::size_t index = 0u; index < result.size(); ++index)
        {
            if (!result[index].valid() ||
                (index != 0u &&
                 result[index - 1u].overlay_participant_id ==
                     result[index].overlay_participant_id))
            {
                throw std::logic_error(
                    "RankOrchestrator collected invalid or duplicate ExpertOverlay controller runtime bindings");
            }
        }
        return result;
    }

    bool RankOrchestrator::installMoEOverlayTransferProgressEpoch(
        std::shared_ptr<MappedTransferProgressEpoch> epoch)
    {
        if (!epoch || !epoch->device().is_gpu())
        {
            LOG_ERROR(
                "RankOrchestrator received an invalid ExpertOverlay transfer-progress epoch");
            return false;
        }

        IInferenceRunner *owner = nullptr;
        const auto find_owner = [&](auto &runners) -> bool
        {
            for (auto &runner : runners)
            {
                if (!runner || runner->primaryDeviceId() != epoch->device())
                    continue;
                if (owner)
                {
                    LOG_ERROR(
                        "RankOrchestrator found duplicate device owners for ExpertOverlay transfer progress on "
                        << epoch->device().toString());
                    return false;
                }
                owner = runner.get();
            }
            return true;
        };
        if (!find_owner(device_runners_) || !find_owner(pp_stage_runners_)) return false;
        if (rank_local_expert_endpoints_)
        {
            const auto devices = rank_local_expert_endpoints_->localDevices();
            if (std::find(devices.begin(), devices.end(), epoch->device()) != devices.end())
            {
                if (owner)
                {
                    LOG_ERROR("Rank auxiliary expert device aliases a continuation parent");
                    return false;
                }
                owner = rank_local_expert_endpoints_.get();
            }
        }
        if (!owner)
        {
            LOG_ERROR(
                "RankOrchestrator could not resolve the exact child for ExpertOverlay transfer progress on "
                << epoch->device().toString());
            return false;
        }
        return owner->installMoEOverlayTransferProgressEpoch(std::move(epoch));
    }

    bool RankOrchestrator::
        submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary()
    {
        if (!pp_stage_runners_.empty() || device_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            !deviceMoERebalanceHostedObservationSchedule().valid() ||
            rank_hosted_device_moe_rebalance_ticket_count_ != 0u)
        {
            LOG_ERROR("[RankOrchestrator] HIP known-non-due MoE boundary has no exact idle LocalTP schedule");
            return false;
        }

        PerfStatsCollector::ScopedTimer submission_timer(
            "moe_rebalance",
            "rank_device_moe_known_non_due_boundary_submission",
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"dispatch", "serial_non_collective"}});

        /* Each participant's complete decode/MTP graph already owns the device
         * cadence edge. This loop validates the immutable embedded-topology
         * contract only; routing it through the persistent TP worker pool would
         * add a host wake/collect round-trip despite having no GPU or collective
         * submission. The due path below still uses concurrent workers because
         * its maintenance graph contains real RCCL collectives. */
        for (size_t index = 0; index < device_runners_.size(); ++index)
        {
            IInferenceRunner *const runner = device_runners_[index].get();
            if (!runner ||
                !runner
                     ->submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary())
            {
                /* Earlier participants may already have advanced their device
                 * clocks. There is no recoverable continuation that can prove
                 * participant symmetry, so fail fatally at the first partial
                 * submission instead of entering the next inference round. */
                LOG_ERROR("[RankOrchestrator] HIP known-non-due MoE boundary failed after serial participant submission began"
                          << " participant=" << index);
                std::terminate();
            }
        }

        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_ticket_observations_elided",
            1.0,
            "decode",
            "rank",
             {{"participants", std::to_string(device_runners_.size())},
             {"submission", "embedded_decode_graph_noop"}});
        return true;
    }

    bool RankOrchestrator::observeDeviceMoERebalanceDispatchTicket(
        DeviceMoERebalanceDispatchTicket *out_ticket)
    {
        if (out_ticket)
            *out_ticket = DeviceMoERebalanceDispatchTicket{};
        if (!out_ticket || !pp_stage_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            device_runners_.empty() ||
            device_runners_.size() >
                rank_hosted_device_moe_rebalance_tickets_.size() ||
            rank_hosted_device_moe_rebalance_ticket_count_ != 0u)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket observation has no exact, idle LocalTP policy");
            return false;
        }

        if (device_runners_.size() == 1u)
        {
            if (!device_runners_.front() ||
                !device_runners_.front()
                     ->observeDeviceMoERebalanceDispatchTicket(
                         &rank_hosted_device_moe_rebalance_tickets_.front()))
            {
                return false;
            }
        }
        else
        {
            if (!tp_worker_pool_)
            {
                tp_worker_pool_ =
                    makeParticipantWorkerPool(device_runners_);
                if (tp_ctx_)
                {
                    tp_worker_pool_->setFailureCallback(
                        [this]()
                        {
                            LOG_WARN("[TPWorkerPool] HIP Device MoE ticket observation failure detected - aborting collective backend");
                            tp_ctx_->requestAbort();
                        });
                }
            }
            if (tp_worker_pool_->numWorkers() != device_runners_.size())
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket participant count does not match the persistent worker pool");
                return false;
            }

            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    if (index >= device_runners_.size() ||
                        !device_runners_[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        device_runners_[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return device_runners_[index]
                        ->observeDeviceMoERebalanceDispatchTicket(
                            &rank_hosted_device_moe_rebalance_tickets_[index]);
                });

            bool all_success = true;
            std::exception_ptr first_exception;
            size_t first_exception_device = 0;
            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            bool worker_timeout = false;
            auto results = tp_worker_pool_->collectAll(timeout_ms);
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
            if (worker_timeout && timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "observeDeviceMoERebalanceDispatchTicket",
                    timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }
            if (first_exception)
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE ticket observation re-throwing participant exception from "
                          << first_exception_device);
                std::rethrow_exception(first_exception);
            }
            if (!all_success)
                return false;
        }

        const auto &authoritative =
            rank_hosted_device_moe_rebalance_tickets_.front();
        for (size_t index = 1; index < device_runners_.size(); ++index)
        {
            const auto &candidate =
                rank_hosted_device_moe_rebalance_tickets_[index];
            if (!authoritative.hasSameDispatchDecision(candidate))
            {
                LOG_ERROR("[RankOrchestrator] HIP Device MoE controllers published divergent maintenance decisions"
                          << " participant=" << index
                          << " authoritative_committed="
                          << authoritative.decode_rounds_committed
                          << " divergent_committed="
                          << candidate.decode_rounds_committed
                          << " authoritative_remaining="
                          << authoritative.decode_rounds_until_maintenance
                          << " divergent_remaining="
                          << candidate.decode_rounds_until_maintenance
                          << " authoritative_due="
                          << authoritative.maintenance_due
                          << " divergent_due=" << candidate.maintenance_due
                          << " authoritative_error="
                          << authoritative.error_code
                          << " divergent_error=" << candidate.error_code);
                std::terminate();
            }
        }

        rank_hosted_device_moe_rebalance_ticket_count_ =
            device_runners_.size();
        *out_ticket = authoritative;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_dispatch_tickets_validated",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"maintenance_due",
              authoritative.maintenance_due ? "true" : "false"},
             {"bytes_per_participant",
              std::to_string(sizeof(DeviceMoERebalanceDispatchTicket))}});
        return true;
    }

    bool RankOrchestrator::
        submitHostScheduledDeviceMoERebalanceMaintenance(
            const DeviceMoERebalanceDispatchTicket &ticket)
    {
        if (!pp_stage_runners_.empty() ||
            deviceMoERebalanceMaintenanceExecutionPolicy() !=
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance ||
            device_runners_.empty() ||
            rank_hosted_device_moe_rebalance_ticket_count_ !=
                device_runners_.size() ||
            !ticket.hasSameDispatchDecision(
                rank_hosted_device_moe_rebalance_tickets_.front()))
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission rejected a stale rank decision");
            return false;
        }

        if (!tp_worker_pool_ && device_runners_.size() > 1u)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission has no persistent participant pool");
            return false;
        }

        bool all_success = true;
        std::exception_ptr first_exception;
        size_t first_exception_device = 0;
        if (device_runners_.size() == 1u)
        {
            all_success = device_runners_.front() &&
                          device_runners_.front()
                              ->submitHostScheduledDeviceMoERebalanceMaintenance(
                                  rank_hosted_device_moe_rebalance_tickets_
                                      .front());
        }
        else
        {
            const auto kernel_phase = KernelProfiler::getCurrentPhase();
            const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
            const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
            const auto kv_phase = KVCacheProfiler::getCurrentPhase();
            const auto executor_phase = GraphExecutorStats::currentPhase();
            tp_worker_pool_->dispatch(
                [this, kernel_phase, rocm_phase, cuda_phase, kv_phase,
                 executor_phase](size_t index) -> bool
                {
                    KernelProfiler::setCurrentPhase(kernel_phase);
                    ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                    CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                    KVCacheProfiler::setCurrentPhase(kv_phase);
                    GraphExecutorStats::setCurrentPhase(executor_phase);
                    if (index >= device_runners_.size() ||
                        !device_runners_[index])
                    {
                        return false;
                    }
                    const DeviceId device =
                        device_runners_[index]->primaryDeviceId();
                    ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                    CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                    return device_runners_[index]
                        ->submitHostScheduledDeviceMoERebalanceMaintenance(
                            rank_hosted_device_moe_rebalance_tickets_[index]);
                });

            const int timeout_ms = effectiveTPWorkerJoinTimeoutMs();
            bool worker_timeout = false;
            auto results = tp_worker_pool_->collectAll(timeout_ms);
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
            if (worker_timeout && timeout_ms > 0)
            {
                abortAfterTPWorkerTimeout(
                    "submitHostScheduledDeviceMoERebalanceMaintenance",
                    timeout_ms,
                    tp_worker_pool_->completedCount(),
                    tp_worker_pool_->numWorkers());
            }
        }

        /* Any participant may have enqueued a collective or acknowledgement
         * before a sibling reported failure. Continuing would leave an
         * unknowable distributed timeline, so post-dispatch failure is fatal. */
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission re-throwing participant exception from "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        if (!all_success)
        {
            LOG_ERROR("[RankOrchestrator] HIP Device MoE maintenance submission failed after distributed dispatch began");
            std::terminate();
        }

        rank_hosted_device_moe_rebalance_ticket_count_ = 0u;
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_device_moe_rebalance_host_scheduled_submissions",
            1.0,
            "decode",
            "rank",
            {{"participants", std::to_string(device_runners_.size())},
             {"maintenance_due",
              ticket.maintenance_due ? "true" : "false"},
             {"submission", "all_participants_before_next_observation"}});
        return true;
    }

    bool RankOrchestrator::maybeApplyDecodeBoundaryMaintenance(
        uint64_t committed_tokens)
    {
        if (committed_tokens == 0u)
        {
            LOG_ERROR("[RankOrchestrator] Device MoE maintenance cannot retire an empty decode transaction");
            return false;
        }
        const bool use_pp_participants = !pp_stage_runners_.empty();
        auto &participants =
            use_pp_participants ? pp_stage_runners_ : device_runners_;
        if (participants.empty())
            return true;
        if (!use_pp_participants)
        {
            const auto policy =
                deviceMoERebalanceMaintenanceExecutionPolicy();
            if (policy ==
                DeviceMoERebalanceMaintenanceExecutionPolicy::
                    HostScheduledCapturedMaintenance)
            {
                const auto schedule =
                    deviceMoERebalanceHostedObservationSchedule();
                if (!schedule.valid())
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE maintenance has no shared immutable observation schedule");
                    return false;
                }
                if (!rank_hosted_device_moe_observation_schedule_initialized_)
                {
                    rank_hosted_device_moe_rebalance_observation_schedule_ =
                        schedule;
                    rank_hosted_device_moe_rounds_until_observation_ =
                        schedule.initial_round_interval;
                    rank_hosted_device_moe_observation_schedule_initialized_ =
                        true;
                    PerfStatsCollector::addCounter(
                        "moe_rebalance",
                        "rank_device_moe_rebalance_observation_schedule_initializations",
                        1.0,
                        "decode",
                        "rank",
                        {{"initial_round_interval",
                          std::to_string(schedule.initial_round_interval)},
                         {"recurring_round_interval",
                          std::to_string(schedule.recurring_round_interval)},
                         {"maximum_committed_rounds_per_boundary",
                          std::to_string(
                              schedule
                                  .maximum_committed_rounds_per_boundary)}});
                }
                else if (
                    schedule !=
                    rank_hosted_device_moe_rebalance_observation_schedule_)
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE observation schedule changed inside one request");
                    return false;
                }
                if (rank_hosted_device_moe_rounds_until_observation_ ==
                    0u)
                {
                    LOG_ERROR("[RankOrchestrator] Hosted HIP Device MoE observation countdown reached an invalid zero state");
                    return false;
                }

                const uint64_t rounds_until_observation =
                    rank_hosted_device_moe_rounds_until_observation_;
                if (committed_tokens < rounds_until_observation)
                {
                    rank_hosted_device_moe_rounds_until_observation_ -=
                        static_cast<uint32_t>(committed_tokens);
                    return submitHostScheduledDeviceMoERebalanceKnownNonDueBoundary();
                }

                rank_hosted_device_moe_rounds_until_observation_ = 0u;

                DeviceMoERebalanceDispatchTicket ticket;
                if (!observeDeviceMoERebalanceDispatchTicket(&ticket) ||
                    !submitHostScheduledDeviceMoERebalanceMaintenance(
                        ticket))
                {
                    return false;
                }
                const uint32_t next_remaining_rounds =
                    ticket.maintenance_due != 0u
                        ? schedule.recurring_round_interval
                        : ticket.decode_rounds_until_maintenance;
                rank_hosted_device_moe_rounds_until_observation_ =
                    next_remaining_rounds;
                if (rank_hosted_device_moe_rounds_until_observation_ ==
                    0u)
                {
                    LOG_ERROR("[RankOrchestrator] Authenticated HIP Device MoE ticket could not arm the next conservative observation interval");
                    return false;
                }
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_device_moe_rebalance_observation_intervals_armed",
                    1.0,
                    "decode",
                    "rank",
                    {{"maintenance_due",
                      ticket.maintenance_due != 0u ? "true" : "false"},
                     {"remaining_rounds",
                      std::to_string(next_remaining_rounds)},
                     {"rounds_until_observation",
                      std::to_string(
                          rank_hosted_device_moe_rounds_until_observation_)}});
                return true;
            }
            if (policy ==
                DeviceMoERebalanceMaintenanceExecutionPolicy::Unsupported)
            {
                LOG_ERROR("[RankOrchestrator] Device MoE maintenance participants do not share one supported scheduling policy");
                return false;
            }
        }
        if (participants.size() == 1)
        {
            return participants.front() &&
                   participants.front()->maybeApplyDecodeBoundaryMaintenance(
                       committed_tokens);
        }

        /*
         * LocalTP maintenance graphs contain NCCL/RCCL collectives and a
         * domain rendezvous.  Every participant must enter the boundary
         * concurrently; serial child calls would make the first device wait
         * for siblings that the host has not dispatched yet.
         */
        if (!tp_worker_pool_)
        {
            tp_worker_pool_ = makeParticipantWorkerPool(participants);
            if (tp_ctx_)
            {
                tp_worker_pool_->setFailureCallback(
                    [this]()
                    {
                        LOG_WARN("[TPWorkerPool] Device MoE decode-boundary maintenance failure detected - aborting collective backend");
                        tp_ctx_->requestAbort();
                    });
            }
        }
        if (tp_worker_pool_->numWorkers() != participants.size())
        {
            LOG_ERROR("[RankOrchestrator] Device MoE decode-boundary maintenance participant count does not match the persistent TP worker pool");
            return false;
        }

        const auto kernel_phase = KernelProfiler::getCurrentPhase();
        const auto rocm_phase = ROCmKernelProfiler::getCurrentPhase();
        const auto cuda_phase = CUDAKernelProfiler::getCurrentPhase();
        const auto kv_phase = KVCacheProfiler::getCurrentPhase();
        const auto executor_phase = GraphExecutorStats::currentPhase();
        tp_worker_pool_->dispatch(
            [this, use_pp_participants, committed_tokens, kernel_phase,
             rocm_phase, cuda_phase, kv_phase, executor_phase](size_t i) -> bool
            {
                KernelProfiler::setCurrentPhase(kernel_phase);
                ROCmKernelProfiler::setCurrentPhase(rocm_phase);
                CUDAKernelProfiler::setCurrentPhase(cuda_phase);
                KVCacheProfiler::setCurrentPhase(kv_phase);
                GraphExecutorStats::setCurrentPhase(executor_phase);

                auto &worker_participants =
                    use_pp_participants ? pp_stage_runners_ : device_runners_;
                if (i >= worker_participants.size() || !worker_participants[i])
                    return false;

                if (use_pp_participants && config_.pp_stages.at(i).moe_runtime &&
                    config_.pp_stages[i].moe_runtime->owners().host_maintenance)
                    return config_.pp_stages[i].moe_runtime->owners().host_maintenance->ready();

                const DeviceId device =
                    worker_participants[i]->primaryDeviceId();
                ROCmKernelProfiler::setCurrentDevice(device.ordinal);
                CUDAKernelProfiler::setCurrentDevice(device.ordinal);
                return worker_participants[i]
                    ->maybeApplyDecodeBoundaryMaintenance(committed_tokens);
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
                "maybeApplyDecodeBoundaryMaintenance",
                collect_timeout_ms,
                tp_worker_pool_->completedCount(),
                tp_worker_pool_->numWorkers());
        }
        if (first_exception)
        {
            LOG_ERROR("[RankOrchestrator] Device MoE decode-boundary maintenance re-throwing primary exception from participant "
                      << first_exception_device);
            std::rethrow_exception(first_exception);
        }
        return all_success;
    }

    void RankOrchestrator::drainCompletedDecodeBoundaryMaintenanceDiagnostics()
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->drainCompletedDecodeBoundaryMaintenanceDiagnostics();
        }
    }

    bool RankOrchestrator::admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request)
    {
        const auto &participants = !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        if (!request.valid() || participants.empty()) return false;
        for (const auto &child : participants)
            if (!child || !child->admitSparseQueryExecutionProgram(request)) return false;
        return true;
    }

    void RankOrchestrator::endSparseQueryExecutionProgram() noexcept
    {
        const auto &participants = !pp_stage_runners_.empty() ? pp_stage_runners_ : device_runners_;
        for (const auto &child : participants) if (child) child->endSparseQueryExecutionProgram();
    }

    int RankOrchestrator::get_position() const
    {
        // PP mode: return position from first PP stage runner
        // All PP stage runners should have synchronized positions
        if (!pp_stage_runners_.empty() && pp_stage_runners_[0])
        {
            return pp_stage_runners_[0]->get_position();
        }

        // TP mode: return position from primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->get_position();
        }

        return current_position_;
    }

    ExecutionPath RankOrchestrator::executionPath() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->executionPath();
        }
        return ExecutionPath::GRAPH;
    }

    const char *RankOrchestrator::architecture() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->architecture();
        }
        return "Unknown";
    }

    uint64_t RankOrchestrator::moePlacementEpoch() const
    {
        uint64_t epoch = 0;
        for (const auto &runner : device_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moePlacementEpoch());
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moePlacementEpoch());
        }
        return epoch;
    }

    uint64_t RankOrchestrator::moeRuntimeMovementEpoch() const
    {
        uint64_t epoch = 0;
        for (const auto &runner : device_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moeRuntimeMovementEpoch());
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
                epoch = std::max(epoch, runner->moeRuntimeMovementEpoch());
        }
        return epoch;
    }

    PrefixLookupResult RankOrchestrator::lookupPrefix(const std::vector<int32_t> &tokens)
    {
        PrefixLookupResult aggregate;
        last_device_prefix_hits_.clear();
        last_pp_prefix_hits_.clear();

        std::vector<PrefixParticipantLookup> participants;
        int block_size = 0;
        int participant_id = 0;

        auto query_runners = [&](
                                 std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                 std::vector<PrefixLookupResult> &hits,
                                 PrefixFingerprintCoordinationPolicy
                                     fingerprint_policy)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult hit = runner->lookupPrefix(tokens);
                LOG_DEBUG("RankOrchestrator::lookupPrefix participant " << participant_id
                          << " device=" << runner->primaryDeviceId().toString()
                          << " supported=" << hit.supported
                          << " cached_tokens=" << hit.cached_tokens
                          << " blocks=" << hit.blocks.size()
                          << " requires_terminal_logits=" << hit.requires_terminal_logits
                          << " has_terminal_logits=" << hit.has_terminal_logits
                          << " requires_terminal_hidden=" << hit.requires_terminal_hidden
                          << " has_terminal_hidden=" << hit.has_terminal_hidden);
                PrefixParticipantLookup participant = makePrefixParticipantLookup(
                    participant_id++,
                    runner->primaryDeviceId(),
                    hit,
                    {},
                    fingerprint_policy);
                participants.push_back(std::move(participant));
                hits.push_back(hit);
                if (block_size <= 0 && hit.block_size > 0)
                    block_size = hit.block_size;
            }
        };

        // Local TP participants own sharded payload slices, so their local
        // fingerprints can legitimately differ. Each child lookup has already
        // validated its own fingerprint; the rank aggregate only clamps to the
        // common restorable token count.
        query_runners(
            device_runners_,
            last_device_prefix_hits_,
            PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally);
        query_runners(
            pp_stage_runners_,
            last_pp_prefix_hits_,
            PrefixFingerprintCoordinationPolicy::ValidateParticipantLocally);

        if (participants.empty())
            return aggregate;

        aggregate = makePrefixLookupResult(coordinatePrefixLookups(std::move(participants)), block_size);
        LOG_DEBUG("RankOrchestrator::lookupPrefix aggregate supported=" << aggregate.supported
                  << " cached_tokens=" << aggregate.cached_tokens
                  << " block_size=" << aggregate.block_size
                  << " requires_terminal_logits=" << aggregate.requires_terminal_logits
                  << " has_terminal_logits=" << aggregate.has_terminal_logits
                  << " requires_terminal_hidden=" << aggregate.requires_terminal_hidden
                  << " has_terminal_hidden=" << aggregate.has_terminal_hidden);
        const int common_tokens = std::max(0, aggregate.cached_tokens);
        if (common_tokens > 0)
        {
            const auto copy_representative_blocks =
                [&](const std::vector<PrefixLookupResult> &hits)
            {
                for (const auto &hit : hits)
                {
                    if (hit.cached_tokens >= common_tokens && !hit.blocks.empty())
                    {
                        aggregate.blocks = hit.clampedTo(common_tokens).blocks;
                        return true;
                    }
                }
                return false;
            };

            if (!copy_representative_blocks(last_device_prefix_hits_))
            {
                copy_representative_blocks(last_pp_prefix_hits_);
            }
        }

        return aggregate;
    }

    PrefixRestoreMetadata RankOrchestrator::prefixRestoreMetadata(const PrefixLookupResult &hit) const
    {
        PrefixRestoreMetadata metadata;
        if (hit.cached_tokens <= 0) return metadata;
        const auto collect = [&](const auto &runners, const auto &hits)
        {
            size_t index = 0;
            for (const auto &runner : runners)
            {
                if (!runner) continue;
                if (index >= hits.size())
                    throw std::logic_error("Prefix restore metadata has no retained child admission");
                const auto selected = hits[index++].clampedTo(hit.cached_tokens);
                if (selected.cached_tokens != hit.cached_tokens)
                    throw std::logic_error("Prefix restore metadata disagrees with the common frontier");
                metadata.merge(runner->prefixRestoreMetadata(selected));
            }
            if (index != hits.size())
                throw std::logic_error("Prefix restore metadata has an unowned child admission");
        };
        collect(device_runners_, last_device_prefix_hits_);
        collect(pp_stage_runners_, last_pp_prefix_hits_);
        return metadata;
    }

    bool RankOrchestrator::preparePrefixHarvest(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        const PrefixHarvestSchedule &schedule)
    {
        if (!admission.supported || !admission.cache_enabled)
            return false;
        const auto prepare = [&](const auto &runners, auto &hits)
        {
            size_t index = 0u;
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (index >= hits.size())
                    return false;
                auto &child_admission = hits[index++];
                child_admission = child_admission.forHarvest(admission.cached_tokens);
                if (!runner->preparePrefixHarvest(child_admission, tokens, schedule))
                    return false;
            }
            return index == hits.size();
        };
        // Geometry/metadata only: start every participant writer before model
        // execution. Preserve distinct child fingerprints and payload owners.
        return prepare(device_runners_, last_device_prefix_hits_) &&
               prepare(pp_stage_runners_, last_pp_prefix_hits_);
    }

    bool RankOrchestrator::populatePrefix(const PrefixLookupResult &hit, int seq_idx)
    {
        (void)hit.restoreBlocks();
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0)
            return true;

        /*
         * Prefix population replaces every child runner's live request state.
         * Retire the aggregate before the first child crosses that boundary so
         * no observer can carry a mailbox from the previous request into the
         * restored transaction, even if child population later reports an
         * error.
         */
        invalidateRankResidentLogicalStateAggregate(
            "populate_prefix",
            "prefix_replaces_child_live_state");

        auto populate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                    const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (hit_index >= hits.size())
                    return false;

                PrefixLookupResult child_hit = hits[hit_index++].clampedTo(common_tokens);
                child_hit.restore_model_runtime_state = hit.restore_model_runtime_state;
                child_hit.restore_hybrid_state_for_suffix_prefill =
                    hit.restore_hybrid_state_for_suffix_prefill;
                if (!runner->populatePrefix(child_hit, seq_idx))
                    return false;
            }
            return hit_index == hits.size();
        };

        const bool populated =
            populate_runners(device_runners_, last_device_prefix_hits_) &&
            populate_runners(pp_stage_runners_, last_pp_prefix_hits_);
        if (!populated)
        {
            clear_cache();
            return false;
        }

        current_position_ = common_tokens;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            common_tokens);
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::harvestPrefix(
        const PrefixLookupResult &admission,
        const std::vector<int32_t> &tokens,
        int prompt_token_count)
    {
        if (!admission.cache_enabled || !admission.supported)
            return false;

        bool saw_runner = false;
        bool ok = true;
        auto harvest_runners = [&]
        (
            std::vector<std::unique_ptr<IInferenceRunner>> &runners,
            const std::vector<PrefixLookupResult> &admissions)
        {
            size_t admission_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                if (admission_index >= admissions.size())
                    return false;
                saw_runner = true;
                ok = runner->harvestPrefix(
                         admissions[admission_index++],
                         tokens,
                         prompt_token_count) &&
                     ok;
            }
            return admission_index == admissions.size();
        };

        const bool admissions_complete =
            harvest_runners(device_runners_, last_device_prefix_hits_) &&
            harvest_runners(pp_stage_runners_, last_pp_prefix_hits_);
        return saw_runner && admissions_complete && ok;
    }

    bool RankOrchestrator::restorePrefixTerminalState(const PrefixLookupResult &hit)
    {
        const int common_tokens = std::max(0, hit.cached_tokens);
        if (common_tokens <= 0 ||
            (hit.requires_terminal_logits && !hit.has_terminal_logits))
            return false;

        /*
         * A nested LocalTP domain can be one non-head PP stage. Its children
         * own KV/runtime prefix payloads but intentionally own no terminal
         * logits. Requiring logits at this composite boundary made the outer
         * pipeline fail even though its final stage had restored the sole
         * authoritative terminal row. The typed requirement bit, rather than
         * orchestration depth, decides whether absence is an error.
         */

        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const std::vector<PrefixLookupResult> &hits)
        {
            size_t hit_index = 0;
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                PrefixLookupResult child_hit =
                    hit_index < hits.size() ? hits[hit_index++].clampedTo(common_tokens)
                                            : hit.clampedTo(common_tokens);
                LOG_DEBUG("RankOrchestrator::restorePrefixTerminalState child device="
                          << runner->primaryDeviceId().toString()
                          << " cached_tokens=" << child_hit.cached_tokens
                          << " requires_terminal_logits=" << child_hit.requires_terminal_logits
                          << " has_terminal_logits=" << child_hit.has_terminal_logits);
                if (!runner->restorePrefixTerminalState(child_hit))
                {
                    LOG_DEBUG("RankOrchestrator::restorePrefixTerminalState child restore failed");
                    return false;
                }
            }
            return true;
        };

        const bool restored =
            restore_runners(device_runners_, last_device_prefix_hits_) &&
            restore_runners(pp_stage_runners_, last_pp_prefix_hits_);
        if (!restored)
            return false;

        if (device_runners_.size() > 1 &&
            logits_gatherer_ &&
            logits_gatherer_->isAllocated())
        {
            const bool has_gpu_child =
                std::any_of(
                    device_runners_.begin(),
                    device_runners_.end(),
                    [](const std::unique_ptr<IInferenceRunner> &runner)
                    {
                        return runner && runner->primaryDeviceId().is_gpu();
                    });

            if (!has_gpu_child)
            {
                /*
                 * A TP prefix hit restores terminal logits into each child
                 * shard. The rank-level CPU sampling surface is a separate
                 * full-vocabulary aggregate that may still contain the final
                 * row from the previous request. Rebuild that aggregate in the
                 * same restore transaction so no caller can observe restored
                 * KV/terminal state paired with stale logits.
                 *
                 * GPU TP deliberately does not enter this path. Its production
                 * sampler consumes device-owned child logits directly, and a
                 * restore must not manufacture a D2H host mirror merely to keep
                 * an obsolete aggregate warm.
                 */
                if (!logits_gatherer_->gather(
                        device_runners_,
                        /*seq_len=*/1,
                        vocab_size()))
                {
                    LOG_ERROR(
                        "RankOrchestrator::restorePrefixTerminalState could not "
                        "refresh CPU TP terminal logits after child restore");
                    return false;
                }
                PerfStatsCollector::addCounterWithSequence(
                    "prefix_cache",
                    "cpu_tp_terminal_logits_aggregate_refreshes",
                    1.0,
                    {static_cast<uint64_t>(common_tokens)},
                    "restore",
                    "rank",
                    {{"participants", std::to_string(device_runners_.size())},
                     {"ownership", "cpu_rank_aggregate"}});
            }
        }

        if (!skip_logits_gather_prefill_ &&
            !pp_stage_runners_.empty() && pp_stage_runners_.back())
        {
            // A full prefix hit is a prefill terminal observation. Preserve
            // the same logits ownership as ordinary prefill: CPU consumers
            // refresh their aggregate, while GPU samplers keep the restored
            // row on its existing device instead of manufacturing a D2H copy.
            if (!logits_gatherer_)
            {
                logits_gatherer_ = std::make_unique<LogitsGatherer>(
                    0,
                    0,
                    logits_backend_resolver_);
                applyLogitsGatherSkipFlags();
            }
            logits_gatherer_->copyFromStage(*pp_stage_runners_.back(),
                                            static_cast<size_t>(vocab_size()));
        }
        current_position_ = common_tokens;
        stats_dirty_ = true;
        return true;
    }

    std::vector<const IInferenceRunner *> RankOrchestrator::replicatedPLEPrefixOwners() const
    {
        if(!replicated_ple_cohort_ || !pp_stage_runners_.empty() ||
            device_runners_.size()!=static_cast<std::size_t>(replicated_ple_cohort_->degree()))
            throw std::invalid_argument("Replicated PLE rollback requires its actual complete full-main Rank child set");
        std::vector<const IInferenceRunner *> owners;owners.reserve(device_runners_.size());
        for(std::size_t p=0;p<device_runners_.size();++p) {
            const auto *runner=device_runners_[p].get();
            if(!dynamic_cast<const DeviceGraphOrchestrator *>(runner))
                throw std::invalid_argument("Replicated PLE rollback requires actual device-owned child producers");
            if (replicated_ple_cohort_->placement().isFixedHeadShards())
                (void)replicated_ple_cohort_->requireParticipant(runner->primaryDeviceId(),static_cast<int>(p));
            else replicated_ple_cohort_->requireConsumer(runner->primaryDeviceId(),static_cast<int>(p));
            owners.push_back(runner);
        }
        return owners;
    }

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixState(int seq_idx) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runners = [&](
                                   const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const char *runner_group)
        {
            size_t participant_index = 0;
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                PrefixStateSnapshot child = runner->captureLivePrefixState(seq_idx);
                if (!child.valid)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_state",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_snapshot_invalid");
                    return false;
                }

                if (!have_common_tokens)
                {
                    common_tokens = child.cached_tokens;
                    have_common_tokens = true;
                }
                else if (child.cached_tokens != common_tokens)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_state",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_token_count_mismatch",
                        common_tokens,
                        child.cached_tokens);
                    return false;
                }

                if (!have_common_provenance)
                {
                    common_provenance = child.provenance;
                    have_common_provenance = true;
                }
                else if (child.provenance != common_provenance)
                {
                    same_provenance = false;
                }

                aggregate.participant_snapshots.push_back(std::move(child));
                ++participant_index;
            }
            return true;
        };

        if (!capture_runners(device_runners_, "device") ||
            !capture_runners(pp_stage_runners_, "pp"))
        {
            return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                "capture_live_prefix_state",
                "rank",
                0,
                nullptr,
                saw_runner ? "missing_common_token_count" : "no_participants");
            return {};
        }

        aggregate.valid = true;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance ? common_provenance
                                               : PrefixStateProvenance::PayloadCheckpoint;
        if(replicated_ple_cohort_)
            aggregate.fixed_head_rollback_origin=std::shared_ptr<const Qwen4ExpReplicatedPLEPrefixOrigin>(
                new Qwen4ExpReplicatedPLEPrefixOrigin(this,replicated_ple_cohort_,replicatedPLEPrefixOwners(),aggregate,seq_idx));
        return aggregate;
    }

    PrefixStateSnapshot RankOrchestrator::captureLivePrefixCheckpoint(
        const PrefixCheckpointCaptureRequest &request) const
    {
        PrefixStateSnapshot aggregate;
        bool saw_runner = false;
        bool have_common_tokens = false;
        bool have_common_provenance = false;
        bool same_provenance = true;
        bool all_logical = true;
        int common_tokens = 0;
        PrefixStateProvenance common_provenance = PrefixStateProvenance::Unknown;

        auto capture_runners = [&](
                                   const std::vector<std::unique_ptr<IInferenceRunner>> &runners,
                                   const char *runner_group)
        {
            size_t participant_index = 0;
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                PrefixStateSnapshot child =
                    runner->captureLivePrefixCheckpoint(request);
                if (!child.valid)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_checkpoint",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_checkpoint_invalid");
                    return false;
                }

                if (!have_common_tokens)
                {
                    common_tokens = child.cached_tokens;
                    have_common_tokens = true;
                }
                else if (child.cached_tokens != common_tokens)
                {
                    rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                        "capture_live_prefix_checkpoint",
                        runner_group,
                        participant_index,
                        runner.get(),
                        "child_token_count_mismatch",
                        common_tokens,
                        child.cached_tokens);
                    return false;
                }

                all_logical = all_logical && child.logical_checkpoint;
                if (!have_common_provenance)
                {
                    common_provenance = child.provenance;
                    have_common_provenance = true;
                }
                else if (child.provenance != common_provenance)
                {
                    same_provenance = false;
                }
                aggregate.participant_snapshots.push_back(std::move(child));
                ++participant_index;
            }
            return true;
        };

        if (!capture_runners(device_runners_, "device") ||
            !capture_runners(pp_stage_runners_, "pp"))
        {
            return {};
        }
        if (!saw_runner || !have_common_tokens)
        {
            rank_orchestrator_detail::recordRankPrefixSnapshotFailure(
                "capture_live_prefix_checkpoint",
                "rank",
                0,
                nullptr,
                saw_runner ? "missing_common_token_count" : "no_participants");
            return {};
        }

        aggregate.valid = true;
        aggregate.logical_checkpoint = all_logical;
        aggregate.cached_tokens = common_tokens;
        aggregate.provenance = same_provenance
                                   ? common_provenance
                                   : (all_logical
                                          ? PrefixStateProvenance::LogicalCheckpoint
                                          : PrefixStateProvenance::PayloadCheckpoint);
        if(replicated_ple_cohort_)
            aggregate.fixed_head_rollback_origin=std::shared_ptr<const Qwen4ExpReplicatedPLEPrefixOrigin>(
                new Qwen4ExpReplicatedPLEPrefixOrigin(this,replicated_ple_cohort_,replicatedPLEPrefixOwners(),aggregate,request.sequence_index));
        return aggregate;
    }

    bool RankOrchestrator::restoreLivePrefixState(const PrefixStateSnapshot &snapshot, int seq_idx)
    {
        if (!snapshot.valid || snapshot.participant_snapshots.empty())
            return false;

        // Validate the complete rollback origin before restoring child0.
        // Portable prefix-cache imports use the existing coordinated lookup
        // path, not this process-local live checkpoint interface.
        if(replicated_ple_cohort_) {
            if(!snapshot.fixed_head_rollback_origin)return false;
            snapshot.fixed_head_rollback_origin->require(this,*replicated_ple_cohort_,
                replicatedPLEPrefixOwners(),snapshot,seq_idx);
        }

        size_t participant_index = 0;
        bool ok = true;
        bool saw_runner = false;
        auto restore_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;

                saw_runner = true;
                if (participant_index >= snapshot.participant_snapshots.size())
                {
                    ok = false;
                    continue;
                }
                ok = runner->restoreLivePrefixState(
                         snapshot.participant_snapshots[participant_index++],
                         seq_idx) &&
                     ok;
            }
        };

        restore_runners(device_runners_);
        restore_runners(pp_stage_runners_);
        if (!(saw_runner && ok && participant_index == snapshot.participant_snapshots.size()))
            return false;

        /*
         * A prefix restore is atomic at the rank boundary as well as at each
         * child runner.  The children own the actual KV/GDN payloads, but rank
         * orchestration still owns public position/sequence bookkeeping used by
         * diagnostics, PP mode, and any code path that cannot delegate directly
         * to a single child.  Keep that aggregate state in lockstep with the
         * restored child snapshots so repeated restore/replay cycles cannot
         * observe stale parent metadata.
         */
        current_position_ = snapshot.cached_tokens;
        current_padded_seq_len_ = 0;
        current_sequence_lengths_.assign(
            static_cast<size_t>(std::max(1, config_.batch_size)),
            snapshot.cached_tokens);
        stats_dirty_ = true;
        return true;
    }

    bool RankOrchestrator::truncateLivePrefixState(int cached_tokens, int seq_idx)
    {
        bool saw_runner = false;
        bool ok = true;
        auto truncate_runners = [&](std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (auto &runner : runners)
            {
                if (!runner)
                    continue;
                saw_runner = true;
                ok = runner->truncateLivePrefixState(cached_tokens, seq_idx) && ok;
            }
        };

        truncate_runners(device_runners_);
        truncate_runners(pp_stage_runners_);
        return saw_runner && ok;
    }

    PrefixRuntimeStateSnapshot RankOrchestrator::prefixStateProbe(
        const PrefixProbeCapturePolicy &capture_policy) const
    {
        PrefixRuntimeStateSnapshot snapshot;
        snapshot.initialized = true;
        snapshot.architecture = architecture();
        snapshot.execution_path = "rank-orchestrator";
        snapshot.current_position = get_position();
        snapshot.positions = {snapshot.current_position};
        snapshot.sequence_lengths = current_sequence_lengths_;

        const bool use_authoritative_child_logical_state =
            pp_stage_runners_.empty() && !device_runners_.empty();
        // Packed scalar observations retain the existing logical-state owner.
        // Child native publication generations are independent; only their
        // authenticated logical values form the symmetric LocalTP consensus.
        const auto condition_owner = capture_policy.capture_device_logical_state &&
                use_authoritative_child_logical_state
            ? deviceResidentLogicalSequenceState()
            : DeviceResidentLogicalSequenceStateHandle{};
        const bool project_condition = condition_owner.valid() &&
            condition_owner.request_count == 1;
        std::optional<PrefixResidentConditionProbe> mirrored_condition;
        std::optional<int> authoritative_child_position;
        std::vector<int> authoritative_child_positions;
        std::vector<int> authoritative_child_sequence_lengths;
        std::optional<bool> authoritative_verifier_identity_present;
        auto adopt_authoritative_child_logical_state =
            [&](const PrefixRuntimeStateSnapshot &child)
        {
            /*
             * A child probe is already a host-visible observation boundary. On
             * GPU it materializes either the live logical-state mailbox or the
             * canonical cache-owned sequence count after mailbox retirement.
             * Requiring a currently live mailbox again at rank scope would
             * discard that proven child truth immediately after restore and
             * revive the rank's scheduler-only host cursor.
             *
             * Pipeline stages may legitimately own different positions, so
             * only symmetric non-PP participants participate in this equality
             * contract. Empty mock/uninitialized probes carry no authority.
             */
            if (!use_authoritative_child_logical_state ||
                !child.initialized)
                return;
            if (!authoritative_child_position.has_value())
            {
                authoritative_child_position = child.current_position;
                authoritative_child_positions = child.positions;
                authoritative_child_sequence_lengths =
                    child.sequence_lengths;
                return;
            }
            if (*authoritative_child_position != child.current_position ||
                authoritative_child_positions != child.positions ||
                authoritative_child_sequence_lengths != child.sequence_lengths)
            {
                throw std::runtime_error(
                    "Rank prefix-state diagnostics observed divergent "
                    "canonical logical metadata across LocalTP participants");
            }
        };

        auto merge_child =
            [&snapshot,
             &authoritative_verifier_identity_present,
             use_authoritative_child_logical_state](
                const PrefixRuntimeStateSnapshot &child)
        {
            constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
            constexpr uint64_t kFnvPrime = 1099511628211ull;
            auto fold_terminal_value = [](uint64_t digest, uint64_t value)
            {
                for (unsigned byte = 0; byte < sizeof(value); ++byte)
                {
                    digest ^= value & 0xffull;
                    digest *= kFnvPrime;
                    value >>= 8;
                }
                return digest;
            };
            auto merge_terminal_hash =
                [&](bool child_available,
                    size_t child_bytes,
                    uint64_t child_hash,
                    bool *aggregate_available,
                    size_t *aggregate_bytes,
                    uint64_t *aggregate_hash)
            {
                if (!child_available)
                    return;

                uint64_t digest =
                    *aggregate_available
                        ? *aggregate_hash
                        : kFnvOffsetBasis;
                digest = fold_terminal_value(
                    digest,
                    static_cast<uint64_t>(child_bytes));
                digest = fold_terminal_value(digest, child_hash);
                *aggregate_available = true;
                *aggregate_bytes += child_bytes;
                *aggregate_hash = digest;
            };

            snapshot.initialized = snapshot.initialized && child.initialized;
            snapshot.has_hidden = snapshot.has_hidden || child.has_hidden;
            snapshot.has_logits = snapshot.has_logits || child.has_logits;
            snapshot.session_epoch = std::max(snapshot.session_epoch, child.session_epoch);
            snapshot.prefix_cache_config_enabled =
                snapshot.prefix_cache_config_enabled || child.prefix_cache_config_enabled;
            snapshot.prefix_cache_ready = snapshot.prefix_cache_ready || child.prefix_cache_ready;
            snapshot.prefix_cache_bypassed =
                snapshot.prefix_cache_bypassed || child.prefix_cache_bypassed;
            if (snapshot.prefix_cache_bypass_reason.empty() &&
                !child.prefix_cache_bypass_reason.empty())
            {
                snapshot.prefix_cache_bypass_reason = child.prefix_cache_bypass_reason;
            }
            snapshot.prefix_cache_lookups += child.prefix_cache_lookups;
            snapshot.prefix_cache_hits += child.prefix_cache_hits;
            snapshot.prefix_cache_partial_hits += child.prefix_cache_partial_hits;
            snapshot.prefix_cache_misses += child.prefix_cache_misses;
            snapshot.prefix_cache_matched_blocks += child.prefix_cache_matched_blocks;
            snapshot.prefix_cache_matched_tokens += child.prefix_cache_matched_tokens;
            snapshot.prefix_cache_stores += child.prefix_cache_stores;
            snapshot.prefix_cache_inserts += child.prefix_cache_inserts;
            snapshot.prefix_cache_evictions += child.prefix_cache_evictions;
            snapshot.prefix_cache_promotions += child.prefix_cache_promotions;
            snapshot.prefix_cache_ram_to_disk_demotions +=
                child.prefix_cache_ram_to_disk_demotions;
            snapshot.prefix_cache_device_hot_promotions +=
                child.prefix_cache_device_hot_promotions;
            snapshot.prefix_cache_device_hot_repromotions +=
                child.prefix_cache_device_hot_repromotions;
            snapshot.prefix_cache_device_hot_evictions +=
                child.prefix_cache_device_hot_evictions;
            snapshot.prefix_cache_disk_evictions +=
                child.prefix_cache_disk_evictions;
            snapshot.prefix_cache_device_hot_direct_hits +=
                child.prefix_cache_device_hot_direct_hits;
            snapshot.prefix_cache_disk_hydrations += child.prefix_cache_disk_hydrations;
            snapshot.prefix_cache_terminal_state_hits += child.prefix_cache_terminal_state_hits;
            snapshot.prefix_cache_ram_bytes += child.prefix_cache_ram_bytes;
            snapshot.prefix_cache_device_bytes += child.prefix_cache_device_bytes;
            snapshot.prefix_cache_disk_bytes += child.prefix_cache_disk_bytes;
            snapshot.prefix_cache_hybrid_state_bytes += child.prefix_cache_hybrid_state_bytes;
            snapshot.prefix_cache_mtp_state_bytes += child.prefix_cache_mtp_state_bytes;
            snapshot.prefix_cache_bypasses += child.prefix_cache_bypasses;
            snapshot.prefix_cache_unsupported_backend_bypasses +=
                child.prefix_cache_unsupported_backend_bypasses;
            snapshot.prefix_cache_fingerprint_bypasses +=
                child.prefix_cache_fingerprint_bypasses;
            snapshot.prefix_cache_terminal_state_bypasses +=
                child.prefix_cache_terminal_state_bypasses;
            snapshot.mtp_config_enabled = snapshot.mtp_config_enabled || child.mtp_config_enabled;
            snapshot.mtp_bypassed = snapshot.mtp_bypassed || child.mtp_bypassed;
            if (snapshot.mtp_bypass_reason.empty() && !child.mtp_bypass_reason.empty())
            {
                snapshot.mtp_bypass_reason = child.mtp_bypass_reason;
            }
            snapshot.mtp_draft_steps += child.mtp_draft_steps;
            snapshot.mtp_accepted_tokens += child.mtp_accepted_tokens;
            snapshot.mtp_rejected_tokens += child.mtp_rejected_tokens;
            snapshot.mtp_rollbacks += child.mtp_rollbacks;
            snapshot.mtp_bypasses += child.mtp_bypasses;
            snapshot.mtp_verifier_runs += child.mtp_verifier_runs;
            snapshot.mtp_verifier_token_count += child.mtp_verifier_token_count;
            snapshot.mtp_last_transaction_draft_depth =
                snapshot.mtp_last_transaction_draft_depth == 0
                    ? child.mtp_last_transaction_draft_depth
                    : std::min(
                          snapshot.mtp_last_transaction_draft_depth,
                          child.mtp_last_transaction_draft_depth);
            snapshot.mtp_last_transaction_emitted_token_count =
                snapshot.mtp_last_transaction_emitted_token_count == 0
                    ? child.mtp_last_transaction_emitted_token_count
                    : std::min(
                          snapshot.mtp_last_transaction_emitted_token_count,
                          child.mtp_last_transaction_emitted_token_count);
            const bool child_has_verifier_identity =
                child.mtp_observed_verifier_transaction_count > 0 ||
                child.mtp_observed_verifier_draft_depth > 0 ||
                child.mtp_observed_verifier_condition_token.has_value() ||
                !child.mtp_observed_verifier_draft_tokens.empty();
            if (!authoritative_verifier_identity_present.has_value())
            {
                authoritative_verifier_identity_present =
                    child_has_verifier_identity;
            }
            else if (use_authoritative_child_logical_state &&
                     *authoritative_verifier_identity_present !=
                         child_has_verifier_identity)
            {
                throw std::runtime_error(
                    "Rank prefix-state diagnostics observed committed MTP "
                    "verifier identity on only a subset of mirrored "
                    "participants");
            }
            if (child_has_verifier_identity)
            {
                if (child.mtp_observed_verifier_transaction_count <= 0 ||
                    child.mtp_observed_verifier_draft_depth <= 0 ||
                    !child.mtp_observed_verifier_condition_token ||
                    *child.mtp_observed_verifier_condition_token < 0 ||
                    child.mtp_observed_verifier_draft_tokens.size() !=
                        static_cast<size_t>(
                            child.mtp_observed_verifier_draft_depth))
                {
                    throw std::runtime_error(
                        "Rank prefix-state diagnostics observed an incomplete "
                        "committed MTP verifier identity");
                }
                if (snapshot.mtp_observed_verifier_draft_tokens.empty())
                {
                    snapshot.mtp_observed_verifier_transaction_count =
                        child.mtp_observed_verifier_transaction_count;
                    snapshot.mtp_observed_verifier_draft_depth =
                        child.mtp_observed_verifier_draft_depth;
                    snapshot.mtp_observed_verifier_condition_token =
                        child.mtp_observed_verifier_condition_token;
                    snapshot.mtp_observed_verifier_draft_tokens =
                        child.mtp_observed_verifier_draft_tokens;
                }
                else if (
                    snapshot.mtp_observed_verifier_transaction_count !=
                        child.mtp_observed_verifier_transaction_count ||
                    snapshot.mtp_observed_verifier_draft_depth !=
                        child.mtp_observed_verifier_draft_depth ||
                    snapshot.mtp_observed_verifier_condition_token !=
                        child.mtp_observed_verifier_condition_token ||
                    snapshot.mtp_observed_verifier_draft_tokens !=
                        child.mtp_observed_verifier_draft_tokens)
                {
                    auto format_tokens = [](const std::vector<int32_t> &tokens)
                    {
                        std::ostringstream out;
                        out << '[';
                        for (size_t index = 0; index < tokens.size(); ++index)
                        {
                            if (index != 0)
                                out << ',';
                            out << tokens[index];
                        }
                        out << ']';
                        return out.str();
                    };
                    throw std::runtime_error(
                        "Rank prefix-state diagnostics observed divergent "
                        "committed MTP verifier draft rows across mirrored "
                        "participants: authoritative=" +
                        format_tokens(
                            snapshot.mtp_observed_verifier_draft_tokens) +
                        " divergent=" +
                        format_tokens(
                            child.mtp_observed_verifier_draft_tokens));
                }
            }
            snapshot.mtp_depth_policy_windows += child.mtp_depth_policy_windows;
            snapshot.mtp_depth_policy_updates += child.mtp_depth_policy_updates;
            snapshot.mtp_depth_policy_promotions += child.mtp_depth_policy_promotions;
            snapshot.mtp_depth_policy_demotions += child.mtp_depth_policy_demotions;
            snapshot.mtp_depth_policy_utility_returns += child.mtp_depth_policy_utility_returns;
            snapshot.mtp_depth_policy_utility_keeps += child.mtp_depth_policy_utility_keeps;
            snapshot.mtp_depth_policy_utility_probes += child.mtp_depth_policy_utility_probes;
            snapshot.mtp_depth_policy_observe_recommendations +=
                child.mtp_depth_policy_observe_recommendations;
            snapshot.mtp_current_depth =
                snapshot.mtp_current_depth == 0
                    ? child.mtp_current_depth
                    : std::min(snapshot.mtp_current_depth, child.mtp_current_depth);
            snapshot.mtp_min_depth =
                snapshot.mtp_min_depth == 0
                    ? child.mtp_min_depth
                    : std::min(snapshot.mtp_min_depth, child.mtp_min_depth);
            snapshot.mtp_max_depth =
                snapshot.mtp_max_depth == 0
                    ? child.mtp_max_depth
                    : std::min(snapshot.mtp_max_depth, child.mtp_max_depth);
            snapshot.prefill_chunk_schedules += child.prefill_chunk_schedules;
            snapshot.prefill_chunk_successful_schedules +=
                child.prefill_chunk_successful_schedules;
            snapshot.prefill_chunks += child.prefill_chunks;
            snapshot.prefill_chunk_real_tokens += child.prefill_chunk_real_tokens;
            snapshot.prefill_chunk_padded_tokens += child.prefill_chunk_padded_tokens;
            snapshot.prefill_chunk_failures += child.prefill_chunk_failures;
            /*
             * LocalTP participants own distinct device buffers even when the
             * model policy replicates their logical rows.  Fold every child in
             * stable participant order rather than selecting participant zero;
             * a one-device restore defect must change the rank-level digest.
             */
            merge_terminal_hash(
                child.terminal_hidden_hash_available,
                child.terminal_hidden_bytes,
                child.terminal_hidden_hash,
                &snapshot.terminal_hidden_hash_available,
                &snapshot.terminal_hidden_bytes,
                &snapshot.terminal_hidden_hash);
            snapshot.terminal_hidden_values.insert(
                snapshot.terminal_hidden_values.end(),
                child.terminal_hidden_values.begin(),
                child.terminal_hidden_values.end());
            merge_terminal_hash(
                child.terminal_logits_hash_available,
                child.terminal_logits_bytes,
                child.terminal_logits_hash,
                &snapshot.terminal_logits_hash_available,
                &snapshot.terminal_logits_bytes,
                &snapshot.terminal_logits_hash);
            snapshot.terminal_logits_values.insert(
                snapshot.terminal_logits_values.end(),
                child.terminal_logits_values.begin(),
                child.terminal_logits_values.end());
            if (snapshot.primary_device.is_cpu() && !child.primary_device.is_cpu())
            {
                snapshot.primary_device = child.primary_device;
            }
            if (snapshot.positions.empty() && !child.positions.empty())
            {
                snapshot.positions = child.positions;
            }
            if (snapshot.sequence_lengths.empty() && !child.sequence_lengths.empty())
            {
                snapshot.sequence_lengths = child.sequence_lengths;
            }
            snapshot.kv_caches.insert(snapshot.kv_caches.end(), child.kv_caches.begin(), child.kv_caches.end());
            snapshot.mtp_kv_caches.insert(snapshot.mtp_kv_caches.end(),
                                          child.mtp_kv_caches.begin(),
                                          child.mtp_kv_caches.end());
            snapshot.gdn_layers.insert(snapshot.gdn_layers.end(), child.gdn_layers.begin(), child.gdn_layers.end());
            snapshot.prefill_graphs.insert(snapshot.prefill_graphs.end(),
                                           child.prefill_graphs.begin(),
                                           child.prefill_graphs.end());
        };

        bool saw_child = false;
        uint64_t tp_movement_epoch = 0;
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                const PrefixRuntimeStateSnapshot child =
                    runner->prefixStateProbe(capture_policy);
                if (project_condition)
                {
                    const auto original = runner->deviceResidentLogicalSequenceState();
                    if (!child.initialized || !child.resident_condition || !original.valid() ||
                        original.request_count != 1 ||
                        child.resident_condition->device != original.device ||
                        child.resident_condition->publication_generation != original.publication_generation ||
                        child.resident_condition->live_state_epoch != original.live_state_epoch ||
                        child.resident_condition->target_position != child.current_position)
                        throw std::runtime_error("Rank prefix-state diagnostics lost an original participant condition publication");
                    const auto& observed = *child.resident_condition;
                    if (mirrored_condition &&
                        (mirrored_condition->target_position != observed.target_position ||
                         mirrored_condition->next_condition_token != observed.next_condition_token ||
                         mirrored_condition->publication_ok != observed.publication_ok ||
                         mirrored_condition->stopped != observed.stopped))
                        throw std::runtime_error("Rank prefix-state diagnostics observed divergent logical condition values across LocalTP participants");
                    mirrored_condition = observed;
                }
                adopt_authoritative_child_logical_state(child);
                tp_movement_epoch = std::max(tp_movement_epoch, child.moe_runtime_movement_epoch.epoch());
                merge_child(child);
                saw_child = true;
            }
        }
        if (project_condition)
        {
            if (!mirrored_condition ||
                !deviceResidentLogicalSequenceState().sameMailboxAs(condition_owner))
                throw std::runtime_error("Rank prefix-state diagnostics rejected a replaced logical condition owner");
            // This is the same already-authenticated aggregate ownership token,
            // not a choice of one GPU's unrelated event generation. The outer
            // runner alone authenticates ReadyMTPCondition response disposition.
            snapshot.resident_condition = *mirrored_condition;
            snapshot.resident_condition->device = condition_owner.device;
            snapshot.resident_condition->publication_generation = condition_owner.publication_generation;
            snapshot.resident_condition->live_state_epoch = condition_owner.live_state_epoch;
            snapshot.resident_condition->response_disposition.reset();
        }
        snapshot.moe_runtime_movement_epoch = PrefixMovementEpochSnapshot::leaf(tp_movement_epoch);
        std::vector<PrefixMovementEpochSnapshot> pipeline_epochs;
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                const auto child = runner->prefixStateProbe(capture_policy);
                pipeline_epochs.push_back(child.moe_runtime_movement_epoch);
                merge_child(child);
                if (runner.get() == pp_stage_runners_.back().get())
                {
                    // Pipeline stages can legitimately be at different rows.
                    // The original terminal stage owns the completed model's
                    // logical position, including reset/restore. Project its
                    // already-acquired observation, never the head's scheduler
                    // cursor and never a maximum across unrelated stages.
                    if (!child.initialized && capture_policy.capture_device_logical_state)
                        throw std::runtime_error("Pipeline prefix diagnostics lost its original terminal logical-state observation");
                    if (child.initialized)
                    {
                        authoritative_child_position = child.current_position;
                        authoritative_child_positions = child.positions;
                        authoritative_child_sequence_lengths = child.sequence_lengths;
                    }
                }
                saw_child = true;
            }
            else
                throw std::logic_error("Pipeline prefix diagnostics lost a declared stage");
        }
        if (!pipeline_epochs.empty())
            snapshot.moe_runtime_movement_epoch = PrefixMovementEpochSnapshot::pipeline(std::move(pipeline_epochs));

        if (!saw_child)
        {
            snapshot.initialized = false;
        }
        if (authoritative_child_position.has_value())
        {
            /*
             * The rank host cursor is only a scheduler/response shadow. Child
             * probes own canonical sequence truth across both active mailbox
             * and post-restore cache-count lifecycles. TP requires consensus;
             * PP projects its original terminal stage. Rank diagnostics must
             * never substitute the shadow merely because an outcome mailbox
             * was retired.
             */
            snapshot.current_position = *authoritative_child_position;
            snapshot.positions =
                std::move(authoritative_child_positions);
            snapshot.sequence_lengths =
                std::move(authoritative_child_sequence_lengths);
        }
        if (snapshot.prefix_cache_bypassed)
        {
            snapshot.prefix_cache_ready = false;
        }

        return snapshot;
    }

    // =========================================================================
    // Hidden State API (for Pipeline Parallelism nesting)
    // =========================================================================

    TensorBase *RankOrchestrator::getHiddenState()
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    const TensorBase *RankOrchestrator::getHiddenState() const
    {
        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, all devices have same hidden state after allreduce
            // Delegate to primary device runner
            if (!device_runners_.empty() && device_runners_[0])
            {
                return device_runners_[0]->getHiddenState();
            }
        }
        else
        {
            // PP or TP_PP mode - last stage has the final hidden state
            if (!pp_stage_runners_.empty() && pp_stage_runners_.back())
            {
                return pp_stage_runners_.back()->getHiddenState();
            }
        }
        return nullptr;
    }

    void RankOrchestrator::setHiddenState(TensorBase *hidden_state)
    {
        hidden_state_input_ = hidden_state;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, set on ALL device runners (they all need the same input)
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->setHiddenState(hidden_state);
                }
            }
        }
        else
        {
            // PP or TP_PP mode - set on first stage runner (stage 0 receives external input)
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->setHiddenState(hidden_state);
            }
        }
    }

    bool RankOrchestrator::hasHiddenStateInput() const
    {
        return hidden_state_input_ != nullptr;
    }

    void RankOrchestrator::clearHiddenStateInput()
    {
        hidden_state_input_ = nullptr;

        if (mode_ == ParallelismMode::TP)
        {
            // In TP mode, clear on all device runners
            for (auto &runner : device_runners_)
            {
                if (runner)
                {
                    runner->clearHiddenStateInput();
                }
            }
        }
        else
        {
            // PP or TP_PP mode - clear on first stage runner
            if (!pp_stage_runners_.empty() && pp_stage_runners_.front())
            {
                pp_stage_runners_.front()->clearHiddenStateInput();
            }
        }
    }

    // =========================================================================
    // Snapshot API
    // =========================================================================

    void RankOrchestrator::enableSnapshotCapture(const std::string &output_dir)
    {
        LOG_DEBUG("RankOrchestrator::enableSnapshotCapture: Enabling on all devices");

        // Enable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }

        // Enable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->enableSnapshotCapture(output_dir);
            }
        }
    }

    void RankOrchestrator::setSnapshotCaptureFilter(const std::vector<std::string> &keys)
    {
        for (auto &runner : device_runners_)
        {
            if (runner)
                runner->setSnapshotCaptureFilter(keys);
        }

        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
                runner->setSnapshotCaptureFilter(keys);
        }
    }

    void RankOrchestrator::disableSnapshotCapture()
    {
        LOG_DEBUG("RankOrchestrator::disableSnapshotCapture: Disabling on all devices");

        // Disable on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }

        // Disable on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->disableSnapshotCapture();
            }
        }
    }

    void RankOrchestrator::clearSnapshots()
    {
        LOG_DEBUG("RankOrchestrator::clearSnapshots: Clearing on all devices");

        // Clear on TP device runners
        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }

        // Clear on PP stage runners
        for (auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                runner->clearSnapshots();
            }
        }
    }

    bool RankOrchestrator::phaseSplitReplicatedDecodeSnapshotsActive() const
    {
        // The started transaction owns the phase. In particular, an explicit
        // one-token prefill still uses sharded prefill weights and snapshots.
        if (last_logits_forward_phase_ != LogitsForwardPhase::Decode)
            return false;
        const int max_decode_like_rows = config_.mtp.enabled
                                             ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
                                             : 1;
        if (current_padded_seq_len_ <= 0 ||
            current_padded_seq_len_ > max_decode_like_rows)
        {
            return false;
        }

        const auto &plan = config_.moe_routed_expert_plan;
        if (!plan || !plan->usesExpertOverlayAuthority())
            return false;

        const auto &dense = plan->continuation_domain_spec;
        return dense.dense_tp_enabled && dense.dense_decode_replicated;
    }

    /** @copydoc RankOrchestrator::phaseSplitReplicatedDecodeEmbeddingSnapshotsActive */
    bool RankOrchestrator::phaseSplitReplicatedDecodeEmbeddingSnapshotsActive() const
    {
        // A full-vocabulary decode table is independently looked up on every
        // participant. It has no embedding allreduce whose CompleteValue
        // publication could supersede the schema's vocabulary-shard partials.
        // The declared embedding placement and actual transaction phase own
        // this distinction; a one-row tensor alone is never sufficient.
        if (last_logits_forward_phase_ != LogitsForwardPhase::Decode)
            return false;
        const int max_decode_like_rows = config_.mtp.enabled
                                             ? std::max(1, resolveMTPMaxTargetQueryRows(config_.mtp))
                                             : 1;
        if (current_padded_seq_len_ <= 0 ||
            current_padded_seq_len_ > max_decode_like_rows)
            return false;
        const auto &plan = config_.moe_routed_expert_plan;
        if (!plan || !plan->usesExpertOverlayAuthority())
            return false;
        const auto &dense = plan->continuation_domain_spec;
        return dense.dense_tp_enabled &&
               (dense.dense_decode_replicated ||
                dense.dense_decode_mirrored_embedding);
    }

    bool RankOrchestrator::isPhaseSplitReplicatedDecodeSnapshotKey(const std::string &key)
    {
        const std::string stage_type = extractStageType(key);

        return stage_type == "ATTENTION_OUTPUT" ||
               stage_type == "FFN_DOWN" ||
               stage_type == "GDN_OUTPUT" ||
               stage_type == "MOE_SHARED_EXPERT_OUTPUT" ||
               stage_type == "MOE_SHARED_GATE_OUTPUT";
    }

    bool RankOrchestrator::isPhaseSplitReplicatedAttentionSnapshotKey(const std::string &key)
    {
        // These exact semantics belong to the complete dense attention block.
        // Vocabulary and routed-expert publications retain independent owners.
        static constexpr std::array<std::string_view, 19> kAttentionStages = {
            "Q_PROJECTION", "K_PROJECTION", "V_PROJECTION", "Q_NORM", "K_NORM",
            "Q_ROPE", "K_ROPE", "FA_GATE", "ATTENTION_CONTEXT", "ATTENTION_CONTEXT_GATED",
            "QKV_PROJECTION", "GDN_Z_PROJECTION", "GDN_ALPHA", "GDN_BETA",
            "GDN_CONV1D_OUTPUT", "GDN_RECURRENCE", "GDN_DELTA_RULE_OUTPUT",
            "GATED_RMSNORM", "GDN_NORM_GATE_OUTPUT",
        };
        const std::string stage_type = extractStageType(key);
        return std::find(kAttentionStages.begin(), kAttentionStages.end(), stage_type) !=
               kAttentionStages.end();
    }

    bool RankOrchestrator::replicatedMTPSidecarSnapshotsActive() const noexcept
    {
        return tp_ctx_ &&
               tp_ctx_->degree() > 1 &&
               retainsMTPGraphCapacity(config_.mtp) &&
               config_.mtp.sidecar_dense_policy ==
                   MTPSidecarDensePolicy::ReplicatedPerParticipant;
    }

    bool RankOrchestrator::mtpSnapshotRetainsIndependentSharding(
        const std::string &stage_type)
    {
        /*
         * Replicating the learned predictor does not replicate its
         * vocabulary table/head, nor does it replace ExpertOverlay's routed
         * expert publication protocol.  Those authorities keep the ordinary
         * schema layout while all other TP-shaped predictor intermediates are
         * complete per-participant tensors.
         */
        static constexpr std::array<std::string_view, 5>
            kIndependentlyOwnedMTPStages = {
                "EMBEDDING",
                "LM_HEAD",
                "MOE_EXPERT_OUTPUT",
                "MOE_ROUTE_CONTRIBUTIONS",
                "MOE_SHARED_RANK_BANK_PUBLISH",
            };
        return std::find(
                   kIndependentlyOwnedMTPStages.begin(),
                   kIndependentlyOwnedMTPStages.end(),
                   stage_type) != kIndependentlyOwnedMTPStages.end();
    }

    SnapshotShardingMode RankOrchestrator::resolveSnapshotShardingMode(const std::string &key) const
    {
        /*
         * Snapshot scopes describe transaction timing, not tensor layout.
         * Resolve the underlying semantic stage through the ordinary model
         * schema so column- and row-parallel LocalTP intermediates are combined
         * exactly as their unscoped counterparts.  Only the terminal LM head
         * is special: MTP explicitly mirrors that full-vocabulary projection on
         * every participant, even while the transformer body remains TP.
         */
        static constexpr std::array<std::string_view, 2> kConditionScopes = {
            "MTP_REQUEST_BATCH_CONDITION_",
            "MTP_SCALAR_CONDITION_",
        };
        std::string_view condition_scope;
        for (const std::string_view candidate : kConditionScopes)
        {
            if (key.rfind(candidate, 0) == 0)
            {
                condition_scope = candidate;
                break;
            }
        }
        const bool condition_snapshot = !condition_scope.empty();
        const size_t chunk_stage = prefillChunkSnapshotSemanticStageOffset(key);
        const bool prefill_chunk_snapshot = chunk_stage != std::string_view::npos;
        const std::string_view semantic_key_view =
            condition_snapshot
                ? std::string_view(key).substr(condition_scope.size())
                : prefill_chunk_snapshot ? std::string_view(key).substr(chunk_stage)
                                         : std::string_view(key);
        const std::string semantic_key(semantic_key_view);
        const bool mtp_model_snapshot =
            isMTPDepthQualifiedSnapshot(semantic_key);
        if (extractStageType(semantic_key) == "LM_HEAD" &&
            (condition_snapshot ||
             (mtp_model_snapshot && publishesFullVocabularyMTPLogits())))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        SnapshotShardingMode mode =
            getStageShardingMode(semantic_key, stage_sharding_map_);

        /*
         * Static schemas describe the main transformer's TP layout.  A
         * replicated MTP predictor deliberately reuses the same semantic stage
         * names while binding complete dense/shared weights and omitting the
         * row-parallel collectives.  Reclassify only depth-qualified model
         * checkpoints whose schema layout denotes a TP partial.  Independent
         * vocabulary and ExpertOverlay publications retain their own authority
         * above; UNKNOWN/GATHERED/ROOT_ONLY modes are never guessed here.
         */
        const std::string stage_type = extractStageType(semantic_key);
        const bool schema_mode_is_tp_partial =
            mode == SnapshotShardingMode::COLUMN_PARALLEL ||
            mode == SnapshotShardingMode::PACKED_COLUMN_PARALLEL ||
            mode == SnapshotShardingMode::ROW_PARALLEL;
        if (mtp_model_snapshot &&
            replicatedMTPSidecarSnapshotsActive() &&
            schema_mode_is_tp_partial &&
            !mtpSnapshotRetainsIndependentSharding(stage_type))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        // Only the ordinary main embedding uses this placement rule. MTP
        // depths and condition scopes retain their own vocabulary authority;
        // generic layer-qualified names are not global embedding checkpoints.
        if (!condition_snapshot && !mtp_model_snapshot && !prefill_chunk_snapshot &&
            semantic_key == "EMBEDDING" &&
            mode == SnapshotShardingMode::ROW_PARALLEL &&
            phaseSplitReplicatedDecodeEmbeddingSnapshotsActive())
        {
            return SnapshotShardingMode::REPLICATED;
        }

        /*
         * Phase-split MoE overlay uses DenseTP for prefill but switches dense and
         * always-on shared decode work to full replicated weights. The schema still
         * marks dense output projections as ROW_PARALLEL because that is correct
         * for plain TP. During replicated decode, summing per-device snapshots
         * doubles the captured tensor and produces false parity failures.
         */
        // A retained prefill chunk predates the current main transaction.
        // Later replicated decode must not reinterpret its original TP values.
        // Explicit immutable producer publications still override schema layout
        // in getTPSnapshot, including complete heads and token-row partitions.
        if (!prefill_chunk_snapshot && mode == SnapshotShardingMode::ROW_PARALLEL &&
            phaseSplitReplicatedDecodeSnapshotsActive() &&
            isPhaseSplitReplicatedDecodeSnapshotKey(semantic_key))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        // Ordinary main decode binds the complete FA/GDN dense block on each
        // participant. It therefore publishes full column/packed intermediates,
        // just as the existing row-output rule above publishes full Wo output.
        // The typed phase and declared placement establish replication; the
        // one-row restriction excludes grouped verifier transactions. Their
        // depth/condition scopes keep the existing independent MTP contracts.
        if (!condition_snapshot && !mtp_model_snapshot && !prefill_chunk_snapshot &&
            current_padded_seq_len_ == 1 &&
            (mode == SnapshotShardingMode::COLUMN_PARALLEL ||
             mode == SnapshotShardingMode::PACKED_COLUMN_PARALLEL) &&
            phaseSplitReplicatedDecodeSnapshotsActive() &&
            isPhaseSplitReplicatedAttentionSnapshotKey(semantic_key))
        {
            return SnapshotShardingMode::REPLICATED;
        }

        return mode;
    }

    const float *RankOrchestrator::getSnapshot(const std::string &key, size_t &out_size) const
    {
        // For GATHERED stages (e.g., LM_HEAD) with multi-device TP, return the gathered combined_logits.
        // This is necessary because each device only has logits_local with vocab_local entries,
        // but tests expect the full vocab_size logits.
        //
        // CRITICAL: For hybrid PP+TP, only the stage that actually HAS the LM head should
        // return combined_logits. Otherwise, a nested TP stage (like PP stage 0) that has
        // logits_local buffers allocated but never computes LM_HEAD would return stale data.
        if (getStageShardingMode(key, stage_sharding_map_) == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool owns_lm_head = true;
            if (config_.nested_pp_stage_config.has_value())
            {
                owns_lm_head = config_.nested_pp_stage_config->has_lm_head;
            }
            else
            {
                auto wm = model_ctx_->weightManager();
                owns_lm_head = wm && wm->hasLMHead();
            }

            if (!owns_lm_head)
            {
                LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD: this stage doesn't own LM_HEAD, "
                          << "falling through to PP stage search"
                          << " (nested_pp_config=" << config_.nested_pp_stage_config.has_value()
                          << " has_lm_head=" << (config_.nested_pp_stage_config.has_value() ? config_.nested_pp_stage_config->has_lm_head : false) << ")");
            }
            else
            {
                bool has_column_parallel_lm_head = false;
                for (const auto &runner : device_runners_)
                {
                    if (runner && runner->hasLogitsLocal())
                    {
                        has_column_parallel_lm_head = true;
                        break;
                    }
                }

                if (has_column_parallel_lm_head)
                {
                    out_size = logits_gatherer_->lastGatheredSize();
                    const float *ptr = logits_gatherer_->data();
                    LOG_DEBUG("RankOrchestrator::getSnapshot LM_HEAD returning combined_logits with "
                              << out_size << " elements (column-parallel gathering), ptr=" << (void *)ptr
                              << " first_element=" << (ptr ? ptr[0] : -999999.0f));
                    return ptr;
                }
            }
        }

        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    const float *result = runner->getSnapshot(key, out_size);
                    if (result != nullptr)
                    {
                        return result;
                    }
                }
            }
            out_size = 0;
            return nullptr;
        }

        // Default (TP mode): prefer the primary device, then fall back to any
        // participant that actually published the key. Some graph diagnostics
        // are intentionally participant-local; getSnapshotKeys() merges all
        // participants, so returning only primary would silently drop those keys.
        if (!device_runners_.empty() && device_runners_[0])
        {
            const float *primary = device_runners_[0]->getSnapshot(key, out_size);
            if (primary)
                return primary;
        }
        for (size_t i = 1; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;
            const float *participant = device_runners_[i]->getSnapshot(key, out_size);
            if (participant)
                return participant;
        }
        out_size = 0;
        return nullptr;
    }

    std::vector<std::string> RankOrchestrator::getSnapshotKeys() const
    {
        // Merge keys from all devices (use set to deduplicate)
        std::set<std::string> all_keys;

        // Collect from TP device runners
        for (const auto &runner : device_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        // Collect from PP stage runners
        for (const auto &runner : pp_stage_runners_)
        {
            if (runner)
            {
                auto keys = runner->getSnapshotKeys();
                all_keys.insert(keys.begin(), keys.end());
            }
        }

        return std::vector<std::string>(all_keys.begin(), all_keys.end());
    }

    /**
     * @brief Retain typed diagnostic bytes and authenticate every authored-row participant.
     * @param key Exact canonical checkpoint identity.
     * @return Immutable owned publication, including a valid empty observation, or absence.
     * @throws std::logic_error When authored interval provenance is missing or disagrees.
     * @details This diagnostic lookup reads no live execution metadata. CompleteValue
     * can supersede a partition only after every declared interval participant agrees.
     */
    SnapshotInfo RankOrchestrator::getSnapshotWithShape(const std::string &key) const
    {
        // PP mode: search across all PP stage runners
        if (!pp_stage_runners_.empty())
        {
            for (const auto &runner : pp_stage_runners_)
            {
                if (runner)
                {
                    auto snap = runner->getSnapshotWithShape(key);
                    if (snap)
                        return snap;
                }
            }
            return {};
        }

        // A completed publication is authoritative even when its owner is not
        // participant zero. Keep an ordinary shard only when no finalizer has
        // published this semantic value; getTPSnapshot owns shard assembly.
        SnapshotInfo first_partition,complete_value;
        std::vector<SnapshotInfo> replicas;
        replicas.reserve(device_runners_.size());
        bool authored=false;
        for (const auto &runner:device_runners_)
        {
            replicas.push_back(runner?runner->getSnapshotWithShape(key):SnapshotInfo{});
            const auto &participant=replicas.back();
            authored|=participant.semantic_axis==SnapshotSemanticAxis::AuthoredTokenRows ||
                participant.authored_token_rows.has_value();
            if(participant && participant.publication==SnapshotPublication::CompleteValue && !complete_value)
                complete_value=participant;
            if(participant && !first_partition)first_partition=participant;
        }
        // Every declared participant must authenticate the same original row
        // provenance before a complete publication can supersede partitions.
        if(authored)validateSnapshotAuthoredTokenRowReplicas(replicas);
        if(complete_value)return complete_value;
        if (first_partition && first_partition.isInteger())
        {
            // Integer identities are replicated declarations, never numerical
            // partial sums. Every participant must publish the exact same bits.
            for (const auto &runner : device_runners_)
            {
                if (!runner) continue;
                const auto replica = runner->getSnapshotWithShape(key);
                if (!replica || !replica.isInteger() || replica.rows != first_partition.rows ||
                    replica.cols != first_partition.cols || replica.size != first_partition.size ||
                    !std::equal(replica.integerValues().begin(), replica.integerValues().end(),
                                first_partition.integerValues().begin()))
                    throw std::logic_error("integer snapshot replicas disagree or are incomplete: " + key);
            }
        }
        return first_partition;
    }

    TPSnapshot RankOrchestrator::getTPSnapshot(const std::string &key) const
    {
        TPSnapshot result;
        result.key = key;
        result.mode = resolveSnapshotShardingMode(key);
        result.tp_degree = static_cast<int>(device_runners_.size());

        LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                          << " mode=" << shardingModeToString(result.mode)
                                                          << " tp_degree=" << result.tp_degree);

        // =========================================================================
        // PP Mode: Delegate to appropriate PP stage runner
        // For PP+TP hybrid, each PP stage runner may be an MDO for TP
        // =========================================================================
        if (!pp_stage_runners_.empty())
        {
            // Find which PP stage owns this snapshot key
            for (size_t stage_idx = 0; stage_idx < pp_stage_runners_.size(); ++stage_idx)
            {
                if (!pp_stage_runners_[stage_idx])
                    continue;

                // Check if this stage has the snapshot (with shape metadata)
                auto snap = pp_stage_runners_[stage_idx]->getSnapshotWithShape(key);
                if (!snap)
                    continue;
                if (snap.isInteger())
                    throw std::logic_error("integer snapshot requires typed getSnapshotWithShape: " + key);

                // Found the owning stage - check if it's a TP domain (RankOrchestrator)
                auto *inner_mdo = dynamic_cast<const RankOrchestrator *>(
                    pp_stage_runners_[stage_idx].get());

                if (inner_mdo)
                {
                    // This PP stage is a TP domain - delegate to its getTPSnapshot
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is TP domain, delegating getTPSnapshot for key=" << key);
                    return inner_mdo->getTPSnapshot(key);
                }
                else
                {
                    // Single-device PP stage - wrap the snapshot as non-TP
                    LOG_DEBUG("RankOrchestrator::getTPSnapshot: PP stage " << stage_idx
                                                                           << " is single device for key=" << key);

                    result.tp_degree = 1;
                    result.mode = SnapshotShardingMode::REPLICATED;

                    DeviceSnapshotData dev_data;
                    dev_data.device_index = 0;
                    dev_data.rows = snap.rows;
                    dev_data.cols = snap.cols;
                    dev_data.global_start_col = 0;
                    dev_data.global_total_cols = snap.cols;
                    dev_data.data.assign(snap.data, snap.data + snap.size);

                    result.device_data.push_back(std::move(dev_data));
                    result.combined_valid = true;
                    result.combined_data = result.device_data[0].data;
                    result.combined_rows = snap.rows;
                    result.combined_cols = snap.cols;

                    return result;
                }
            }

            // Key not found in any PP stage
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: key=" << key
                                                              << " not found in any PP stage");
            return result;
        }

        // =========================================================================
        // TP Mode (non-PP): Collect from device_runners_
        // =========================================================================

        // A typed subset consumer cannot use the legacy KV<TP full-replica override.
        // Keep actual producer geometry alongside retained publications; no live cache is queried.
        const bool kv_snapshot = isKVHeadSnapshotStage(extractStageType(std::string(kvHeadSnapshotProducerKey(key))));
        std::vector<KVHeadSnapshotProducer> kv_producers;
        // Acquire each immutable publication once. A root finalizer can share
        // a semantic key with pre-finalization partials left on other devices.
        // Complete values supersede those partials; when several finalizers
        // publish, REPLICATED assembly still verifies their agreement.
        std::vector<SnapshotInfo> publications(device_runners_.size());
        bool has_complete_publication = false;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (device_runners_[i])
                publications[i] = device_runners_[i]->getSnapshotWithShape(key);
            has_complete_publication |=
                publications[i] && publications[i].publication ==
                                       SnapshotPublication::CompleteValue;
        }
        if (has_complete_publication)
            result.mode = SnapshotShardingMode::REPLICATED;
        else if (std::any_of(publications.begin(), publications.end(), [](const auto &snap) {
                     return snap && snap.publication == SnapshotPublication::RowPartition; }))
        {
            for (const auto &snap : publications)
            {
                if (!snap || snap.publication != SnapshotPublication::RowPartition)
                    throw std::runtime_error("Snapshot '" + key + "' has missing or mixed explicit row partitions");
            }
            // This is host diagnostic assembly, not an inference collective.
            // Each row is copied once. Adding zero-filled unowned rows would
            // needlessly change signed-zero/NaN bits and hide missing owners.
            result.mode = SnapshotShardingMode::TOKEN_ROW_PARTITION;
        }
        else if (std::any_of(publications.begin(), publications.end(),
                     [](const SnapshotInfo &snap) {
                         return snap && snap.publication == SnapshotPublication::ColumnPartition;
                     }))
        {
            // A column producer owns a contiguous slice in participant order.
            // Dropping a missing participant would silently shrink/reindex the
            // tensor; mixing schema partial sums with columns would corrupt it.
            // Complete finalizers above retain their existing supersession rule.
            if (!std::all_of(publications.begin(), publications.end(),
                    [](const SnapshotInfo &snap) {
                        return snap && snap.publication == SnapshotPublication::ColumnPartition;
                    }))
                throw std::runtime_error("Snapshot '" + key +
                    "' has missing or mixed explicit output-column partitions");
            result.mode = SnapshotShardingMode::COLUMN_PARALLEL;
        }

        const bool retained_prefill_chunk =
            prefillChunkSnapshotSemanticStageOffset(key) != std::string_view::npos;
        if (kv_snapshot && !has_complete_publication)
        {
            // CompleteValue supersedes partial source metadata above. Only an
            // actual subset assembly needs original absolute-layer ownership.
            for (const auto &runner : device_runners_)
            {
                const auto *owner = dynamic_cast<const DeviceGraphOrchestrator *>(runner.get());
                const auto *builder = owner ? owner->graphBuilder() : nullptr;
                const auto *cfg = builder ? &builder->config() : nullptr;
                const auto geometry = cfg && hasQueryIntervalKVDeclaration(*cfg)
                    ? resolveQueryKVSnapshotLayerGeometry(*cfg, key) : KVHeadSnapshotLayerGeometry{};
                kv_producers.push_back({cfg, runner ? runner->primaryDeviceId() : DeviceId::cpu(), 0,
                    geometry.columns_per_head, geometry.original_kv_heads});
            }
            for (size_t i = 0; i < publications.size(); ++i)
                kv_producers[i].row_columns = publications[i] ? publications[i].cols : 0;
            const bool mtp_depth = isMTPDepthQualifiedSnapshot(key);
            const auto demand = (mtp_depth && replicatedMTPSidecarSnapshotsActive()) ||
                (!mtp_depth && !retained_prefill_chunk && phaseSplitReplicatedDecodeSnapshotsActive())
                ? KVHeadConsumerDemand::FullModel : KVHeadConsumerDemand::AssignedQueryInterval;
            const auto groups = resolveQueryIntervalKVSnapshotGroups(kv_producers, demand);
            if (groups)
            {
                for (const auto &publication : publications)
                    if (!publication || publication.publication != SnapshotPublication::SchemaPartition)
                        throw std::invalid_argument("query KV snapshot has missing or conflicting publication ownership");
                result.mode = SnapshotShardingMode::PACKED_COLUMN_PARALLEL;
                result.column_groups = *groups;
            }
        }

        if (retained_prefill_chunk &&
            extractStageType(key) == "LM_HEAD" &&
            result.mode == SnapshotShardingMode::GATHERED)
        {
            // A retained chunk belongs to its producing transaction. The current
            // logits gatherer can already contain a later decode, and GATHERED
            // assembly alone would incorrectly accept the first local shard.
            // Explicit complete/column/row publications were resolved above.
            // Only declared mirrored heads establish full values without an
            // explicit publication; verify all replicas through normal assembly.
            if (!mtpTerminalHeadIsMirrored(config_.mtp.terminal_head_policy))
                throw std::runtime_error("Snapshot '" + key +
                    "' lacks retained terminal-head publication ownership");
            if (!std::all_of(publications.begin(), publications.end(),
                    [](const SnapshotInfo &snap) { return static_cast<bool>(snap); }))
                throw std::runtime_error("Snapshot '" + key +
                    "' has missing retained mirrored terminal-head publications");
            result.mode = SnapshotShardingMode::REPLICATED;
        }

        // The live gatherer is valid only for an unretained current transaction.
        if (!retained_prefill_chunk &&
            result.mode == SnapshotShardingMode::GATHERED &&
            device_runners_.size() > 1 && logits_gatherer_ && logits_gatherer_->isAllocated() && tp_ctx_)
        {
            bool has_column_parallel_lm_head = false;
            for (const auto &runner : device_runners_)
            {
                if (runner && runner->hasLogitsLocal())
                {
                    has_column_parallel_lm_head = true;
                    break;
                }
            }

            if (has_column_parallel_lm_head)
            {
                size_t gathered_size = logits_gatherer_->lastGatheredSize();
                const float *gathered_ptr = logits_gatherer_->data();
                DeviceSnapshotData gathered;
                gathered.device_id = GlobalDeviceId::gpu(0, 0, config_.devices[0].device_type);
                gathered.device_index = 0;
                gathered.rows = 1;
                gathered.cols = gathered_size;
                gathered.global_start_col = 0;
                gathered.global_total_cols = gathered_size;
                gathered.data.assign(gathered_ptr, gathered_ptr + gathered_size);

                result.device_data.push_back(std::move(gathered));
                result.combined_valid = true;
                result.combined_data = result.device_data[0].data;
                result.combined_rows = result.device_data[0].rows;
                result.combined_cols = result.device_data[0].cols;

                LOG_DEBUG("RankOrchestrator::getTPSnapshot: LM_HEAD using combined_logits "
                          << "size=" << gathered_size);
                return result;
            }
        }

        // Collect only the publications belonging to the selected assembly.
        size_t global_col_offset = 0;
        for (size_t i = 0; i < device_runners_.size(); ++i)
        {
            if (!device_runners_[i])
                continue;

            // Use shape-aware snapshot retrieval — the stage itself reported
            // its output rows/cols via getDumpInfo() at capture time, so we
            // don't need model-specific dimension calculations here.
            const auto &snap = publications[i];

            if (has_complete_publication && snap.publication !=
                                                SnapshotPublication::CompleteValue)
                continue;

            if (!snap)
            {
                LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                     << " has no data for key=" << key);
                continue;
            }

            if (snap.isInteger())
                throw std::logic_error("integer snapshot requires typed getSnapshotWithShape: " + key);
            // Debug: log data pointer and first 4 values to verify each device has different data
            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " key=" << key << " ptr=" << static_cast<const void *>(snap.data)
                                                                 << " size=" << snap.size
                                                                 << " shape=[" << snap.rows << "x" << snap.cols << "]"
                                                                 << " val[0-3]=" << snap.data[0] << "," << snap.data[1] << "," << snap.data[2] << "," << snap.data[3]);

            DeviceSnapshotData dev_data;
            // Use device type from config if available, otherwise default to CUDA
            DeviceType dev_type = DeviceType::CUDA;
            if (i < config_.devices.size())
            {
                dev_type = config_.devices[i].device_type;
            }
            dev_data.device_id = GlobalDeviceId::gpu(0, static_cast<int>(i), dev_type);
            if (kv_snapshot && !result.column_groups.empty())
            {
                const auto actual = kv_producers[i].device;
                dev_data.device_id = actual.is_cpu() ? GlobalDeviceId::cpu(0)
                    : GlobalDeviceId::gpu(0, actual.ordinal, actual.type);
            }
            dev_data.device_index = static_cast<int>(i);
            dev_data.data.assign(snap.data, snap.data + snap.size);
            dev_data.row_ownership = snap.row_ownership;

            // Use shape metadata from the stage's getDumpInfo() output.
            // This is model-agnostic: stages report their own dimensions
            // (e.g., KV projections report kv_dim cols, FFN reports d_ff cols).
            if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = global_col_offset;
                global_col_offset += snap.cols;
            }
            else
            {
                // Replicated or row-parallel - each device has full output
                dev_data.rows = snap.rows;
                dev_data.cols = snap.cols;
                dev_data.global_start_col = 0;
                dev_data.global_total_cols = snap.cols;
            }

            LOG_DEBUG("RankOrchestrator::getTPSnapshot: device " << i
                                                                 << " size=" << snap.size
                                                                 << " cols=" << dev_data.cols
                                                                 << " start_col=" << dev_data.global_start_col);

            result.device_data.push_back(std::move(dev_data));
        }

        // Set global_total_cols for column-parallel stages
        if (result.mode == SnapshotShardingMode::COLUMN_PARALLEL && !result.device_data.empty())
        {
            size_t total_cols = global_col_offset;
            for (auto &dev : result.device_data)
            {
                dev.global_total_cols = total_cols;
            }
        }
        else if (result.mode == SnapshotShardingMode::PACKED_COLUMN_PARALLEL &&
                 !result.device_data.empty() && result.column_groups.empty())
        {
            /*
             * Qwen GDN participants publish dependency-closed local rows.
             * Resolve every semantic repeat from the model's authoritative
             * head geometry and each producing stage's real shape; no retired
             * replicated-Q/K representation is accepted.
             */
            std::string layout_error;
            if (!model_ctx_ || !model_ctx_->loader())
            {
                layout_error = "model loader is unavailable";
            }
            else
            {
                const auto loader = model_ctx_->loader();
                const std::string metadata_prefix =
                    model_ctx_->architecture() + ".ssm.";
                const int key_heads =
                    loader->getInt(metadata_prefix + "group_count", 0);
                const int value_heads =
                    loader->getInt(metadata_prefix + "time_step_rank", 0);
                const int state_width =
                    loader->getInt(metadata_prefix + "state_size", 0);
                result.column_groups = resolvePackedGDNColumnGroups(
                    result.device_data,
                    result.tp_degree,
                    extractStageType(key),
                    key_heads,
                    value_heads,
                    state_width,
                    &layout_error);
            }

            if (result.column_groups.empty())
            {
                LOG_ERROR("RankOrchestrator::getTPSnapshot: packed checkpoint '"
                          << key << "' has no valid GDN TP layout: "
                          << layout_error);
            }
            else
            {
                const size_t total_cols = std::accumulate(
                    result.column_groups.begin(),
                    result.column_groups.end(),
                    size_t{0},
                    [](size_t sum, const SnapshotColumnGroup &group)
                    {
                        return sum + group.global_cols;
                    });
                for (auto &device : result.device_data)
                {
                    device.global_start_col = 0;
                    device.global_total_cols = total_cols;
                }
            }
        }

        if (kv_snapshot && !result.column_groups.empty())
        {
            size_t total = 0;
            for (const auto &group : result.column_groups) total += group.global_cols;
            for (auto &producer : result.device_data) producer.global_total_cols = total;
        }
        return result;
    }

    std::string RankOrchestrator::describeFailedMirroredMTPState() const
    {
        if (!DebugEnv::isTruthyEnv(
                "LLAMINAR_MTP_GRAPH_REUSE_DIAGNOSTICS"))
        {
            return {};
        }

        std::vector<std::vector<MTPMirroredTensorDigest>>
            participant_digests;
        participant_digests.reserve(device_runners_.size());
        for (const auto &participant : device_runners_)
        {
            auto *device_orchestrator =
                dynamic_cast<DeviceGraphOrchestrator *>(participant.get());
            participant_digests.push_back(
                device_orchestrator
                    ? device_orchestrator
                          ->captureMirroredMTPDigests()
                    : std::vector<MTPMirroredTensorDigest>{});
        }
        return describeMirroredDigestMismatch(participant_digests);
    }

    std::vector<std::pair<std::string, SnapshotShardingMode>>
    RankOrchestrator::getSnapshotKeysWithSharding() const
    {
        auto keys = getSnapshotKeys();
        std::vector<std::pair<std::string, SnapshotShardingMode>> result;
        result.reserve(keys.size());

        for (const auto &key : keys)
        {
            result.emplace_back(key, resolveSnapshotShardingMode(key));
        }

        return result;
    }

    // =========================================================================
    // Profiling API
    // =========================================================================

    const GraphExecutorStats *RankOrchestrator::executorStats() const
    {
        aggregateStats();
        return aggregated_stats_.get();
    }

    void RankOrchestrator::resetExecutorStats()
    {
        LOG_DEBUG("RankOrchestrator::resetExecutorStats: Resetting on all devices");

        for (auto &runner : device_runners_)
        {
            if (runner)
            {
                runner->resetExecutorStats();
            }
        }

        if (aggregated_stats_)
        {
            aggregated_stats_->reset();
        }
        tp_decode_stats_.reset();
        stats_dirty_ = true;
    }

    // =========================================================================
    // Orchestration API
    // =========================================================================

    bool RankOrchestrator::hasPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->hasPlacementPlan();
        }
        return false;
    }

    const PlacementPlan &RankOrchestrator::getPlacementPlan() const
    {
        // Delegate to primary device
        if (!device_runners_.empty() && device_runners_[0])
        {
            return device_runners_[0]->getPlacementPlan();
        }
        throw std::runtime_error("No placement plan available: no device runners");
    }

    // =========================================================================
    // IRankOrchestrator Interface Implementation
    // =========================================================================

    int RankOrchestrator::device_count() const
    {
        return static_cast<int>(device_runners_.size());
    }

    IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx)
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    const IInferenceRunner *RankOrchestrator::deviceRunner(int device_idx) const
    {
        if (device_idx < 0 || device_idx >= static_cast<int>(device_runners_.size()))
        {
            throw std::out_of_range("Device index " + std::to_string(device_idx) +
                                    " out of range [0, " + std::to_string(device_runners_.size()) + ")");
        }
        return device_runners_[device_idx].get();
    }

    ILocalTPContext *RankOrchestrator::localTPContext()
    {
        return tp_ctx_.get();
    }

    const ILocalTPContext *RankOrchestrator::localTPContext() const
    {
        return tp_ctx_.get();
    }

    bool RankOrchestrator::allDevicesReady() const
    {
        if (device_runners_.empty())
        {
            return false;
        }

        for (const auto &runner : device_runners_)
        {
            if (!runner)
            {
                return false;
            }
            // Check if runner is ready (has vocab_size > 0 indicates initialization)
            if (runner->vocab_size() <= 0)
            {
                return false;
            }
        }

        return true;
    }

    void RankOrchestrator::synchronizeDevices()
    {
        LOG_DEBUG("RankOrchestrator::synchronizeDevices: Synchronizing all devices");

        if (tp_ctx_)
        {
            tp_ctx_->synchronize();
        }

        /*
         * createForTest() may inject runners whose placement metadata names a
         * CUDA or ROCm device.  Those identifiers are policy fixtures, not live
         * hardware ownership.  The injected LocalTP context above is host-only,
         * and the unit-test boundary must end before BackendManager lookup so
         * teardown cannot create a physical GPU context as a side effect.
         */
        if (!external_device_backend_access_enabled_)
            return;

        auto synchronize_runner_device =
            [](const std::unique_ptr<IInferenceRunner> &runner)
        {
            if (!runner)
                return;
            const DeviceId device = runner->primaryDeviceId();
            if (!device.is_gpu())
                return;

            IBackend *backend = getBackendFor(device);
            if (!backend)
            {
                LOG_WARN("RankOrchestrator::synchronizeDevices: Backend unavailable for "
                         << device.toString());
                return;
            }
            if (!backend->synchronize(device.gpu_ordinal()))
            {
                LOG_WARN("RankOrchestrator::synchronizeDevices: Device synchronization failed for "
                         << device.toString());
            }
        };

        for (const auto &runner : device_runners_)
        {
            synchronize_runner_device(runner);
        }
        for (const auto &runner : pp_stage_runners_)
        {
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
            {
                rank->synchronizeDevices();
                continue;
            }
            synchronize_runner_device(runner);
        }
    }

    IBackend *RankOrchestrator::resolveLogitsBackend(DeviceId device) const
    {
        if (logits_backend_resolver_)
            return logits_backend_resolver_(device);
        return getBackendFor(device);
    }

    MoERebalanceController *RankOrchestrator::moeRebalanceController() const
    {
        auto controllers = moeRebalanceControllers();
        return selectActiveMoERebalanceController(controllers);
    }

    std::vector<MoERebalanceController *> RankOrchestrator::moeRebalanceControllers() const
    {
        std::vector<MoERebalanceController *> controllers;
        auto append_from = [&](const std::vector<std::unique_ptr<IInferenceRunner>> &runners)
        {
            for (const auto &runner : runners)
            {
                if (!runner)
                    continue;
                for (auto *controller : runner->moeRebalanceControllers())
                {
                    if (controller &&
                        std::find(controllers.begin(), controllers.end(), controller) == controllers.end())
                    {
                        controllers.push_back(controller);
                    }
                }
            }
        };

        append_from(device_runners_);
        append_from(pp_stage_runners_);
        return controllers;
    }

    MoERebalanceController *RankOrchestrator::moeRebalanceControllerForDomain(
        const std::string &domain_id) const
    {
        for (auto *controller : moeRebalanceControllers())
        {
            if (controller && controller->domainId() == domain_id)
                return controller;
        }
        return nullptr;
    }

    void RankOrchestrator::requireMoEPipelineRuntimeOwnership(
        MoEPipelineOwnershipValidation frontier) const
    {
        if (frontier != MoEPipelineOwnershipValidation::OwnersBound &&
            frontier != MoEPipelineOwnershipValidation::ServingMaterialized)
            throw std::invalid_argument("MoE pipeline ownership validation has an unknown setup frontier");
        if (pp_stage_runners_.empty() || !device_runners_.empty() ||
            pp_stage_runners_.size() != config_.pp_stages.size())
            throw std::logic_error("MoE pipeline lost its exact participant-local stage owners");
        for (std::size_t index = 0; index < pp_stage_runners_.size(); ++index)
        {
            const auto &child = pp_stage_runners_[index];
            const auto &stage = config_.pp_stages[index];
            const auto &binding = stage.moe_runtime;
            if (!child || !binding)
                throw std::logic_error("MoE pipeline has a missing stage runtime owner");
            // Reuse the complete construction-time scope, including the union
            // of parent participant types that owns exact hosted graph capacity.
            const auto projected = config_.forPipelineStage(index);
            binding->requireDestination(projected.nested_pp_stage_config.value(),
                stage.stage_devices, stage.tp_weights, stage.tp_backend, config_.moe_rebalance.mode);
            binding->requireRuntimeOwner(*child, frontier);
        }
        if (frontier == MoEPipelineOwnershipValidation::ServingMaterialized &&
            !validatePPGraphTransactionReady("MoE stage ownership certification"))
            throw std::logic_error("MoE pipeline serving segments are not completely materialized");
    }

    MoEOverlayAuthorityExecutionKind
    RankOrchestrator::moeOverlayAuthorityExecution() const
    {
        const auto &runners = pp_stage_runners_.empty()
                                  ? device_runners_
                                  : pp_stage_runners_;
        if (runners.empty())
            return MoEOverlayAuthorityExecutionKind::Unresolved;

        MoEOverlayAuthorityExecutionKind selected =
            MoEOverlayAuthorityExecutionKind::Unresolved;
        for (const auto &runner : runners)
        {
            if (!runner)
            {
                throw std::logic_error(
                    "ExpertOverlay authority query reached a null rank participant");
            }
            const auto participant =
                runner->moeOverlayAuthorityExecution();
            if (selected ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
            {
                selected = participant;
                continue;
            }
            if (participant != selected)
            {
                throw std::logic_error(
                    "RankOrchestrator participants disagree on the frozen ExpertOverlay authority execution backend");
            }
        }
        return selected;
    }

    bool RankOrchestrator::
        deviceResidentMoEOverlayMaintenanceReady() const
    {
        const auto &runners = pp_stage_runners_.empty()
                                  ? device_runners_
                                  : pp_stage_runners_;
        if (runners.empty() ||
            moeOverlayAuthorityExecution() !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident)
        {
            return false;
        }
        return std::all_of(
            runners.begin(),
            runners.end(),
            [](const std::unique_ptr<IInferenceRunner> &runner)
            {
                return runner &&
                       runner
                           ->deviceResidentMoEOverlayMaintenanceReady();
            });
    }

    bool RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const MoELayeredExpertOwnership *previous_ownership)
    {
        auto snapshot = snapshotMoEExpertMasksForAllDevices(
            controller,
            replica_arrivals,
            previous_ownership);
        return applyMoEExpertMasksForAllDevices(
            snapshot.masks_by_participant,
            snapshot.domain_id,
            snapshot.transferMasks());
    }

    RankOrchestrator::MoEExpertMaskSnapshot RankOrchestrator::snapshotMoEExpertMasksForAllDevices(
        const MoERebalanceController &controller,
        const ExpertReplicaSet *replica_arrivals,
        const MoELayeredExpertOwnership *previous_ownership) const
    {
        MoEExpertMaskSnapshot snapshot;
        snapshot.domain_id = controller.domainId();

        const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
        if (gpu_cache_experts > 0)
        {
            snapshot.masks_by_participant = controller.computeGpuCacheExpertMasks(gpu_cache_experts);
            return snapshot;
        }

        snapshot.masks_by_participant.reserve(device_runners_.size());
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
            snapshot.masks_by_participant.push_back(controller.computeExpertMasksForParticipant(static_cast<int>(device_idx)));

        if (replica_arrivals)
        {
            snapshot.transfer_masks_by_participant =
                rank_orchestrator_detail::buildReplicaArrivalTransferMasks(
                    controller,
                    *replica_arrivals);
        }
        else if (previous_ownership)
        {
            snapshot.transfer_masks_by_participant =
                rank_orchestrator_detail::buildOwnershipArrivalTransferMasks(
                    controller,
                    *previous_ownership);
        }

        return snapshot;
    }

    RankOrchestrator::PreparedMoEExpertMaskUpdate RankOrchestrator::prepareMoEExpertMaskTransfersForAllDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
        const std::string &domain_id,
        const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant)
    {
        using Clock = std::chrono::steady_clock;
        int direct_transfer_pair_batches = 0;
        int serialized_transfer_pair_batches = 0;
        size_t transfer_mask_entries = 0;
        const std::string stats_device = primaryDeviceId().is_valid()
                                             ? primaryDeviceId().toString()
                                             : std::string{};
        const PerfStatsCollector::Tags phase_tags{{"domain_id", domain_id}};

        std::vector<DeviceGraphOrchestrator *> local_dgos;
        local_dgos.reserve(device_runners_.size());
        for (auto &runner : device_runners_)
            local_dgos.push_back(dynamic_cast<DeviceGraphOrchestrator *>(runner.get()));
        std::vector<bool> gpu_direct_prepare_ok(local_dgos.size(), true);
        for (auto *dgo : local_dgos)
        {
            if (dgo)
                dgo->clearPendingGpuDirectExpertTransfers();
        }

        auto count_mask_entries = [](const std::vector<std::vector<bool>> &masks) -> size_t
        {
            size_t count = 0;
            for (const auto &layer_mask : masks)
                count += static_cast<size_t>(std::count(layer_mask.begin(), layer_mask.end(), true));
            return count;
        };

        auto collect_local_transfers = [&](size_t destination_idx,
                                           const std::vector<std::vector<bool>> &target_masks,
                                           const std::vector<std::vector<bool>> &transfer_masks)
        {
            ReceivedWeightsMap merged;
            (void)target_masks;
            auto direct_remaining_masks = transfer_masks;
            auto serialized_remaining_masks = transfer_masks;
            bool direct_only_gpu_transfer_attempted = false;
            for (size_t source_idx = 0; source_idx < local_dgos.size(); ++source_idx)
            {
                if (source_idx == destination_idx || !local_dgos[source_idx])
                    continue;

                auto is_same_backend_gpu_pair = [&]()
                {
                    if (!local_dgos[destination_idx] || !local_dgos[source_idx])
                        return false;
                    const DeviceId dst_device = local_dgos[destination_idx]->primaryDeviceId();
                    const DeviceId src_device = local_dgos[source_idx]->primaryDeviceId();
                    return rank_orchestrator_detail::sameBackendGpuExpertTransferIsDirectOnly(
                        dst_device,
                        src_device);
                };
                const bool same_backend_gpu_pair = is_same_backend_gpu_pair();
                auto &remaining_masks =
                    same_backend_gpu_pair ? direct_remaining_masks : serialized_remaining_masks;
                const size_t requested_entries = count_mask_entries(remaining_masks);
                if (requested_entries == 0)
                    continue;

                if (same_backend_gpu_pair)
                {
                    ++direct_transfer_pair_batches;
                }
                else
                {
                    ++serialized_transfer_pair_batches;
                }
                transfer_mask_entries += requested_entries;

                if (same_backend_gpu_pair)
                {
                    direct_only_gpu_transfer_attempted = true;
                    if (local_dgos[destination_idx])
                    {
                        auto prepared_direct =
                            local_dgos[destination_idx]->prepareExpertWeightsDirectForMasksFrom(
                                *local_dgos[source_idx],
                                remaining_masks);
                        remaining_masks = std::move(prepared_direct.remaining_masks);
                    }
                    continue;
                }

                auto source_blobs = local_dgos[source_idx]->collectExpertWeightsForMasks(remaining_masks);
                for (auto &layer_entry : source_blobs)
                {
                    auto &dst_layer = merged[layer_entry.first];
                    for (auto &expert_entry : layer_entry.second)
                    {
                        if (dst_layer.find(expert_entry.first) == dst_layer.end())
                        {
                            dst_layer.emplace(expert_entry.first, std::move(expert_entry.second));
                            if (layer_entry.first >= 0 &&
                                layer_entry.first < static_cast<int>(remaining_masks.size()) &&
                                expert_entry.first >= 0 &&
                                expert_entry.first < static_cast<int>(remaining_masks[static_cast<size_t>(layer_entry.first)].size()))
                            {
                                remaining_masks[static_cast<size_t>(layer_entry.first)][static_cast<size_t>(expert_entry.first)] = false;
                            }
                        }
                    }
                }
            }
            if (direct_only_gpu_transfer_attempted &&
                count_mask_entries(direct_remaining_masks) > 0 &&
                destination_idx < gpu_direct_prepare_ok.size())
            {
                gpu_direct_prepare_ok[destination_idx] = false;
                LOG_ERROR("[RankOrchestrator] Same-backend GPU expert transfer prepare left "
                          << count_mask_entries(direct_remaining_masks)
                          << " arrival mask entries unsatisfied for participant "
                          << destination_idx << " domain=" << domain_id);
            }
            return merged;
        };

        std::vector<ReceivedWeightsMap> received_by_device(device_runners_.size());
        const auto transfer_start = Clock::now();
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx < masks_by_participant.size() && local_dgos[device_idx])
            {
                const auto &transfer_masks =
                    (transfer_masks_by_participant &&
                     device_idx < transfer_masks_by_participant->size())
                        ? (*transfer_masks_by_participant)[device_idx]
                        : masks_by_participant[device_idx];
                received_by_device[device_idx] =
                    collect_local_transfers(
                        device_idx,
                        masks_by_participant[device_idx],
                        transfer_masks);
            }
        }
        const auto transfer_end = Clock::now();

        PerfStatsCollector::recordTimingNs(
            "moe_rebalance",
            "rank_local_transfer_prepare",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      transfer_end - transfer_start)
                                      .count()),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_transfer_mask_entries",
            static_cast<double>(transfer_mask_entries),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_direct_transfer_pair_batches",
            static_cast<double>(direct_transfer_pair_batches),
            "rebalance",
            stats_device,
            phase_tags);
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_local_serialized_transfer_pair_batches",
            static_cast<double>(serialized_transfer_pair_batches),
            "rebalance",
            stats_device,
            phase_tags);

        PreparedMoEExpertMaskUpdate prepared;
        prepared.domain_id = domain_id;
        prepared.masks_by_participant = masks_by_participant;
        prepared.received_by_device = std::move(received_by_device);
        prepared.gpu_direct_prepare_ok_by_device = std::move(gpu_direct_prepare_ok);
        return prepared;
    }

    bool RankOrchestrator::publishPreparedMoEExpertMasksForAllDevices(
        const PreparedMoEExpertMaskUpdate &prepared)
    {
        using Clock = std::chrono::steady_clock;
        int applied = 0;
        const std::string stats_device = primaryDeviceId().is_valid()
                                             ? primaryDeviceId().toString()
                                             : std::string{};
        const PerfStatsCollector::Tags phase_tags{{"domain_id", prepared.domain_id}};

        std::vector<DeviceGraphOrchestrator *> local_dgos;
        local_dgos.reserve(device_runners_.size());
        for (auto &runner : device_runners_)
            local_dgos.push_back(dynamic_cast<DeviceGraphOrchestrator *>(runner.get()));

        const auto publish_start = Clock::now();
        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;
            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;
            if (device_idx < prepared.gpu_direct_prepare_ok_by_device.size() &&
                !prepared.gpu_direct_prepare_ok_by_device[device_idx])
            {
                LOG_ERROR("[RankOrchestrator] Aborting MoE expert mask publish for domain="
                          << prepared.domain_id
                          << " because same-backend GPU-direct staging was incomplete for participant "
                          << device_idx);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_local_mask_publish_aborts",
                    1.0,
                    "rebalance",
                    stats_device,
                    {{"domain_id", prepared.domain_id},
                     {"reason", "incomplete_gpu_direct_prepare"}});
                for (auto *pending_dgo : local_dgos)
                {
                    if (pending_dgo)
                        pending_dgo->clearPendingGpuDirectExpertTransfers();
                }
                return false;
            }
        }

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;

            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;
            if (!dgo->activatePendingGpuDirectExpertTransfers())
            {
                LOG_ERROR("[RankOrchestrator] Aborting MoE expert mask publish for domain="
                          << prepared.domain_id
                          << " because staged GPU-direct activation failed for participant "
                          << device_idx);
                PerfStatsCollector::addCounter(
                    "moe_rebalance",
                    "rank_local_mask_publish_aborts",
                    1.0,
                    "rebalance",
                    stats_device,
                    {{"domain_id", prepared.domain_id},
                     {"reason", "gpu_direct_activation_failed"}});
                for (auto *pending_dgo : local_dgos)
                {
                    if (pending_dgo)
                        pending_dgo->clearPendingGpuDirectExpertTransfers();
                }
                return false;
            }
        }

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            if (device_idx >= prepared.masks_by_participant.size())
                continue;

            auto *dgo = local_dgos[device_idx];
            if (!dgo)
                continue;

            const ReceivedWeightsMap empty_received;
            const auto &received =
                device_idx < prepared.received_by_device.size()
                    ? prepared.received_by_device[device_idx]
                    : empty_received;
            dgo->applyExpertMasksForDomain(
                prepared.domain_id,
                prepared.masks_by_participant[device_idx],
                received);
            ++applied;
        }

        if (!pp_stage_runners_.empty())
        {
            for (auto &runner : pp_stage_runners_)
            {
                if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
                {
                    if (!rank->applyMoEExpertMasksForAllDevices(
                        prepared.masks_by_participant,
                        prepared.domain_id))
                    {
                        LOG_ERROR("[RankOrchestrator] Nested LocalTP MoE expert mask publish failed for domain="
                                  << prepared.domain_id);
                        return false;
                    }
                    ++applied;
                    continue;
                }
                if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                {
                    if (!prepared.masks_by_participant.empty())
                    {
                        dgo->applyExpertMasksForDomain(
                            prepared.domain_id,
                            prepared.masks_by_participant.front(),
                            ReceivedWeightsMap{});
                        ++applied;
                    }
                }
            }
        }
        const auto publish_end = Clock::now();
        PerfStatsCollector::recordTimingNs(
            "moe_rebalance",
            "rank_local_mask_publish",
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                      publish_end - publish_start)
                                      .count()),
            "rebalance",
            stats_device,
            phase_tags);
        LOG_DEBUG("[RankOrchestrator] Applied LocalTP MoE expert masks to "
                  << applied << "/" << device_runners_.size() << " device runners");
        return true;
    }

    void RankOrchestrator::clearPendingGpuDirectExpertTransfersForAllDevices()
    {
        for (auto &runner : device_runners_)
        {
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                dgo->clearPendingGpuDirectExpertTransfers();
        }
        for (auto &runner : pp_stage_runners_)
        {
            if (auto *rank = dynamic_cast<RankOrchestrator *>(runner.get()))
            {
                rank->clearPendingGpuDirectExpertTransfersForAllDevices();
                continue;
            }
            if (auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(runner.get()))
                dgo->clearPendingGpuDirectExpertTransfers();
        }
    }

    bool RankOrchestrator::applyMoEExpertMasksForAllDevices(
        const std::vector<std::vector<std::vector<bool>>> &masks_by_participant,
        const std::string &domain_id,
        const std::vector<std::vector<std::vector<bool>>> *transfer_masks_by_participant)
    {
        auto prepared = prepareMoEExpertMaskTransfersForAllDevices(
            masks_by_participant,
            domain_id,
            transfer_masks_by_participant);
        return publishPreparedMoEExpertMasksForAllDevices(prepared);
    }

    void RankOrchestrator::setExpertReplicaSetForAllDevices(const ExpertReplicaSet &replicas)
    {
        PerfStatsCollector::addCounter(
            "moe_rebalance",
            "rank_replica_set_device_broadcasts",
            static_cast<double>(device_runners_.size()),
            "rebalance",
            primaryDeviceId().toString(),
            {{"domain_id", replicas.domain_id},
             {"replicas", std::to_string(replicas.num_replicated)}});

        for (size_t device_idx = 0; device_idx < device_runners_.size(); ++device_idx)
        {
            auto *dgo = dynamic_cast<DeviceGraphOrchestrator *>(device_runners_[device_idx].get());
            if (dgo)
                dgo->setExpertReplicaSet(replicas, static_cast<int>(device_idx));
        }
    }

    void RankOrchestrator::setSuppressTimeline(bool suppress)
    {
        for (auto &runner : device_runners_)
            runner->setSuppressTimeline(suppress);
    }

    void RankOrchestrator::setAccumulatePrefill(bool accumulate)
    {
        for (auto &runner : device_runners_)
            runner->setAccumulatePrefill(accumulate);
    }

    void RankOrchestrator::flushStageTimeline()
    {
        // Print per-device GPU stage timelines
        for (auto &runner : device_runners_)
            runner->flushStageTimeline();

        // Print TP orchestrator decode summary if profiling data was collected
        if (tp_decode_stats_.iterations > 0)
        {
            const size_t n = tp_decode_stats_.iterations;
            const double avg_wall = tp_decode_stats_.total_wall_ms / n;
            const double avg_dispatch = tp_decode_stats_.total_dispatch_ms / n;
            const double avg_wait = tp_decode_stats_.total_wait_ms / n;
            const double avg_gather = tp_decode_stats_.total_gather_ms / n;
            // "Other" captures orchestrator overhead not attributed to
            // dispatch/wait/gather: exception checking, position tracking,
            // condition evaluation, etc.
            const double avg_other = avg_wall - avg_dispatch - avg_wait - avg_gather;

            fort::utf8_table table;
            table.set_border_style(FT_DOUBLE2_STYLE);

            // Title row
            {
                std::ostringstream title;
                title << "TP DECODE ORCHESTRATOR PROFILING"
                      << " (avg of " << n << " decode steps, "
                      << device_runners_.size() << " devices)";
                table << title.str() << "" << "" << fort::endr;
                table[0][0].set_cell_span(3);
                table[0][0].set_cell_text_align(fort::text_align::center);
            }

            // Summary
            {
                std::ostringstream info;
                info << std::fixed << std::setprecision(2);
                info << "Wall Avg: " << avg_wall << " ms/tok";
                if (avg_wall > 0)
                    info << "  |  " << std::setprecision(1) << (1000.0 / avg_wall) << " tok/s";
                table << info.str() << "" << "" << fort::endr;
                table[1][0].set_cell_span(3);
            }

            // Header
            table << fort::header << "PHASE" << "AVG (ms)" << "%" << fort::endr;
            table.column(0).set_cell_text_align(fort::text_align::left);
            table.column(1).set_cell_text_align(fort::text_align::right);
            table.column(2).set_cell_text_align(fort::text_align::right);

            auto fmt_row = [&](const char *name, double ms)
            {
                std::ostringstream ms_str, pct_str;
                ms_str << std::fixed << std::setprecision(3) << ms;
                double pct = avg_wall > 0 ? (ms / avg_wall) * 100.0 : 0.0;
                pct_str << std::fixed << std::setprecision(1) << pct << "%";
                table << name << ms_str.str() << pct_str.str() << fort::endr;
            };

            fmt_row("Dispatch (wake workers)", avg_dispatch);
            fmt_row("Wait (device forward)", avg_wait);
            fmt_row("Gather logits", avg_gather);
            if (avg_other > 0.001)
                fmt_row("Other overhead", avg_other);

            // Separator + total
            table << fort::separator;
            {
                std::ostringstream ms_str;
                ms_str << std::fixed << std::setprecision(3) << avg_wall;
                table << "TOTAL" << ms_str.str() << "100.0%" << fort::endr;
            }

            std::cout << table.to_string() << std::flush;
            tp_decode_stats_.reset();
        }
    }

    PrefixCacheTelemetrySources RankOrchestrator::prefixCacheTelemetrySources() const
    {
        PrefixCacheTelemetrySources result;
        const auto append = [&](const auto &runners) {
            for (const auto &runner : runners)
                if (runner)
                {
                    const auto sources = runner->prefixCacheTelemetrySources();
                    result.insert(result.end(), sources.begin(), sources.end());
                }
        };
        append(device_runners_);
        append(pp_stage_runners_);
        return result;
    }

    /** @copydoc RankOrchestrator::collectPrefixCacheShutdown */
    void RankOrchestrator::collectPrefixCacheShutdown(PrefixCacheShutdown &shutdown) const
    {
        for (const auto &runner : device_runners_)
            if (runner) runner->collectPrefixCacheShutdown(shutdown);
        for (const auto &runner : pp_stage_runners_)
            if (runner) runner->collectPrefixCacheShutdown(shutdown);
    }

} // namespace llaminar2
