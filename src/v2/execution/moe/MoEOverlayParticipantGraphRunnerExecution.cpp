/**
 * @file MoEOverlayParticipantGraphRunnerExecution.cpp
 * @brief Participant graph runner sparse protocol execution, rebalance transactions, and retirement.
 */
#include "MoEOverlayParticipantGraphRunnerInternal.h"

namespace llaminar2
{
    /**
     * @brief Execute one exact live-row sparse protocol transaction.
     *
     * Tokens are intentionally not read: the continuation rank alone owns
     * token embedding and publishes the routed hidden rows through the first
     * sparse dispatch boundary.  The supplied count is still authoritative for
     * graph-cache identity and for advancing this participant's request cursor.
     */
    bool MoEOverlayParticipantGraphRunner::forward(
        const int *tokens,
        int seq_len)
    {
        /*
         * The unsegmented public runner API has no phase parameter. Preserve
         * its established serial-row attribution here; the explicit chunked
         * prefill API below passes Prefill for every member of a request,
         * including its one-row tail.
         */
        return executeAtLogicalStep(
            tokens,
            seq_len,
            position_,
            seq_len == 1
                ? SparseTransactionPhase::Decode
                : SparseTransactionPhase::Prefill);
    }

    bool MoEOverlayParticipantGraphRunner::forwardPrefill(
        const int *tokens,
        int seq_len)
    {
        return executeAtLogicalStep(
            tokens,
            seq_len,
            position_,
            SparseTransactionPhase::Prefill);
    }

    /**
     * @brief Execute one expert-only graph using an explicit protocol cursor.
     *
     * The caller supplies the continuation runner's logical start row.  We
     * require it to match the local cursor before executing, which catches a
     * missing prefix/reset publication before it can become a sparse MPI
     * mismatch. The already-built fixed-capacity graph remains independent of
     * the continuation graph's capture history because its manual boundaries
     * receive the shared identity directly.
     */
    bool MoEOverlayParticipantGraphRunner::executeAtLogicalStep(
        const int *tokens,
        int seq_len,
        int logical_step,
        SparseTransactionPhase phase,
        int physical_rows)
    {
        (void)tokens;
        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            LOG_ERROR("Rank-local expert endpoints require their original continuation coordinator");
            return false;
        }
        if (physical_rows < 0)
            physical_rows = seq_len;
        const ExpertHistogramSource histogram_source =
            phase == SparseTransactionPhase::Decode
                ? ExpertHistogramSource::DecodeToken
                : (phase == SparseTransactionPhase::Prefill
                       ? ExpertHistogramSource::PrefillChunk
                       : ExpertHistogramSource::SyntheticTest);
        MoEOverlayInferenceInterferenceScope interference_scope(
            interference_probe_.get(),
            makeMoEOverlayInferenceWorkloadIdentity(
                histogram_source,
                seq_len,
                physical_rows,
                /*transaction_count=*/1,
                /*speculative_depth=*/0));
        try
        {
            if (seq_len <= 0 || physical_rows < seq_len || logical_step < 0 ||
                logical_step != position_)
            {
                interference_scope.discard();
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Invalid sparse "
                    "collective request cursor logical_step="
                    << logical_step << " local_position=" << position_
                    << " rows=" << seq_len);
                return false;
            }
            if (overlay_collective_request_generation_ == 0)
            {
                interference_scope.discard();
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Refusing sparse "
                    "graph execution before the orchestration layer published "
                    "a request generation");
                return false;
            }
            auto &cached = graphForRows(seq_len);
            if (!stampMoEOverlayCollectiveRuntime(
                    cached,
                    overlay_collective_request_generation_,
                    static_cast<uint64_t>(logical_step),
                    phase))
            {
                interference_scope.discard();
                return false;
            }
            std::string execution_error;
            const bool ok = executor_.executeRetainedMultiDevice(
                cached.execution_plan, &execution_error);
            if (!ok && !execution_error.empty())
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Retained participant schedule failed: "
                    << execution_error);
            }
            if (!ok)
                interference_scope.discard();
            if (ok)
            {
                position_ += seq_len;
                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "fixed_capacity_graph_reuses",
                    1.0,
                    phase == SparseTransactionPhase::Decode
                        ? "decode"
                        : "prefill",
                    participantDeviceList(local_participants_),
                    {{"participants", participantIdList(local_participants_)},
                     {"logical_rows", std::to_string(seq_len)},
                     {"row_capacity", std::to_string(
                          config_.max_graph_activation_rows)},
                     {"parallel_gpu_waves",
                      std::to_string(
                          cached.execution_plan.concurrent_gpu_wave_count)},
                     {"max_wave_width",
                      std::to_string(
                          cached.execution_plan.max_wave_width)},
                     {"allocation_policy", "setup_only"}});
                const char *const evidence_phase =
                    phase == SparseTransactionPhase::Decode
                        ? "decode"
                        : "prefill";
                /*
                 * Fold identity into a bounded sequence witness; never retain
                 * request generations or logical steps as aggregation keys.
                 */
                PerfStatsCollector::recordOrderedSequenceStep(
                    "forward_graph",
                    "moe_overlay_collective_transaction_sequence",
                    {overlay_collective_request_generation_,
                     static_cast<uint64_t>(logical_step),
                     static_cast<uint64_t>(seq_len),
                     static_cast<uint64_t>(seq_len)},
                    evidence_phase,
                    participantDeviceList(local_participants_),
                    {{"role", "expert_participant_graph"},
                     {"identity_source", "orchestration_request_and_chunk"},
                     {"logical_step_semantics", "monotonic_transaction"}});
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_collective_transaction",
                    1.0,
                    evidence_phase,
                    participantDeviceList(local_participants_),
                    {{"role", "expert_participant_graph"},
                     {"identity_source", "orchestration_request_and_chunk"},
                     {"logical_step_semantics", "monotonic_transaction"},
                     {"logical_rows", std::to_string(seq_len)}});
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                ok ? "successful_steps" : "failed_steps",
                1.0,
                seq_len == 1 ? "decode" : "prefill",
                participantDeviceList(local_participants_),
                    {
                        {"participants", participantIdList(local_participants_)},
                        {"logical_rows", std::to_string(seq_len)},
                    });
            return ok;
        }
        catch (const std::exception &error)
        {
            interference_scope.discard();
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Forward failed on rank "
                << config_.mpi_context->rank() << ": " << error.what());
            return false;
        }
    }

    /**
     * @brief Stamp all sparse manual boundaries with this rank's shared operation identity.
     *
     * The root and remote runners derive @p logical_step from the same
     * request cursor and frozen chunk schedule. The stage objects themselves
     * may have unrelated capture/replay histories, so this loop is the one
     * place that connects their wire packets to the root-authoritative
     * request generation.
     */
    bool MoEOverlayParticipantGraphRunner::stampMoEOverlayCollectiveRuntime(
        CachedParticipantGraph &cached,
        uint64_t generation_id,
        uint64_t logical_step,
        SparseTransactionPhase phase,
        int mtp_graph_depth,
        uint64_t placement_epoch)
    {
        if (!cached.graph || generation_id == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Invalid sparse "
                "collective runtime stamp");
            return false;
        }

        using Semantics = IComputeStage::MoEOverlayCollectiveRuntimeParams::
            ExecutionSemantics;
        Semantics execution_semantics = Semantics::Unspecified;
        switch (phase)
        {
        case SparseTransactionPhase::Prefill:
            execution_semantics = Semantics::Prefill;
            break;
        case SparseTransactionPhase::Decode:
            execution_semantics = Semantics::Decode;
            break;
        case SparseTransactionPhase::MTPDraft:
            execution_semantics = Semantics::MTPDraft;
            break;
        case SparseTransactionPhase::GroupedVerifier:
            execution_semantics = Semantics::GroupedVerifier;
            break;
        }
        const bool main_semantics =
            phase == SparseTransactionPhase::Prefill ||
            phase == SparseTransactionPhase::Decode;
        if ((main_semantics && mtp_graph_depth != -1) ||
            (!main_semantics && mtp_graph_depth < 0) ||
            (phase == SparseTransactionPhase::MTPDraft &&
             (cached.role != ParticipantGraphFamilyRole::MTPDraft ||
              cached.mtp_graph_depth != mtp_graph_depth)) ||
            (phase != SparseTransactionPhase::MTPDraft &&
             cached.role != ParticipantGraphFamilyRole::Main))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Sparse graph role and MTP namespace depth disagree");
            return false;
        }

        const IComputeStage::MoEOverlayCollectiveRuntimeParams params{
            .generation_id = generation_id,
            .step_id = logical_step,
            .execution_semantics = execution_semantics,
            .mtp_depth = mtp_graph_depth,
            .placement_epoch = placement_epoch,
        };
        size_t stamped_stages = 0;
        for (const auto &node_name : cached.graph->getExecutionOrder())
        {
            auto *node = cached.graph->getNode(node_name);
            if (!node || !node->stage ||
                !node->stage->hasMoEOverlayCollectiveRuntimeParams())
            {
                continue;
            }
            node->stage->updateMoEOverlayCollectiveRuntimeParams(params);
            ++stamped_stages;
        }
        if (stamped_stages == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Graph-native overlay "
                "participant graph has no sparse stages to stamp");
            return false;
        }
        return true;
    }

    /**
     * @brief Accept a root-published request generation for future sparse operations.
     */
    bool MoEOverlayParticipantGraphRunner::setMoEOverlayCollectiveRequestGeneration(
        uint64_t generation_id)
    {
        if (generation_id == 0)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Sparse collective "
                "request generation must be non-zero");
            return false;
        }
        overlay_collective_request_generation_ = generation_id;
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::
        setMoEOverlayInferenceInterferenceProbe(
            std::shared_ptr<MoEOverlayInferenceInterferenceProbe> probe)
    {
        if (!probe || (interference_probe_ && interference_probe_ != probe))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Requires one stable "
                "non-null ExpertOverlay interference probe");
            return false;
        }
        interference_probe_ = std::move(probe);
        return true;
    }

    /**
     * @brief Original mapped submission retained between publish and retire.
     *
     * This is the same local state previously held on executeMappedFollowerTransaction's
     * stack. Moving it into the original runner lets its existing coordinator
     * publish all endpoints before launching the matching continuation graph.
     */
    struct MoEOverlayParticipantGraphRunner::MappedFollowerTransaction
    {
        using Clock = std::chrono::steady_clock;
        struct CPULayerTimelineSample
        {
            int participant_id = -1;
            int model_layer_index = -1;
            std::uint32_t stage_ordinal = 0u;
            std::size_t live_rows = 0u;
            std::size_t live_entries = 0u;
            std::uint64_t dispatch_wait_ns = 0u;
            std::uint64_t packet_prepare_ns = 0u;
            std::uint64_t expert_compute_ns = 0u;
            std::uint64_t local_execute_ns = 0u;
            std::uint64_t endpoint_post_service_ns = 0u; ///< Nested in local_execute_ns.
            std::uint64_t joint_publication_ns = 0u;
            std::optional<MoECPUExpertShardRouteStage::RootTimingSample> joint_root; ///< Absent for contributor/failed/unobserved calls.
            std::uint64_t return_publication_ns = 0u;
        };
        struct ArmedLane
        {
            std::shared_ptr<CachedParticipantGraph::MappedLaneAuthority> authority;
            MoEOverlayActivationEpochIdentity identity;
            std::optional<MoEOverlayActivationEpochTraffic> original_completed_traffic;
        };
        struct SubmittedInferenceBoundary
        {
            ParticipantGpuRuntime *runtime = nullptr;
            std::uint64_t generation = 0u;
            bool graph_submitted = false;
        };
        struct PreparedGpuSubmission
        {
            std::shared_ptr<CachedParticipantGraph::MappedGPUFollowerEndpoint> endpoint;
            IDeviceContext *context = nullptr;
            IWorkerGPUContext *worker = nullptr;
            bool had_native_transaction = false;
        };
        struct SubmittedGpuTerminal
        {
            CachedParticipantGraph::MappedGPUFollowerEndpoint *endpoint = nullptr;
            DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket ticket;
        };
        /** One bounded worker handle and its original native enqueue outcome. */
        struct PendingGpuEnqueue
        {
            std::future<void> completion;
            CachedParticipantGraph::MappedGPUFollowerEndpoint *endpoint = nullptr;
            DeviceGraphExecutor::GraphSegmentCache::CaptureStreamTerminalTicket terminal;
            bool queued = false;
            bool joined = false;
            bool launch_attempted = false;
            bool successful = false;
            bool terminal_recorded = false;
            bool had_native_transaction = false;
            std::exception_ptr failure;
        };
        explicit MappedFollowerTransaction(
            CachedParticipantGraph &original,
            const MoEOverlayInferenceTransactionTicket &original_ticket)
            : cached(&original), ticket(original_ticket)
        {
        }
        CachedParticipantGraph *cached = nullptr;
        MoEOverlayInferenceTransactionTicket ticket;
        std::optional<MoEOverlayOriginalNativeInvocationRetirement> original_native_retirement;
        std::optional<MoEOverlayInferenceOriginalSlot> original_slot; ///< SAME submitted protocol slot, retained until all aggregate proofs retire.
        bool native_terminals_joined = false;
        std::shared_ptr<const std::function<void()>> original_source_cancellation; ///< Immutable SAME request callback retained by this original submission through every enqueue/drain.
        std::mutex source_cancellation_failure_mutex;
        std::exception_ptr source_cancellation_failure; ///< First original source/channel cancellation fault; native drain never erases it.

        /**
         * @brief Preserve the original cancellation failure beyond native proof.
         * @throws The original exception when its source cancellation failed.
         * Native event completion permits disposal, never successful packet or
         * protocol retirement after a failed original source cancellation.
         */
        void requireOriginalSourceCancellationSuccessful()
        {
            std::lock_guard lock(source_cancellation_failure_mutex);
            if (source_cancellation_failure)
                std::rethrow_exception(source_cancellation_failure);
        }

        /**
         * @brief Reuse cold-allocated metadata after its original native proof.
         * @param original Next retained family owned by the same runner.
         * @param next Exact next scheduler identity, not mutable model state.
         * @throws std::logic_error Before a prior submitted native parent is joined.
         */
        void reset(CachedParticipantGraph &original,
                   const MoEOverlayInferenceTransactionTicket &next)
        {
            requireOriginalSubmissionRetired(pending_gpu_enqueues, submitted_boundaries,
                                             native_terminals_joined);
            cached = &original;
            ticket = next;
            original_slot.reset();
            original_native_retirement.reset();
            native_terminals_joined = false;
            source_cancellation_failure = {};
            original_source_cancellation = {};
            shape = nullptr;
            collect_timeline = false;
            physical_rows_u64 = 0u;
            cpu_dispatch_wait_ns = cpu_packet_prepare_ns = cpu_expert_compute_ns = 0u;
            cpu_return_publication_ns = endpoint_completion_wait_ns = 0u;
            captured_gpu_transactions = replayed_gpu_transactions = cpu_layer_dispatches = 0u;
            mapped_dispatch_bytes = mapped_return_bytes = mapped_dispatch_rows = 0u;
            mapped_return_rows = mapped_dispatch_entries = 0u;
            transaction_phase = nullptr;
            cpu_layer_timeline.clear();
            armed.clear();
            submitted_boundaries.clear();
            prepared_gpu_submissions.clear();
            submitted_gpu_terminals.clear();
            pending_gpu_enqueues.clear();
        }
        CachedParticipantGraph::MappedGPUFollowerShape *shape = nullptr;
        bool collect_timeline = false;
        std::uint64_t physical_rows_u64 = 0u;
        std::uint64_t cpu_dispatch_wait_ns = 0u;
        std::uint64_t cpu_packet_prepare_ns = 0u;
        std::uint64_t cpu_expert_compute_ns = 0u;
        std::uint64_t cpu_return_publication_ns = 0u;
        std::uint64_t endpoint_completion_wait_ns = 0u;
        std::vector<CPULayerTimelineSample> cpu_layer_timeline;
        std::vector<ArmedLane> armed;
        std::vector<SubmittedInferenceBoundary> submitted_boundaries;
        std::vector<PreparedGpuSubmission> prepared_gpu_submissions;
        std::vector<SubmittedGpuTerminal> submitted_gpu_terminals;
        std::vector<PendingGpuEnqueue> pending_gpu_enqueues;
        std::size_t captured_gpu_transactions = 0u;
        std::size_t replayed_gpu_transactions = 0u;
        std::size_t cpu_layer_dispatches = 0u;
        std::uint64_t mapped_dispatch_bytes = 0u;
        std::uint64_t mapped_return_bytes = 0u;
        std::uint64_t mapped_dispatch_rows = 0u;
        std::uint64_t mapped_return_rows = 0u;
        std::uint64_t mapped_dispatch_entries = 0u;
        const char *transaction_phase = nullptr;
    };

    /**
     * @brief Submit all mapped follower parents and retire their shared lease.
     *
     * Ticket reception is the only host-visible scheduler boundary. Every GPU
     * lane is device-owned after admission. A configured CPU lane is then
     * serviced directly from its shared pages at the explicit heterogeneous
     * boundary, while the already-submitted GPU parents continue concurrently.
     * The CPU loop never observes or forwards a GPU sibling's packet.
     */
    bool MoEOverlayParticipantGraphRunner::executeMappedFollowerTransaction(
        CachedParticipantGraph &cached,
        const MoEOverlayInferenceTransactionTicket &ticket,
        std::string *error)
    {
        MappedFollowerTransaction submission(cached, ticket);
        bool success = false;
        std::string primary_error;
        try
        {
            success = submitMappedFollowerTransaction(submission, &primary_error) &&
                      serviceMappedCPUFollowers(submission, &primary_error) &&
                      retireMappedFollowerTransaction(submission, &primary_error);
        }
        catch (const std::exception &exception)
        {
            primary_error = exception.what();
        }
        catch (...)
        {
            primary_error = "Non-standard original expert transaction failure";
        }
        if (!success)
            LOG_ERROR("[MoEOverlayParticipantGraphRunner] Original mapped follower transaction failed before cancellation: "
                << primary_error << "; ticket=" << ticket.toString());
        if (!success && !submission.native_terminals_joined)
        {
            std::string retirement_error;
            if (!drainCancelledMappedFollowerTransaction(submission, &retirement_error))
            {
                LOG_ERROR("Original expert transaction cannot release an unproven native parent: "
                    << retirement_error << "; original failure=" << primary_error);
                std::terminate();
            }
        }
        if (!success && error) *error = primary_error;
        return success;
    }

    bool MoEOverlayParticipantGraphRunner::submitMappedFollowerTransaction(
        MappedFollowerTransaction &submission, std::string *error)
    {
        using Clock = std::chrono::steady_clock;
        using ArmedLane = MappedFollowerTransaction::ArmedLane;
        using CPULayerTimelineSample = MappedFollowerTransaction::CPULayerTimelineSample;
        auto &cached = *submission.cached;
        auto &ticket = submission.ticket;
        auto &shape = submission.shape;
        auto &collect_timeline = submission.collect_timeline;
        auto &physical_rows_u64 = submission.physical_rows_u64;
        auto &cpu_dispatch_wait_ns = submission.cpu_dispatch_wait_ns;
        auto &cpu_packet_prepare_ns = submission.cpu_packet_prepare_ns;
        auto &cpu_expert_compute_ns = submission.cpu_expert_compute_ns;
        auto &cpu_return_publication_ns = submission.cpu_return_publication_ns;
        auto &endpoint_completion_wait_ns = submission.endpoint_completion_wait_ns;
        auto &cpu_layer_timeline = submission.cpu_layer_timeline;
        auto &armed = submission.armed;
        auto &submitted_boundaries = submission.submitted_boundaries;
        auto &prepared_gpu_submissions = submission.prepared_gpu_submissions;
        auto &submitted_gpu_terminals = submission.submitted_gpu_terminals;
        auto &captured_gpu_transactions = submission.captured_gpu_transactions;
        auto &replayed_gpu_transactions = submission.replayed_gpu_transactions;
        auto &cpu_layer_dispatches = submission.cpu_layer_dispatches;
        auto &mapped_dispatch_bytes = submission.mapped_dispatch_bytes;
        auto &mapped_return_bytes = submission.mapped_return_bytes;
        auto &mapped_dispatch_rows = submission.mapped_dispatch_rows;
        auto &mapped_return_rows = submission.mapped_return_rows;
        auto &mapped_dispatch_entries = submission.mapped_dispatch_entries;
        auto &transaction_phase = submission.transaction_phase;
        const auto elapsed_nanoseconds = [](
                                             Clock::time_point begin,
                                             Clock::time_point end)
            -> std::uint64_t
        {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count();
            return elapsed > 0
                       ? static_cast<std::uint64_t>(elapsed)
                       : 0u;
        };
        const auto accumulate_elapsed = [&elapsed_nanoseconds](
                                            std::uint64_t &total,
                                            Clock::time_point begin,
                                            Clock::time_point end)
            -> std::uint64_t
        {
            const std::uint64_t elapsed = elapsed_nanoseconds(begin, end);
            if (elapsed != 0u &&
                total <= std::numeric_limits<std::uint64_t>::max() -
                             elapsed)
            {
                total += elapsed;
            }
            return elapsed;
        };
        if (error)
            error->clear();
        const auto reject = [error](std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        };
        using SubmittedInferenceBoundary = MappedFollowerTransaction::SubmittedInferenceBoundary;
        using PreparedGpuSubmission = MappedFollowerTransaction::PreparedGpuSubmission;
        using SubmittedGpuTerminal = MappedFollowerTransaction::SubmittedGpuTerminal;
        collect_timeline = PerfStatsCollector::isDomainEnabled("moe_overlay_participant_graph");
        if (serving_graph_family_lifecycle_ ==
            ServingGraphFamilyLifecycle::Built)
        {
            return reject(
                "Mapped ExpertOverlay follower received a ticket before its physical-fabric graph branches were sealed");
        }
        physical_rows_u64 =
            static_cast<uint64_t>(ticket.request_count) *
            static_cast<uint64_t>(ticket.physical_rows_per_request);
        if (physical_rows_u64 == 0u ||
            physical_rows_u64 >
                static_cast<uint64_t>(std::numeric_limits<int>::max()))
        {
            return reject(
                "Mapped ExpertOverlay follower ticket has an invalid physical row geometry");
        }
        shape = cached.mappedShapeForRows(
            static_cast<int>(physical_rows_u64));
        if (!shape)
        {
            std::vector<int> retained_rows;
            for(const auto& retained:cached.mapped_gpu_shapes)retained_rows.push_back(retained.physical_rows);
            return reject(MoEOverlayFollowerRowGeometry::missingShapeDiagnostic(
                static_cast<int>(physical_rows_u64),retained_rows));
        }
        if(shape->endpoints.size()+cached.mapped_cpu_endpoints.size()!=cached.mapped_lane_authorities.size())
            return reject("Mapped ExpertOverlay follower exact executable lost its original endpoint/lane roster for physical_rows="+
                std::to_string(physical_rows_u64));
        if (collect_timeline)
        {
            std::size_t layer_capacity = 0u;
            for (const auto &endpoint : cached.mapped_cpu_endpoints)
            {
                if (endpoint)
                    layer_capacity += endpoint->layers.size();
            }
            cpu_layer_timeline.reserve(layer_capacity);
        }

        LOG_DEBUG(
            "[MoEOverlayParticipantGraphRunner] Admitted mapped follower "
            "transaction role="
            << static_cast<std::uint32_t>(ticket.graph_role)
            << " command=" << ticket.command_id
            << " ordinal=" << ticket.transaction_ordinal
            << " logical_step=" << ticket.logical_step_id
            << " physical_rows=" << physical_rows_u64
            << " gpu_endpoints=" << shape->endpoints.size()
            << " cpu_endpoints=" << cached.mapped_cpu_endpoints.size());

        armed.reserve(cached.mapped_lane_authorities.size());

        const auto timeout_eligibility =
            MoEOverlayActivationRendezvousDeadline::begin(
                MoEOverlayActivationRendezvousKind::EndpointCompletion,
                std::chrono::milliseconds(
                    collective_timeout_policy::
                        kDefaultCollectiveTimeoutMs));
        const auto timeout_not_before_ns =
            timeout_eligibility.deadlineNanoseconds();
        if (!timeout_not_before_ns)
        {
            return reject(
                "Mapped ExpertOverlay follower could not derive a watchdog timeout lower bound");
        }

        const auto abort_armed = [this, &submission]() noexcept
        {
            cancelMappedFollowerTransaction(submission);
        };

        for (const auto &authority : cached.mapped_lane_authorities)
        {
            if (!authority || !authority->protocol ||
                authority->next_epoch_generation == 0u ||
                authority->next_epoch_generation >
                    kMoEOverlayActivationMaxGeneration)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower exhausted or lost a lane generation authority");
            }
            std::string arm_error;
            const uint64_t generation =
                authority->next_epoch_generation;
            std::optional<MoEOverlayActivationEpochIdentity> identity;
            if (submission.original_slot)
            {
                auto *mapped = dynamic_cast<IMoEOverlayMappedActivationTransport *>(
                    authority->transport_lifetime.get());
                if (!mapped || !original_native_slot_geometry_)
                    throw std::logic_error("Original native admission lost its declared mapped geometry");
                identity = mapped->publishOriginalNativeInvocation(*submission.original_slot,
                    authority->participant_id, authority->graph_family_ordinal,
                    generation, *timeout_not_before_ns);
            }
            else
            {
                identity = authority->protocol->arm(ticket, generation,
                    *timeout_not_before_ns, &arm_error);
            }
            if (!identity)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower could not arm participant " +
                        std::to_string(authority->participant_id) + ": " +
                        arm_error);
            }
            ++authority->next_epoch_generation;
            armed.push_back(ArmedLane{
                .authority = authority,
                .identity = *identity,
            });
        }

        submitted_boundaries.reserve(shape->endpoints.size());
        prepared_gpu_submissions.reserve(shape->endpoints.size());
        submitted_gpu_terminals.reserve(shape->endpoints.size());
        transaction_phase =
            ticket.graph_role ==
                    MoEOverlayInferenceGraphRole::MainPrefill
                ? "prefill"
                : "decode";

        /*
         * Resolve every immutable launch dependency before publishing any
         * inference admission.  This keeps setup errors exactly reversible:
         * after the second pass begins, every receipt already protects a real
         * executable that is ready to enter its backend worker queue.
         */
        for (const auto &endpoint : shape->endpoints)
        {
            if (!endpoint || !endpoint->graph || !endpoint->executor ||
                !endpoint->lane_authority ||
                !endpoint->cache.capture_stream)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower endpoint lost its retained graph identity");
            }
            const auto context_it = execution_contexts_.find(endpoint->device);
            IWorkerGPUContext *const worker =
                participantWorkerContext(endpoint->device);
            if (context_it == execution_contexts_.end() ||
                !context_it->second || !worker ||
                endpoint->graph->nativeCaptureEnvelope() !=
                    GraphNativeCaptureEnvelope::
                        DeviceOwnedTimelineTransaction)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay follower endpoint is missing its device context or native timeline envelope");
            }

            prepared_gpu_submissions.push_back({
                .endpoint = endpoint,
                .context = context_it->second,
                .worker = worker,
                .had_native_transaction =
                    endpoint->cache.retained_full_graph_replay.valid(),
            });
        }

        const auto cancel_unsubmitted_boundaries = [&]() noexcept
        {
            bool cancelled_every_boundary = true;
            for (auto boundary = submitted_boundaries.rbegin();
                 boundary != submitted_boundaries.rend();
                 ++boundary)
            {
                if (boundary->graph_submitted)
                    continue;
                cancelled_every_boundary =
                    boundary->runtime &&
                    boundary->runtime->inference_boundary_receipt
                        .cancelUnsubmitted(boundary->generation) &&
                    cancelled_every_boundary;
            }
            return cancelled_every_boundary;
        };

        /*
         * Publish every local follower admission before launching the first
         * graph.  The controller's topology preflight can therefore observe
         * either the wholly quiescent prior transaction or an in-flight member
         * of this transaction; it can never certify a graph that has already
         * entered a backend queue but is still absent from its receipt.
         */
        if (!prepared_gpu_submissions.empty())
        {
            for (const auto &prepared : prepared_gpu_submissions)
            {
                auto &runtime = participantGpuRuntimeForParticipant(
                    prepared.endpoint->participant_id);
                if (runtime.device != prepared.endpoint->device)
                {
                    (void)cancel_unsubmitted_boundaries();
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower prepared a cache on the wrong controller-runtime device");
                }
                const std::uint64_t boundary_generation =
                    runtime.inference_boundary_receipt.beginSubmission();
                if (boundary_generation == 0u)
                {
                    (void)cancel_unsubmitted_boundaries();
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay follower exhausted its inference-boundary generation namespace for participant " +
                        std::to_string(prepared.endpoint->participant_id));
                }
                submitted_boundaries.push_back({
                    .runtime = &runtime,
                    .generation = boundary_generation,
                    .graph_submitted = false,
                });
            }
        }

        if (original_pooled_participants_ && !submission.original_source_cancellation &&
            std::any_of(prepared_gpu_submissions.begin(),prepared_gpu_submissions.end(),[&](const auto& prepared) {
                return original_pooled_participants_->hasOriginalExpertOwner(prepared.endpoint->device);
            })) {
            (void)cancel_unsubmitted_boundaries();
            abort_armed();
            return reject("Expert storage submission lacks its exact original source cancellation binding");
        }

        // Queue every original worker before returning to continuation launch.
        // Futures and results stay in the cold-reserved original submission;
        // no stack reference, detached callback or extra progress worker escapes.
        submission.pending_gpu_enqueues.resize(prepared_gpu_submissions.size());
        for (std::size_t index = 0; index < prepared_gpu_submissions.size(); ++index)
        {
            const auto prepared = prepared_gpu_submissions[index];
            auto *const endpoint = prepared.endpoint.get();
            auto &pending = submission.pending_gpu_enqueues[index];
            pending.endpoint = endpoint;
            pending.had_native_transaction = prepared.had_native_transaction;
            const auto epoch = transfer_progress_epochs_.find(endpoint->device);
            const auto branch_factory = epoch == transfer_progress_epochs_.end() ?
                GraphCaptureAuxiliaryBranchFactory{} : epoch->second->graphBranchFactory();
            try
            {
                pending.completion = prepared.worker->submitAsync(
                    [this, &submission, index, prepared, branch_factory]()
                    {
                        auto &owned = submission.pending_gpu_enqueues[index];
                        owned.successful = runOriginalEndpointEnqueue([&]()
                        {
                            auto &runtime=participantGpuRuntimeForParticipant(prepared.endpoint->participant_id);
                            if (original_native_slot_geometry_)
                            {
                                if (!submission.original_slot)
                                    throw std::logic_error("Original local follower has no submitted protocol slot");
                                submission.original_slot->domain()->requireActiveAdmission(*submission.original_slot);
                            }
                            // Aborted queued work must reject before even the
                            // predecessor device edge can become native work.
                            runtime.enqueueOriginalInferencePredecessor(*prepared.endpoint,prepared.worker);
                            owned.launch_attempted = true;
                            const bool launched = prepared.endpoint->executor->executeWithCachedGraphReplay(
                                *prepared.endpoint->graph, prepared.context, prepared.endpoint->cache,
                                prepared.endpoint->cache.capture_stream, prepared.worker,
                                nullptr, false, false, true, {},
                                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph,
                                {}, {}, {},
                                DeviceGraphExecutor::GraphInitialSubmissionPolicy::CaptureInstantiateAndLaunch,
                                branch_factory);
                            if (!launched)
                                return false;
                            submission.submitted_boundaries[index].graph_submitted = true;
                            owned.terminal = original_native_slot_geometry_
                                ? prepared.endpoint->cache.publishCaptureStreamTerminal(*submission.original_slot)
                                : prepared.endpoint->cache.publishCaptureStreamTerminal();
                            runtime.publishOriginalInferencePredecessor(prepared.endpoint,owned.terminal);
                            if (!prepared.endpoint->cache.retained_full_graph_replay.valid())
                                throw std::logic_error("Original expert enqueue lost its sealed native replay identity");
                            return owned.terminal.valid();
                        }, [&]() noexcept { cancelMappedFollowerTransaction(submission); });
                    });
                pending.queued = true;
            }
            catch (const std::exception &exception)
            {
                cancelMappedFollowerTransaction(submission);
                return reject("Original expert worker queue rejected submission: " + std::string(exception.what()));
            }
        }
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::serviceMappedCPUFollowers(
        MappedFollowerTransaction &submission, std::string *error)
    {
        using Clock = std::chrono::steady_clock;
        using ArmedLane = MappedFollowerTransaction::ArmedLane;
        using CPULayerTimelineSample = MappedFollowerTransaction::CPULayerTimelineSample;
        auto &cached = *submission.cached;
        auto &ticket = submission.ticket;
        auto &shape = submission.shape;
        auto &collect_timeline = submission.collect_timeline;
        auto &physical_rows_u64 = submission.physical_rows_u64;
        auto &cpu_dispatch_wait_ns = submission.cpu_dispatch_wait_ns;
        auto &cpu_packet_prepare_ns = submission.cpu_packet_prepare_ns;
        auto &cpu_expert_compute_ns = submission.cpu_expert_compute_ns;
        auto &cpu_return_publication_ns = submission.cpu_return_publication_ns;
        auto &endpoint_completion_wait_ns = submission.endpoint_completion_wait_ns;
        auto &cpu_layer_timeline = submission.cpu_layer_timeline;
        auto &armed = submission.armed;
        auto &submitted_boundaries = submission.submitted_boundaries;
        auto &prepared_gpu_submissions = submission.prepared_gpu_submissions;
        auto &submitted_gpu_terminals = submission.submitted_gpu_terminals;
        auto &captured_gpu_transactions = submission.captured_gpu_transactions;
        auto &replayed_gpu_transactions = submission.replayed_gpu_transactions;
        auto &cpu_layer_dispatches = submission.cpu_layer_dispatches;
        auto &mapped_dispatch_bytes = submission.mapped_dispatch_bytes;
        auto &mapped_return_bytes = submission.mapped_return_bytes;
        auto &mapped_dispatch_rows = submission.mapped_dispatch_rows;
        auto &mapped_return_rows = submission.mapped_return_rows;
        auto &mapped_dispatch_entries = submission.mapped_dispatch_entries;
        auto &transaction_phase = submission.transaction_phase;
        const auto elapsed_nanoseconds = [](
                                             Clock::time_point begin,
                                             Clock::time_point end)
            -> std::uint64_t
        {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count();
            return elapsed > 0
                       ? static_cast<std::uint64_t>(elapsed)
                       : 0u;
        };
        const auto accumulate_elapsed = [&elapsed_nanoseconds](
                                            std::uint64_t &total,
                                            Clock::time_point begin,
                                            Clock::time_point end)
            -> std::uint64_t
        {
            const std::uint64_t elapsed = elapsed_nanoseconds(begin, end);
            if (elapsed != 0u &&
                total <= std::numeric_limits<std::uint64_t>::max() -
                             elapsed)
            {
                total += elapsed;
            }
            return elapsed;
        };
        if (error)
            error->clear();
        const auto reject = [error](std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        };
        const auto abort_armed = [this, &submission]() noexcept
        {
            cancelMappedFollowerTransaction(submission);
        };

        const auto armed_lane_for = [&](const auto &authority)
            -> const ArmedLane *
        {
            const auto found = std::find_if(
                armed.begin(),
                armed.end(),
                [&](const ArmedLane &lane)
                {
                    return lane.authority == authority;
                });
            return found == armed.end() ? nullptr : &*found;
        };

        for (const auto &endpoint : cached.mapped_cpu_endpoints)
        {
            const ArmedLane *const lane = endpoint
                                               ? armed_lane_for(
                                                     endpoint->lane_authority)
                                               : nullptr;
            const auto context_it = endpoint
                                        ? execution_contexts_.find(
                                              endpoint->device)
                                        : execution_contexts_.end();
                if (!endpoint || !endpoint->device.is_cpu() || !lane ||
                !endpoint->lane_authority ||
                !endpoint->lane_authority->protocol ||
                !endpoint->lane.valid() ||
                !endpoint->input_rows || !endpoint->output_rows ||
                context_it == execution_contexts_.end() ||
                !context_it->second)
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU endpoint lost its exact protocol, packet, or execution-context authority");
            }

            auto &protocol = *endpoint->lane_authority->protocol;
            std::string protocol_error;
            if (!protocol.activate(
                    MoEOverlayActivationEndpoint::Follower,
                    lane->identity,
                    &protocol_error))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not activate participant " +
                    std::to_string(endpoint->participant_id) + ": " +
                    protocol_error);
            }

            /*
             * Bind the exact hidden matrix before the first descriptor is
             * acquired. The object address retained by every CPU local-expert
             * stage stays fixed; only this typed, ticket-derived view changes.
             * Metadata aliases are identical across shapes, while the hidden
             * pointer/layout pair is indivisible and may not be overridden by
             * a layer or inferred from live descriptor counts.
             */
            const auto admitted_payload =
                endpoint->lane.hostDispatchPayload(
                    static_cast<std::int32_t>(physical_rows_u64));
            if (!admitted_payload.hidden_rows_fp32 ||
                admitted_payload.hidden_row_capacity !=
                    static_cast<size_t>(physical_rows_u64) ||
                !isValidMoEOverlayActivationHiddenPayloadLayout(
                    admitted_payload.hidden_payload_layout))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not bind the exact ticket-selected activation payload view");
            }
            *endpoint->input_rows = admitted_payload;

            for (const auto &layer : endpoint->layers)
            {
                if (!layer.local_expert || layer.model_layer_index < 0)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower lost a prepared layer endpoint");
                }
                CPULayerTimelineSample layer_timing{
                    .participant_id = endpoint->participant_id,
                    .model_layer_index = layer.model_layer_index,
                    .stage_ordinal = layer.stage_ordinal,
                };
                const std::uint32_t bank =
                    moeOverlayActivationBufferIndex(layer.stage_ordinal);
                const std::uint64_t expected_timeline =
                    moeOverlayActivationLeasedTimelineValue(
                        moeOverlayActivationBufferVisit(
                            layer.stage_ordinal));
                /*
                 * Each routed layer is one collective rendezvous. Starting a
                 * fresh deadline here prevents a healthy 48-layer transaction
                 * from inheriting a wall-clock budget created before layer 0.
                 */
                const auto dispatch_rendezvous =
                    MoEOverlayActivationRendezvousDeadline::begin(
                        MoEOverlayActivationRendezvousKind::
                            DispatchPublication,
                        std::chrono::milliseconds(
                            collective_timeout_policy::
                                kDefaultCollectiveTimeoutMs));
                const auto dispatch_wait_begin = collect_timeline
                                                     ? Clock::now()
                                                     : Clock::time_point{};
                std::uint32_t dispatch_spin = 0u;
                while (protocol.dispatchTimeline(bank) < expected_timeline)
                {
                    if ((dispatch_spin & 4095u) == 4095u &&
                        !dispatch_rendezvous.waitingAllowed())
                    {
                        break;
                    }
                    if (protocol.endpointState(
                            MoEOverlayActivationEndpoint::Continuation) ==
                        MoEOverlayActivationEndpointState::Aborted)
                    {
                        break;
                    }
                    if (++dispatch_spin < 262144u)
                    {
#if defined(__x86_64__) || defined(_M_X64)
                        __builtin_ia32_pause();
#endif
                    }
                    else if ((dispatch_spin & 1023u) == 0u)
                    {
                        std::this_thread::yield();
                    }
                }

                auto descriptor = protocol.consumeDispatch(
                    lane->identity,
                    layer.stage_ordinal,
                    &protocol_error);
                if (collect_timeline)
                {
                    layer_timing.dispatch_wait_ns = accumulate_elapsed(
                        cpu_dispatch_wait_ns,
                        dispatch_wait_begin,
                        Clock::now());
                }
                if (!descriptor)
                {
                    /*
                     * This is a terminal-only snapshot of the shared protocol.
                     * It adds no steady-path polling, but distinguishes an
                     * expired watchdog from an illegal timeline regression or
                     * a continuation graph that never opened its generation.
                     */
                    const auto now = std::chrono::steady_clock::now();
                    const auto remaining_us =
                        dispatch_rendezvous.remainingMicroseconds(now);
                    const auto active_identity = protocol.activeIdentity();
                    const auto continuation = protocol.endpointStatus(
                        MoEOverlayActivationEndpoint::Continuation);
                    const auto follower = protocol.endpointStatus(
                        MoEOverlayActivationEndpoint::Follower);
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Mapped ExpertOverlay CPU follower could not consume layer "
                        << layer.model_layer_index
                        << " for participant " << endpoint->participant_id
                        << ": " << protocol_error
                        << "; terminal_snapshot={timeline="
                        << protocol.dispatchTimeline(bank)
                        << ",expected=" << expected_timeline
                        << ",deadline_remaining_us=" << remaining_us
                        << ",admission="
                        << static_cast<std::uint32_t>(
                               protocol.admissionState())
                        << ",active_generation="
                        << active_identity.epoch_generation
                        << ",active_placement_floor="
                        << active_identity.placement_epoch_floor
                        << ",continuation_state=" << continuation.state
                        << ",continuation_code=" << continuation.code
                        << ",continuation_operation="
                        << continuation.operation
                        << ",continuation_observed="
                        << continuation.observed_timeline
                        << ",continuation_diagnostic="
                        << continuation.failure_diagnostic
                        << ",continuation_auxiliary="
                        << continuation.failure_auxiliary
                        << ",continuation_generation="
                        << continuation.generation
                        << ",continuation_published="
                        << continuation.last_published_stage
                        << ",follower_state=" << follower.state
                        << ",follower_code=" << follower.code
                        << ",follower_operation=" << follower.operation
                        << ",follower_generation=" << follower.generation
                        << ",follower_consumed="
                        << follower.last_consumed_stage << "}";
                    if (config_.device_controller_fabric)
                    {
                        /*
                         * A missing dispatch can be downstream of the
                         * continuation group's captured epoch-acquire barrier.
                         * The barrier lives in mapped host pages, so this
                         * terminal-only diagnostic can expose every participant
                         * sequence without submitting GPU work or perturbing
                         * the inference stream whose failure we are reporting.
                         */
                        diagnostic
                            << ' '
                            << config_.device_controller_fabric
                                   ->describeInferenceEpochBarrier();
                    }
                    abort_armed();
                    return reject(diagnostic.str());
                }

                auto &input = *endpoint->input_rows;
                auto &output = *endpoint->output_rows;
                const auto packet_prepare_begin = collect_timeline
                                                      ? Clock::now()
                                                      : Clock::time_point{};
                if (descriptor->live_rows > physical_rows_u64 ||
                    descriptor->live_rows > input.row_capacity ||
                    descriptor->live_entries > input.entry_capacity ||
                    descriptor->live_entries >
                        descriptor->live_rows *
                            static_cast<std::uint64_t>(input.top_k))
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU dispatch descriptor exceeds its exact ticket or shared-page capacity");
                }

                switch (ticket.graph_role)
                {
                case MoEOverlayInferenceGraphRole::MainPrefill:
                case MoEOverlayInferenceGraphRole::MainDecode:
                    input.key = makeMoEOverlayCollectiveKey(
                        ticket.request_generation,
                        ticket.logical_step_id,
                        layer.model_layer_index,
                        endpoint->tier_index,
                        endpoint->routed_domain_ordinal,
                        endpoint->participant_id,
                        MoEOverlayCollectiveDirection::Dispatch);
                    input.key.histogram_source =
                        ticket.graph_role ==
                                MoEOverlayInferenceGraphRole::MainPrefill
                            ? ExpertHistogramSource::PrefillChunk
                            : ExpertHistogramSource::DecodeToken;
                    break;
                case MoEOverlayInferenceGraphRole::MTPDraft:
                case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                    input.key = makeMTPMoEOverlayCollectiveKey(
                        ticket.request_generation,
                        ticket.logical_step_id,
                        ticket.graph_role ==
                                MoEOverlayInferenceGraphRole::MTPDraft
                            ? cached.mtp_graph_depth
                            : ticket.draft_depth,
                        layer.model_layer_index,
                        endpoint->tier_index,
                        endpoint->routed_domain_ordinal,
                        endpoint->participant_id,
                        MoEOverlayCollectiveDirection::Dispatch);
                    input.key.histogram_source =
                        ExpertHistogramSource::GroupedVerifier;
                    break;
                case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
                case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
                case MoEOverlayInferenceGraphRole::None:
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower received a terminal or pipeline-only graph role");
                }
                input.residency_epoch = descriptor->placement_epoch;
                input.source_participant =
                    lane->identity.source_participant_id;
                input.target_participant = endpoint->participant_id;
                input.live_row_count =
                    static_cast<size_t>(descriptor->live_rows);
                input.live_entry_count =
                    static_cast<size_t>(descriptor->live_entries);
                layer_timing.live_rows = input.live_row_count;
                layer_timing.live_entries = input.live_entry_count;

                /*
                 * Match the device consume kernel's structural validation before
                 * CPU GEMM sees any shared bytes. Release publication of the
                 * descriptor is the acquire edge for every payload read below.
                 */
                const size_t expected_payload_bytes =
                    compactMoEOverlayDispatchBytes(input);
                bool valid_packet =
                    descriptor->payload_bytes == expected_payload_bytes &&
                    input.entry_offsets_host[0] == 0;
                std::int32_t previous_row = -1;
                std::int32_t previous_offset = 0;
                for (size_t row = 0;
                     valid_packet && row < input.live_row_count;
                     ++row)
                {
                    const std::int32_t row_id = input.row_ids_host[row];
                    const std::int32_t begin =
                        input.entry_offsets_host[row];
                    const std::int32_t end =
                        input.entry_offsets_host[row + 1u];
                    valid_packet =
                        row_id > previous_row && row_id >= 0 &&
                        static_cast<std::uint64_t>(row_id) <
                            physical_rows_u64 &&
                        begin == previous_offset && end > begin &&
                        end - begin <= input.top_k &&
                        end <= static_cast<std::int32_t>(
                                   input.live_entry_count);
                    previous_row = row_id;
                    previous_offset = end;
                }
                valid_packet = valid_packet &&
                               previous_offset ==
                                   static_cast<std::int32_t>(
                                       input.live_entry_count);
                if (!valid_packet)
                {
                    std::ostringstream diagnostic;
                    diagnostic
                        << "Mapped ExpertOverlay CPU follower rejected "
                           "malformed shared dispatch rows"
                        << " layer=" << layer.model_layer_index
                        << " participant=" << endpoint->participant_id
                        << " live_rows=" << input.live_row_count
                        << " live_entries=" << input.live_entry_count
                        << " payload_bytes=" << descriptor->payload_bytes
                        << " expected_payload_bytes="
                        << expected_payload_bytes
                        << " initial_offset="
                        << input.entry_offsets_host[0]
                        << " final_offset=" << previous_offset;
                    if (input.live_row_count > 0u)
                    {
                        diagnostic
                            << " first_row_id=" << input.row_ids_host[0]
                            << " first_row_end="
                            << input.entry_offsets_host[1u];
                    }
                    abort_armed();
                    return reject(diagnostic.str());
                }
                if (collect_timeline)
                {
                    layer_timing.packet_prepare_ns = accumulate_elapsed(
                        cpu_packet_prepare_ns,
                        packet_prepare_begin,
                        Clock::now());
                }

                output.live_row_count = 0u;
                const auto expert_compute_begin = collect_timeline
                                                      ? Clock::now()
                                                      : Clock::time_point{};
                bool expert_compute_ok = layer.local_expert->execute(context_it->second);
                const auto local_completed_at = collect_timeline ? Clock::now() : Clock::time_point{};
                if (collect_timeline)
                {
                    layer_timing.local_execute_ns = static_cast<std::uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            local_completed_at - expert_compute_begin).count());
                    layer_timing.endpoint_post_service_ns =
                        layer.local_expert->completedPostServiceNanoseconds();
                }
                std::optional<CPUExpertShardRouteCompletion> joint_completed;
                if(expert_compute_ok && layer.joint_route_collective) {
                    const auto joint_begin = collect_timeline ? Clock::now() : Clock::time_point{};
                    auto root_observation=layer.joint_route_collective->observeRootTiming(collect_timeline);
                    joint_completed=layer.joint_route_collective->executeOriginalJointPublication(
                        context_it->second,protocol,lane->identity,layer.stage_ordinal);
                    if (collect_timeline)
                        layer_timing.joint_publication_ns = static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                Clock::now() - joint_begin).count());
                    layer_timing.joint_root=root_observation.completed();
                    expert_compute_ok=joint_completed.has_value();
                }
                const bool joint_contributor =
                    layer.joint_route_collective &&
                    lane->identity.return_role ==
                        static_cast<uint32_t>(MoEOverlayActivationReturnRole::JointContributor);
                const auto expected_return_rows=joint_contributor?size_t{0}:input.live_row_count;
                if (collect_timeline)
                {
                    layer_timing.expert_compute_ns = accumulate_elapsed(
                        cpu_expert_compute_ns,
                        expert_compute_begin,
                        Clock::now());
                }
                if (!expert_compute_ok ||
                    output.live_row_count != expected_return_rows ||
                    output.residency_epoch != input.residency_epoch ||
                    output.source_participant != endpoint->participant_id ||
                    output.target_participant != input.source_participant)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU local expert failed or returned an incomplete packet for layer " +
                        std::to_string(layer.model_layer_index) +
                        " participant " +
                        std::to_string(endpoint->participant_id));
                }
                for (size_t row = 0;
                     row < output.live_row_count;
                     ++row)
                {
                    if (output.row_ids_host[row] !=
                        input.row_ids_host[row])
                    {
                        abort_armed();
                        return reject(
                            "Mapped ExpertOverlay CPU return row identity diverged from dispatch order");
                    }
                }

                const auto return_publication_begin = collect_timeline
                                                          ? Clock::now()
                                                          : Clock::time_point{};
                const std::uint64_t return_bytes =
                    moeOverlayReturnPayloadBytes(
                        joint_contributor?size_t{0}:input.live_entry_count,
                        static_cast<std::uint32_t>(input.d_model));
                const bool published=joint_completed
                    ?protocol.publishJointReturn(lane->identity,layer.stage_ordinal,return_bytes,*joint_completed,&protocol_error)
                    :protocol.publishReturn(lane->identity,layer.stage_ordinal,return_bytes,&protocol_error);
                if (!published)
                {
                    abort_armed();
                    return reject(
                        "Mapped ExpertOverlay CPU follower could not publish layer " +
                        std::to_string(layer.model_layer_index) + ": " +
                        protocol_error);
                }
                if (collect_timeline)
                {
                    layer_timing.return_publication_ns = accumulate_elapsed(
                        cpu_return_publication_ns,
                        return_publication_begin,
                        Clock::now());
                    cpu_layer_timeline.push_back(layer_timing);
                }
                ++cpu_layer_dispatches;
            }

            if (!protocol.complete(
                    MoEOverlayActivationEndpoint::Follower,
                    lane->identity,
                    &protocol_error))
            {
                abort_armed();
                return reject(
                    "Mapped ExpertOverlay CPU follower could not publish terminal completion: " +
                    protocol_error);
            }
        }

        return true;
    }

    /** @brief Host enqueue results only; actual native event proof remains a separate required phase. */
    struct MoEOverlayParticipantGraphRunner::MappedFollowerEnqueueJoin
    {
        bool successful = true;
        bool missing_terminal = false;
        std::string first_error;
    };

    void MoEOverlayParticipantGraphRunner::cancelMappedFollowerTransaction(
        MappedFollowerTransaction &submission) noexcept
    {
        if (submission.original_source_cancellation)
            cancelOriginalPooledSource(*submission.original_source_cancellation,
                submission.source_cancellation_failure_mutex,submission.source_cancellation_failure);
        for (const auto &lane : submission.armed)
        {
            if (!lane.authority || !lane.authority->protocol)
                continue;
            if (submission.original_slot)
            {
                try
                {
                    auto *mapped = dynamic_cast<IMoEOverlayMappedActivationTransport *>(
                        lane.authority->transport_lifetime.get());
                    if (!mapped)
                        throw std::logic_error("Original native cancellation lost its mapped transport");
                    mapped->cancelOriginalNativeInvocation(*submission.original_slot,
                        lane.authority->participant_id, lane.identity);
                }
                catch (...)
                {
                    std::lock_guard lock(submission.source_cancellation_failure_mutex);
                    if (!submission.source_cancellation_failure)
                        submission.source_cancellation_failure = std::current_exception();
                }
            }
            else
            {
                std::string ignored;
                (void)lane.authority->protocol->cancelOriginalInvocation(lane.identity,
                    MoEOverlayActivationStatusCode::ExplicitAbort, &ignored);
            }
        }
        // The original CPU collective is an independent explicit graph edge.
        // Releasing GPU/host activation waits cannot complete its missing peer;
        // preserve GlobalTP's existing fatal abort instead of stranding a gather.
        if(config_.cpu_joint_context)config_.cpu_joint_context->requestAbort();
    }

    MoEOverlayParticipantGraphRunner::MappedFollowerEnqueueJoin
    MoEOverlayParticipantGraphRunner::joinMappedFollowerEnqueues(MappedFollowerTransaction &submission)
    {
        MappedFollowerEnqueueJoin result;
        std::string queue_error;
        (void)joinAllOriginalNativeTerminals(submission.pending_gpu_enqueues,
            [](auto &pending)
            {
                if (!pending.queued || pending.joined)
                    return;
                try
                {
                    pending.completion.get();
                    pending.joined = true;
                }
                catch (...)
                {
                    pending.failure = std::current_exception();
                    pending.joined = true;
                    throw;
                }
            }, [&]() noexcept { cancelMappedFollowerTransaction(submission); }, &queue_error);
        for (auto &pending : submission.pending_gpu_enqueues)
        {
            if (!pending.queued || pending.failure || !pending.successful)
            {
                result.successful = false;
                if (result.first_error.empty())
                {
                    result.first_error = "Original expert enqueue did not complete successfully";
                    if (pending.failure)
                    {
                        try { std::rethrow_exception(pending.failure); }
                        catch (const std::exception &exception) { result.first_error = exception.what(); }
                        catch (...) { result.first_error = "Non-standard original expert enqueue failure"; }
                    }
                }
                cancelMappedFollowerTransaction(submission);
            }
            if (pending.launch_attempted && !pending.terminal.valid())
                result.missing_terminal = true;
            if (pending.terminal.valid() && !pending.terminal_recorded)
            {
                submission.submitted_gpu_terminals.push_back({pending.endpoint, pending.terminal});
                pending.terminal_recorded = true;
                if (pending.successful)
                {
                    if (pending.had_native_transaction)
                        ++submission.replayed_gpu_transactions;
                    else
                        ++submission.captured_gpu_transactions;
                }
            }
        }
        // Only completed enqueue handles can prove that a begun admission was
        // never launched. A possibly launched parent without its exact event
        // remains unproven regardless of another peer's successful completion.
        for (std::size_t index = 0; index < submission.submitted_boundaries.size(); ++index)
        {
            const auto &boundary = submission.submitted_boundaries[index];
            if (index < submission.pending_gpu_enqueues.size() &&
                submission.pending_gpu_enqueues[index].launch_attempted &&
                !submission.pending_gpu_enqueues[index].terminal.valid())
                continue; // Preserve the in-flight receipt for an unproven native launch.
            if (!boundary.runtime || boundary.graph_submitted)
                continue;
            auto &receipt = boundary.runtime->inference_boundary_receipt;
            if (receipt.submittedGeneration() == boundary.generation - 1u &&
                receipt.completedGeneration() == boundary.generation - 1u)
                continue;
            if (!receipt.cancelUnsubmitted(boundary.generation))
                result.missing_terminal = true;
        }
        {
            std::lock_guard lock(submission.source_cancellation_failure_mutex);
            if (submission.source_cancellation_failure) {
                result.successful = false;
                const auto prefix = result.first_error.empty() ? std::string{} : std::string("; ");
                try { std::rethrow_exception(submission.source_cancellation_failure); }
                catch (const std::exception& error) { result.first_error += prefix + "Original pooled source cancellation failed: " + error.what(); }
                catch (...) { result.first_error += prefix + "Original pooled source cancellation failed with a non-standard exception"; }
            }
        }
        return result;
    }

    bool MoEOverlayParticipantGraphRunner::drainCancelledMappedFollowerTransaction(
        MappedFollowerTransaction &submission, std::string *error)
    {
        const auto cancel = [this, &submission]() noexcept
        {
            cancelMappedFollowerTransaction(submission);
        };
        cancel();
        const auto enqueues = joinMappedFollowerEnqueues(submission);
        if (!joinAllOriginalNativeTerminals(submission.submitted_gpu_terminals,
                [&](const auto &terminal)
                {
                    if (!terminal.endpoint || !terminal.ticket.valid())
                        throw std::logic_error("Cancelled expert submission lost its exact native terminal");
                    terminal.endpoint->cache.waitForPublishedCaptureStreamTerminal(terminal.ticket,
                        DeviceGraphExecutor::GraphSegmentCache::HostFenceWaitPolicy::ActiveProgress);
                    participantGpuRuntimeForParticipant(terminal.endpoint->participant_id)
                        .retireOriginalInferencePredecessor(terminal.endpoint,terminal.ticket);
                }, cancel, error)) return false;
        if (enqueues.missing_terminal)
        {
            if (error) *error = enqueues.first_error.empty() ?
                "Original expert enqueue has no exact native terminal proof" : enqueues.first_error;
            return false;
        }
        const auto submitted = static_cast<std::size_t>(std::count_if(
            submission.submitted_boundaries.begin(), submission.submitted_boundaries.end(),
            [](const auto &boundary) { return boundary.graph_submitted; }));
        if (submitted != submission.submitted_gpu_terminals.size())
        {
            if (error) *error = "Cancelled original expert launch lacks its exact native terminal";
            return false;
        }
        for (const auto &boundary : submission.submitted_boundaries)
        {
            if (!boundary.runtime || (!boundary.graph_submitted &&
                (boundary.runtime->inference_boundary_receipt.submittedGeneration() != boundary.generation - 1u ||
                 boundary.runtime->inference_boundary_receipt.completedGeneration() != boundary.generation - 1u)))
            {
                if (error) *error = "Cancelled original expert admission lacks exact unlaunched rollback";
                return false;
            }
        }
        // Native proof permits teardown only. Aborted packets, policy state and
        // maintenance receipts are never reset or published as successful work.
        submission.native_terminals_joined = true;
        submission.original_source_cancellation = {};
        {
            std::lock_guard lock(submission.source_cancellation_failure_mutex);
            if (submission.source_cancellation_failure) {
                if (error) *error = enqueues.first_error;
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::observeOriginalNativePacketReceipts(
        MappedFollowerTransaction &submission, std::string *error)
    {
        try
        {
            submission.requireOriginalSourceCancellationSuccessful();
            if (!submission.original_slot || !submission.native_terminals_joined)
                throw std::logic_error("Original packet observation requires its aggregate native proof");
            for (auto &lane : submission.armed)
            {
                if (lane.original_completed_traffic)
                    continue;
                auto *mapped = dynamic_cast<IMoEOverlayMappedActivationTransport *>(
                    lane.authority->transport_lifetime.get());
                if (!mapped)
                    throw std::logic_error("Original native packet lost its mapped transport owner");
                lane.original_completed_traffic = mapped->completedOriginalNativeInvocation(
                    *submission.original_slot, lane.authority->participant_id, lane.identity);
            }
            return true;
        }
        catch (const std::exception &failure)
        {
            if (error) *error = failure.what();
            return false;
        }
    }

    bool MoEOverlayParticipantGraphRunner::clearOriginalNativePacketReceipts(
        MappedFollowerTransaction &submission, std::string *error)
    {
        try
        {
            submission.requireOriginalSourceCancellationSuccessful();
            if (submission.original_native_retirement)
                return true;
            if (!submission.original_slot || !submission.native_terminals_joined ||
                std::any_of(submission.armed.begin(), submission.armed.end(),
                    [](const auto &lane) { return !lane.original_completed_traffic; }))
                throw std::logic_error("Original packet clear lacks complete aggregate physical proof");
            const MoEOverlayOriginalNativeInvocationRetirement permit(*submission.original_slot);
            for (const auto &lane : submission.armed)
            {
                auto *mapped = dynamic_cast<IMoEOverlayMappedActivationTransport *>(
                    lane.authority->transport_lifetime.get());
                if (!mapped)
                    throw std::logic_error("Original native packet clear lost its mapped owner");
                mapped->clearOriginalNativeInvocation(permit, lane.authority->participant_id, lane.identity);
            }
            submission.original_native_retirement = permit;
            return true;
        }
        catch (const std::exception &failure)
        {
            if (error) *error = failure.what();
            return false;
        }
    }

    bool MoEOverlayParticipantGraphRunner::retireMappedFollowerTransaction(
        MappedFollowerTransaction &submission, std::string *error,
        MappedFollowerRetirementPhase retirement_phase)
    {
        using Clock = std::chrono::steady_clock;
        using ArmedLane = MappedFollowerTransaction::ArmedLane;
        using CPULayerTimelineSample = MappedFollowerTransaction::CPULayerTimelineSample;
        auto &cached = *submission.cached;
        auto &ticket = submission.ticket;
        auto &shape = submission.shape;
        auto &collect_timeline = submission.collect_timeline;
        auto &physical_rows_u64 = submission.physical_rows_u64;
        auto &cpu_dispatch_wait_ns = submission.cpu_dispatch_wait_ns;
        auto &cpu_packet_prepare_ns = submission.cpu_packet_prepare_ns;
        auto &cpu_expert_compute_ns = submission.cpu_expert_compute_ns;
        auto &cpu_return_publication_ns = submission.cpu_return_publication_ns;
        auto &endpoint_completion_wait_ns = submission.endpoint_completion_wait_ns;
        auto &cpu_layer_timeline = submission.cpu_layer_timeline;
        auto &armed = submission.armed;
        auto &submitted_boundaries = submission.submitted_boundaries;
        auto &prepared_gpu_submissions = submission.prepared_gpu_submissions;
        auto &submitted_gpu_terminals = submission.submitted_gpu_terminals;
        auto &captured_gpu_transactions = submission.captured_gpu_transactions;
        auto &replayed_gpu_transactions = submission.replayed_gpu_transactions;
        auto &cpu_layer_dispatches = submission.cpu_layer_dispatches;
        auto &mapped_dispatch_bytes = submission.mapped_dispatch_bytes;
        auto &mapped_return_bytes = submission.mapped_return_bytes;
        auto &mapped_dispatch_rows = submission.mapped_dispatch_rows;
        auto &mapped_return_rows = submission.mapped_return_rows;
        auto &mapped_dispatch_entries = submission.mapped_dispatch_entries;
        auto &transaction_phase = submission.transaction_phase;
        const auto elapsed_nanoseconds = [](
                                             Clock::time_point begin,
                                             Clock::time_point end)
            -> std::uint64_t
        {
            const auto elapsed =
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - begin)
                    .count();
            return elapsed > 0
                       ? static_cast<std::uint64_t>(elapsed)
                       : 0u;
        };
        const auto accumulate_elapsed = [&elapsed_nanoseconds](
                                            std::uint64_t &total,
                                            Clock::time_point begin,
                                            Clock::time_point end)
            -> std::uint64_t
        {
            const std::uint64_t elapsed = elapsed_nanoseconds(begin, end);
            if (elapsed != 0u &&
                total <= std::numeric_limits<std::uint64_t>::max() -
                             elapsed)
            {
                total += elapsed;
            }
            return elapsed;
        };
        if (error)
            error->clear();
        const auto reject = [error](std::string message)
        {
            if (error)
                *error = std::move(message);
            return false;
        };
        const auto abort_armed = [this, &submission]() noexcept
        {
            cancelMappedFollowerTransaction(submission);
        };

        /*
         * A source parent fans one sparse stage out to every follower before it
         * can advance.  Waiting endpoint events one at a time means the first
         * incomplete event is not necessarily the lane withholding a return:
         * it may be parked behind another participant at the source fan-in.
         * Capture one topology-wide snapshot for every timeout so the fatal
         * report identifies the actual protocol frontier without adding any
         * steady-path polling, transfer, or synchronization.
         */
        const auto topology_timeout_diagnostic =
            [this, shape, &ticket](
                const CachedParticipantGraph::MappedGPUFollowerEndpoint *
                    observed_endpoint)
        {
            /*
             * A timed-out resident wait can occupy the backend scheduler in a
             * way that also prevents a newly submitted diagnostic D2H copy from
             * making progress. Reading device protocol structs here therefore
             * risks hanging the fatal path after its bounded terminal timeout.
             * The activation epoch and controller fabrics are host-mapped by
             * design, so report their acquire-loaded state directly without
             * submitting GPU work or introducing a second synchronization
             * protocol solely for diagnostics.
             */
            std::ostringstream out;
            out << "ticket={" << ticket.toString() << "}"
                << " physical_rows=" << shape->physical_rows
                << " gpu_endpoint_count=" << shape->endpoints.size();
            if (config_.device_controller_fabric)
            {
                /*
                 * Continuation participants freeze one exact placement epoch
                 * through a node-local mapped barrier before they can arm the
                 * remote sparse lanes.  A follower terminal timeout is often
                 * the first host-visible symptom of a missing continuation
                 * arrival, so include the acquire-loaded barrier lanes once
                 * per topology snapshot.  This reads only the fabric's mapped
                 * diagnostic alias; it neither submits GPU work nor changes
                 * the inference/maintenance ordering under investigation.
                 */
                out << ' '
                    << config_.device_controller_fabric
                           ->describeInferenceEpochBarrier();
            }
            for (const auto &owned_endpoint : shape->endpoints)
            {
                const auto *const endpoint = owned_endpoint.get();
                out << " endpoint={";
                if (!endpoint)
                {
                    out << "null}";
                    continue;
                }
                out << "observed="
                    << (endpoint == observed_endpoint ? 1 : 0)
                    << ",participant=" << endpoint->participant_id
                    << ",device=" << endpoint->device.toString()
                    << ",cache_initialized="
                    << (endpoint->cache.initialized ? 1 : 0)
                    << ",submission_state="
                    << static_cast<std::uint32_t>(
                           endpoint->cache.executable_submission_state)
                    << ",replay_ready="
                    << (endpoint->cache.retained_full_graph_replay.valid()
                            ? 1
                            : 0);
                if (!endpoint->lane_authority ||
                    !endpoint->lane_authority->protocol)
                {
                    out << ",protocol=missing}";
                    continue;
                }

                const auto &authority = *endpoint->lane_authority;
                const auto &protocol = *authority.protocol;
                const auto identity = protocol.activeIdentity();
                const auto continuation = protocol.endpointStatus(
                    MoEOverlayActivationEndpoint::Continuation);
                const auto follower = protocol.endpointStatus(
                    MoEOverlayActivationEndpoint::Follower);
                out << ",next_generation="
                    << authority.next_epoch_generation
                    << ",active_generation=" << identity.epoch_generation
                    << ",active_step=" << identity.logical_step_id
                    << ",active_role=" << identity.graph_role
                    << ",active_rows="
                    << identity.physical_rows_per_request
                    << ",admission="
                    << static_cast<std::uint32_t>(
                           protocol.admissionState())
                    << ",continuation={state=" << continuation.state
                    << ",code=" << continuation.code
                    << ",operation=" << continuation.operation
                    << ",evidence=" << continuation.observed_timeline
                    << ",diagnostic="
                    << continuation.failure_diagnostic
                    << ",auxiliary="
                    << continuation.failure_auxiliary
                    << ",consumed="
                    << continuation.last_consumed_stage
                    << ",published="
                    << continuation.last_published_stage << "}"
                    << ",follower={state=" << follower.state
                    << ",code=" << follower.code
                    << ",operation=" << follower.operation
                    << ",evidence=" << follower.observed_timeline
                    << ",diagnostic=" << follower.failure_diagnostic
                    << ",auxiliary=" << follower.failure_auxiliary
                    << ",consumed=" << follower.last_consumed_stage
                    << ",published=" << follower.last_published_stage
                    << "}";
                for (std::uint32_t bank = 0u;
                     bank < kMoEOverlayActivationBufferCount;
                     ++bank)
                {
                    out << ",bank" << bank << "={dispatch="
                        << protocol.dispatchTimeline(bank)
                        << ",return="
                        << protocol.returnTimeline(bank) << "}";
                }

                const auto runtime_it =
                    participant_gpu_runtimes_.find(
                        endpoint->participant_id);
                if (runtime_it != participant_gpu_runtimes_.end() &&
                    runtime_it->second)
                {
                    const auto &receipt =
                        runtime_it->second->inference_boundary_receipt;
                    out << ",boundary_receipt={submitted="
                        << receipt.submittedGeneration()
                        << ",completed="
                        << receipt.completedGeneration()
                        << ",consumed="
                        << receipt.consumedGeneration() << "}";
                }
                const auto progress_it =
                    transfer_progress_epochs_.find(endpoint->device);
                if (progress_it != transfer_progress_epochs_.end() &&
                    progress_it->second)
                {
                    const auto progress = progress_it->second->stats();
                    out << ",transfer_progress={slots="
                        << progress.slots_reserved
                        << ",published=" << progress.commands_published
                        << ",completed=" << progress.commands_completed
                        << ",bytes=" << progress.bytes_completed
                        << ",copy_submissions="
                        << progress.copy_submissions
                        << ",idle_skips="
                        << progress.idle_submission_skips
                        << ",in_flight_observations="
                        << progress.in_flight_observations
                        << ",failures=" << progress.command_failures
                        << "}";
                }
                if (config_.device_controller_fabric)
                {
                    try
                    {
                        // Failure diagnostics use the fabric's explicit host
                        // view. Participant bindings contain GPU-only aliases;
                        // dereferencing them can crash while reporting the
                        // original failure when host and device VAs differ.
                        out << ',' << config_.device_controller_fabric
                            ->describeInferenceEpochBarrier();
                    }
                    catch (const std::exception &diagnostic_error)
                    {
                        out << ",controller=<unavailable:"
                            << diagnostic_error.what() << ">";
                    }
                }
                out << "}";
            }
            return out.str();
        };

        if (!submission.native_terminals_joined)
        {
            const auto enqueues = joinMappedFollowerEnqueues(submission);
            if (!enqueues.successful)
            {
                std::string proof_error;
                if (!drainCancelledMappedFollowerTransaction(submission, &proof_error))
                    return reject("Original expert asynchronous failure lacks native retirement: " + proof_error);
                return reject(enqueues.first_error);
            }

            std::string native_error;
            if (!joinAllOriginalNativeTerminals(
                    submitted_gpu_terminals,
                    [&](const auto &terminal)
                    {
                        if (!terminal.endpoint || !terminal.ticket.valid())
                            throw std::logic_error("Mapped follower lost an exact GPU terminal ticket");
                        terminal.endpoint->cache.waitForPublishedCaptureStreamTerminal(
                            terminal.ticket,
                            DeviceGraphExecutor::GraphSegmentCache::HostFenceWaitPolicy::ActiveProgress,
                            [&, observed_endpoint = terminal.endpoint]()
                            {
                                return topology_timeout_diagnostic(observed_endpoint);
                            });
                        participantGpuRuntimeForParticipant(terminal.endpoint->participant_id)
                            .retireOriginalInferencePredecessor(terminal.endpoint,terminal.ticket);
                    }, abort_armed, &native_error))
            {
                return reject("Mapped ExpertOverlay follower terminal event failed: " + native_error);
            }

            const auto submitted_count = static_cast<std::size_t>(std::count_if(
                submitted_boundaries.begin(), submitted_boundaries.end(),
                [](const auto &boundary) { return boundary.graph_submitted; }));
            if (submitted_count != submitted_gpu_terminals.size())
                return reject("Original expert parent launch has no exact published native terminal");
            for (const auto &boundary : submitted_boundaries)
            {
                if (!boundary.runtime)
                    return reject("Original expert inference admission lost its runtime owner");
                if (!boundary.graph_submitted &&
                    (boundary.runtime->inference_boundary_receipt.submittedGeneration() != boundary.generation - 1u ||
                     boundary.runtime->inference_boundary_receipt.completedGeneration() != boundary.generation - 1u))
                    return reject("Original unsubmitted expert admission has not been cancelled");
            }
            submission.native_terminals_joined = true;

        }
        if (retirement_phase == MappedFollowerRetirementPhase::NativeProofOnly)
            return true;
        if (submission.original_slot && !submission.original_native_retirement)
            return reject("Original native follower lacks complete aggregate packet-retirement proof");
        submission.requireOriginalSourceCancellationSuccessful();
        submission.original_source_cancellation = {};

        if (!submission.original_slot)
        {
            /* Endpoint completion is a final independent rendezvous after all
             * exact local work and GPU terminal events have retired. */
            const auto completion_rendezvous =
                MoEOverlayActivationRendezvousDeadline::begin(
                    MoEOverlayActivationRendezvousKind::EndpointCompletion,
                    std::chrono::milliseconds(
                        collective_timeout_policy::
                            kDefaultCollectiveTimeoutMs));
            const auto endpoint_completion_begin = collect_timeline
                                                       ? Clock::now()
                                                       : Clock::time_point{};
            while (completion_rendezvous.waitingAllowed())
            {
                bool complete = true;
                for (const auto &lane : armed)
                {
                    const auto follower_state =
                        lane.authority->protocol->endpointState(
                            MoEOverlayActivationEndpoint::Follower);
                    const auto continuation_state =
                        lane.authority->protocol->endpointState(
                            MoEOverlayActivationEndpoint::Continuation);
                    if (follower_state ==
                            MoEOverlayActivationEndpointState::Aborted ||
                        continuation_state ==
                            MoEOverlayActivationEndpointState::Aborted)
                    {
                        const auto follower_status =
                            lane.authority->protocol->endpointStatus(
                                MoEOverlayActivationEndpoint::Follower);
                        const auto continuation_status =
                            lane.authority->protocol->endpointStatus(
                                MoEOverlayActivationEndpoint::Continuation);
                        std::ostringstream diagnostic;
                        diagnostic
                            << "Mapped ExpertOverlay follower observed an aborted "
                               "device endpoint for participant "
                            << lane.authority->participant_id
                            << "; continuation={state="
                            << continuation_status.state << ",code="
                            << continuation_status.code << ",operation="
                            << continuation_status.operation << ",timeline="
                            << continuation_status.observed_timeline
                            << ",published_stage="
                            << continuation_status.last_published_stage
                            << ",consumed_stage="
                            << continuation_status.last_consumed_stage
                            << ",model_layer="
                            << continuation_status.last_model_layer
                            << "}; follower={state=" << follower_status.state
                            << ",code=" << follower_status.code
                            << ",operation=" << follower_status.operation
                            << ",timeline=" << follower_status.observed_timeline
                            << ",published_stage="
                            << follower_status.last_published_stage
                            << ",consumed_stage="
                            << follower_status.last_consumed_stage
                            << ",model_layer="
                            << follower_status.last_model_layer << "}";
                        try
                        {
                            auto *const mapped = dynamic_cast<
                                IMoEOverlayMappedActivationTransport *>(
                                lane.authority->transport_lifetime.get());
                            if (mapped && lane.authority->transport_lifetime &&
                                lane.authority->transport_lifetime
                                    ->hasSharedRowStorage())
                            {
                                const auto &control =
                                    mapped->activationEpochControl(
                                        lane.authority->participant_id,
                                        lane.authority->graph_family_ordinal);
                                const auto rows =
                                    lane.authority->transport_lifetime
                                        ->sharedDispatchRows(
                                            lane.authority->participant_id);
                                const auto &buffer = control.buffers[0];
                                diagnostic
                                    << "; shared_stage0={admission="
                                    << control.admission.ready_signal << "/"
                                    << control.admission.state
                                    << ",dispatch_signal="
                                    << buffer.dispatch_signal.value
                                    << ",return_signal="
                                    << buffer.return_signal.value
                                    << ",descriptor_timeline="
                                    << buffer.dispatch_descriptor.timeline
                                    << ",descriptor_rows="
                                    << buffer.dispatch_descriptor.live_rows
                                    << ",descriptor_entries="
                                    << buffer.dispatch_descriptor.live_entries
                                    << ",descriptor_bytes="
                                    << buffer.dispatch_descriptor.payload_bytes
                                    << ",descriptor_stage="
                                    << buffer.dispatch_descriptor.stage_ordinal
                                    << ",descriptor_layer="
                                    << buffer.dispatch_descriptor.model_layer_index;
                                if (rows.row_ids_host &&
                                    rows.entry_offsets_host)
                                {
                                    diagnostic
                                        << ",row0=" << rows.row_ids_host[0]
                                        << ",offset0="
                                        << rows.entry_offsets_host[0]
                                        << ",offset1="
                                        << rows.entry_offsets_host[1];
                                }
                                diagnostic << "}";
                            }
                        }
                        catch (const std::exception &exception)
                        {
                            diagnostic << "; shared_stage0=<unavailable:"
                                       << exception.what() << ">";
                        }
                        abort_armed();
                        return reject(diagnostic.str());
                    }
                    complete = complete &&
                               follower_state ==
                                   MoEOverlayActivationEndpointState::Complete &&
                               continuation_state ==
                                   MoEOverlayActivationEndpointState::Complete;
                }
                if (complete)
                    break;
#if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
#else
                std::this_thread::yield();
#endif
            }
            if (collect_timeline)
            {
                accumulate_elapsed(
                    endpoint_completion_wait_ns,
                    endpoint_completion_begin,
                    Clock::now());
            }

            for (const auto &lane : armed)
            {
                if (lane.authority->protocol->endpointState(
                        MoEOverlayActivationEndpoint::Follower) !=
                        MoEOverlayActivationEndpointState::Complete ||
                    lane.authority->protocol->endpointState(
                        MoEOverlayActivationEndpoint::Continuation) !=
                        MoEOverlayActivationEndpointState::Complete)
                {
                    std::string timeout_error;
                    const auto timeout_observation =
                        completion_rendezvous.timeoutObservationNanoseconds();
                    if (!timeout_observation)
                    {
                        return reject(
                            "Mapped ExpertOverlay follower reached an incomplete endpoint outside a valid timeout observation");
                    }
                    (void)lane.authority->protocol->markTimedOut(
                        lane.identity.epoch_generation,
                        *timeout_observation,
                        &timeout_error);
                    return reject(
                        "Mapped ExpertOverlay follower timed out waiting for both endpoint-complete publications: " +
                            timeout_error);
                }
            }

        }

        /*
         * Native original-slot traffic comes from the complete aggregate's
         * immutable post-payload receipts. Host-ordered traffic retains its
         * original acquired endpoint Complete proof. Materialize PerfStats
         * from those live extents, never allocation capacity or a reused lane.
         */
        const auto add_checked = [&](std::uint64_t &total,
                                     std::uint64_t value,
                                     const char *label) -> bool
        {
            if (total > std::numeric_limits<std::uint64_t>::max() - value)
            {
                if (error)
                    *error = std::string("Mapped ExpertOverlay ") + label +
                             " evidence overflowed uint64_t";
                return false;
            }
            total += value;
            return true;
        };
        const char *const service_source = [&ticket]() noexcept
        {
            switch (ticket.graph_role)
            {
            case MoEOverlayInferenceGraphRole::MainPrefill:
                return "prefill";
            case MoEOverlayInferenceGraphRole::MainDecode:
            case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
            case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
                return "decode";
            case MoEOverlayInferenceGraphRole::MTPDraft:
            case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
                return "grouped_verifier";
            case MoEOverlayInferenceGraphRole::None:
                return "invalid";
            }
            return "invalid";
        }();
        for (const auto &lane : armed)
        {
            std::string traffic_error;
            const auto traffic = submission.original_slot
                ? lane.original_completed_traffic
                : lane.authority->protocol->completedTraffic(lane.identity, &traffic_error);
            if (!traffic)
            {
                return reject(
                    "Mapped ExpertOverlay follower could not acquire completed traffic for participant " +
                    std::to_string(lane.authority->participant_id) + ": " +
                    traffic_error);
            }
            if (!add_checked(
                    mapped_dispatch_bytes,
                    traffic->dispatch_payload_bytes,
                    "dispatch-byte") ||
                !add_checked(
                    mapped_return_bytes,
                    traffic->return_payload_bytes,
                    "return-byte") ||
                !add_checked(
                    mapped_dispatch_rows,
                    traffic->dispatch_live_rows,
                    "dispatch-row") ||
                !add_checked(
                    mapped_return_rows,
                    traffic->return_live_rows,
                    "return-row") ||
                !add_checked(
                    mapped_dispatch_entries,
                    traffic->dispatch_live_entries,
                    "dispatch-entry"))
            {
                return false;
            }

            const auto &authority = *lane.authority;
            const char *const device_kind =
                authority.device.is_cpu()
                    ? "CPU"
                    : (authority.device.is_cuda() ? "CUDA" : "ROCm");
            const PerfStatsCollector::Tags traffic_tags{
                {"completion", "device_owned_epoch_complete"},
                {"device_kind", device_kind},
                {"identity_source", "device_owned_activation_epoch"},
                {"participant",
                 std::to_string(authority.participant_id)},
                {"stage_count",
                 std::to_string(traffic->dispatch_stage_count)},
                {"tier", std::to_string(authority.tier_index)},
                {"tier_priority",
                 std::to_string(authority.tier_priority)},
                {"transport", "compact"},
            };
            if (traffic->dispatch_payload_bytes > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "compact_dispatch_bytes",
                    static_cast<double>(traffic->dispatch_payload_bytes),
                    "gn_sparse_dispatch",
                    authority.device.to_string(),
                    traffic_tags);
            }
            if (traffic->dispatch_shared_hidden_bytes > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay", "shared_dispatch_hidden_bytes",
                    static_cast<double>(traffic->dispatch_shared_hidden_bytes),
                    "gn_sparse_dispatch", authority.device.to_string(), traffic_tags);
            }
            if (traffic->return_payload_bytes > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    "compact_return_bytes",
                    static_cast<double>(traffic->return_payload_bytes),
                    "gn_return_reduce",
                    authority.device.to_string(),
                    traffic_tags);
            }
            if (traffic->return_live_rows > 0u)
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay",
                    authority.device.is_cpu()
                        ? "device_epoch_cpu_rows"
                        : "device_epoch_gpu_rows",
                    static_cast<double>(traffic->return_live_rows),
                    "gn_local_expert",
                    authority.device.to_string(),
                    {{"completion", "device_owned_epoch_complete"},
                     {"domain_kind",
                      authority.device.is_cpu() ? "CPU" : "GPU"},
                     {"participant",
                      std::to_string(authority.participant_id)},
                     {"tier", std::to_string(authority.tier_index)},
                     {"transport", "local"}});
            }
            if (traffic->dispatch_live_entries > 0u)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "moe_overlay_local_expert_active_routes",
                    static_cast<double>(traffic->dispatch_live_entries),
                    "moe_overlay",
                    authority.device.to_string(),
                    {{"completion", "device_owned_epoch_complete"},
                     {"device_kind", device_kind},
                     {"identity_source",
                      "device_owned_activation_epoch"},
                     {"input_rows",
                      std::to_string(traffic->dispatch_live_rows)},
                     {"output_rows",
                      std::to_string(traffic->return_live_rows)},
                     {"participant",
                      std::to_string(authority.participant_id)},
                     {"service_source", service_source},
                     {"tier", std::to_string(authority.tier_index)}});
            }
        }

        if (captured_gpu_transactions > 0u)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "device_timeline_transaction_captures",
                static_cast<double>(captured_gpu_transactions),
                transaction_phase,
                participantDeviceList(local_participants_),
                {{"gpu_host_layer_dispatches", "0"},
                 {"physical_rows", std::to_string(physical_rows_u64)},
                 {"role", "expert_transaction_follower"}});
        }
        if (replayed_gpu_transactions > 0u)
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                "device_timeline_transaction_replays",
                static_cast<double>(replayed_gpu_transactions),
                transaction_phase,
                participantDeviceList(local_participants_),
                {{"gpu_host_layer_dispatches", "0"},
                 {"physical_rows", std::to_string(physical_rows_u64)},
                 {"role", "expert_transaction_follower"}});
        }

        if (!submission.original_slot)
        {
            for (const auto &lane : armed)
            {
                std::string reset_error;
                if (!lane.authority->protocol->reset(
                        lane.identity, &reset_error))
                {
                    return reject(
                        "Mapped ExpertOverlay follower could not retire participant " +
                            std::to_string(
                                lane.authority->participant_id) + ": " +
                            reset_error);
                }
            }
        }

        /*
         * Original native metadata clears only after ALL aggregate event and
         * post-payload receipts; host-ordered protocols retain their own reset.
         * Publish each participant receipt last, in original submission order,
         * so maintenance cannot mistake a submitted parent for completed work.
         */
        for (const auto &boundary : submitted_boundaries)
        {
            if (!boundary.runtime ||
                !boundary.runtime->inference_boundary_receipt
                     .completeSubmission(boundary.generation))
            {
                return reject(
                    "Mapped ExpertOverlay follower could not publish its serial inference-boundary receipt");
            }
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "mapped_device_owned_epochs",
            1.0,
            transaction_phase,
            participantDeviceList(local_participants_),
            {{"captured_gpu_transactions",
              std::to_string(captured_gpu_transactions)},
             {"dispatch_bytes", std::to_string(mapped_dispatch_bytes)},
             {"dispatch_entries",
              std::to_string(mapped_dispatch_entries)},
             {"dispatch_rows", std::to_string(mapped_dispatch_rows)},
             {"gpu_endpoints", std::to_string(shape->endpoints.size())},
             {"cpu_endpoints",
              std::to_string(cached.mapped_cpu_endpoints.size())},
             {"physical_rows", std::to_string(shape->physical_rows)},
             {"gpu_host_layer_dispatches", "0"},
             {"cpu_host_layer_dispatches",
              std::to_string(cpu_layer_dispatches)},
             {"replayed_gpu_transactions",
              std::to_string(replayed_gpu_transactions)},
             {"return_bytes", std::to_string(mapped_return_bytes)},
             {"return_rows", std::to_string(mapped_return_rows)},
             {"terminal_event_fences",
              std::to_string(shape->endpoints.size())}});
        if (collect_timeline && !cached.mapped_cpu_endpoints.empty())
        {
            const PerfStatsCollector::Tags timeline_tags{
                {"cpu_endpoints",
                 std::to_string(cached.mapped_cpu_endpoints.size())},
                {"cpu_layer_dispatches",
                 std::to_string(cpu_layer_dispatches)},
                {"graph_role",
                 std::to_string(
                     static_cast<std::uint32_t>(ticket.graph_role))},
                {"physical_rows", std::to_string(physical_rows_u64)},
                {"timing_semantics", "aggregate_nonoverlapping_cpu_phases"},
            };
            const auto record_timeline = [&](const char *name,
                                             std::uint64_t ns)
            {
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_participant_graph",
                    name,
                    std::max<std::uint64_t>(1u, ns),
                    transaction_phase,
                    participantDeviceList(local_participants_),
                    timeline_tags);
            };
            record_timeline(
                "mapped_cpu_dispatch_wait", cpu_dispatch_wait_ns);
            record_timeline(
                "mapped_cpu_packet_prepare", cpu_packet_prepare_ns);
            record_timeline(
                "mapped_cpu_expert_compute", cpu_expert_compute_ns);
            record_timeline(
                "mapped_cpu_return_publication",
                cpu_return_publication_ns);
            record_timeline(
                "mapped_endpoint_completion_wait",
                endpoint_completion_wait_ns);

            /* Emit fine-grained observations after every live endpoint and
             * terminal event has completed. The recorded values therefore
             * describe the unperturbed rendezvous rather than PerfStats work
             * interleaved with the inference transaction. */
            for (const auto &sample : cpu_layer_timeline)
            {
                const PerfStatsCollector::Tags layer_tags{
                    {"graph_role",
                     std::to_string(
                         static_cast<std::uint32_t>(ticket.graph_role))},
                    {"live_entries", std::to_string(sample.live_entries)},
                    {"live_rows", std::to_string(sample.live_rows)},
                    {"model_layer",
                     std::to_string(sample.model_layer_index)},
                    {"participant",
                     std::to_string(sample.participant_id)},
                    {"physical_rows", std::to_string(physical_rows_u64)},
                    {"stage_ordinal",
                     std::to_string(sample.stage_ordinal)},
                    {"timing_semantics",
                     "deferred_per_layer_nonoverlapping"},
                };
                const auto record_layer_timeline =
                    [&](const char *name, std::uint64_t ns)
                {
                    PerfStatsCollector::recordTimingNs(
                        "moe_overlay_participant_graph",
                        name,
                        std::max<std::uint64_t>(1u, ns),
                        transaction_phase,
                        "CPU",
                        layer_tags);
                };
                record_layer_timeline(
                    "mapped_cpu_layer_dispatch_wait",
                    sample.dispatch_wait_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_packet_prepare",
                    sample.packet_prepare_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_expert_compute",
                    sample.expert_compute_ns);
                record_layer_timeline(
                    "mapped_cpu_layer_return_publication",
                    sample.return_publication_ns);
                // Child intervals overlap the existing expert-compute parent;
                // post-service is additionally nested in local-execute. Publish
                // only here, after terminal completion, never inside a layer.
                auto child_tags = layer_tags;
                child_tags["timing_semantics"] = "deferred_child_of_expert_compute";
                for (const auto &[name, ns] : std::array{
                         std::pair{"mapped_cpu_layer_local_execute", sample.local_execute_ns},
                         std::pair{"mapped_cpu_layer_joint_publication", sample.joint_publication_ns}})
                    if (ns > 0u)
                        PerfStatsCollector::recordTimingNs("moe_overlay_participant_graph", name,
                            ns, transaction_phase, "CPU", child_tags);
                if(sample.joint_root) {
                    child_tags["timing_semantics"]="root_only_components_nested_in_joint_publication";
                    child_tags["gather_semantics"]="rooted_gather_includes_peer_wait";
                    for(const auto& [name,ns]:std::array{
                        std::pair{"mapped_cpu_layer_root_gather",sample.joint_root->gather_ns},
                        std::pair{"mapped_cpu_layer_root_validated_fixed_order_combine",sample.joint_root->combine_ns},
                        std::pair{"mapped_cpu_layer_root_preweight_publication",sample.joint_root->preweight_publication_ns},
                        std::pair{"mapped_cpu_layer_root_fused_combine_preweight_publication",sample.joint_root->fused_combine_preweight_publication_ns}})
                        if(ns)PerfStatsCollector::recordTimingNs("moe_overlay_participant_graph",name,*ns,
                            transaction_phase,"CPU",child_tags);
                    child_tags.erase("gather_semantics");
                }
                child_tags["timing_semantics"] = "deferred_postservice_nested_in_local_execute";
                // Zero means this optional child observation was absent (for
                // example an empty endpoint's early return), never a 1ns sample.
                if (sample.endpoint_post_service_ns > 0u)
                    PerfStatsCollector::recordTimingNs("moe_overlay_participant_graph",
                        "mapped_cpu_layer_endpoint_post_service", sample.endpoint_post_service_ns,
                        transaction_phase, "CPU", child_tags);
            }
        }
        return true;
    }

    /** @brief Original retained family selected from one authenticated immutable ticket. */
    struct MoEOverlayParticipantGraphRunner::RetainedGraphSelection
    {
        CachedParticipantGraph *graph = nullptr;
        SparseTransactionPhase phase = SparseTransactionPhase::Decode;
        int mtp_graph_depth = -1;
        const char *role_name = "invalid";
        std::uint64_t logical_rows = 0u;
        std::uint64_t physical_rows = 0u;
    };

    /**
     * @brief Same original ticket authority with process-local native endpoints.
     *
     * The enclosing Rank owns this adapter and the original runner. Publication
     * submits every follower before the continuation can start; retirement is
     * called only after that continuation has consumed its sparse returns.
     * Each original protocol slot retains its cold submission record across partial failure.
     */
    class MoEOverlayParticipantGraphRunner::RankLocalInferencePublisher final
        : public IMoEOverlayInferenceTransactionPublisher
    {
    public:
        /** @brief Retain the original runner and one reusable host submission record per admitted protocol slot. */
        explicit RankLocalInferencePublisher(std::shared_ptr<MoEOverlayParticipantGraphRunner> original)
            : original_(std::move(original)), state_(original_->rankLocalPublicationState())
        {
            if (original_->config_.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary ||
                !original_->main_graph_ || original_->admitted_rank_local_endpoint_ids_.empty() ||
                original_->serving_graph_family_lifecycle_ != ServingGraphFamilyLifecycle::Sealed ||
                topologyIdentity().source_world_rank != topologyIdentity().target_world_rank)
                throw std::invalid_argument("Local expert publisher requires its original prepared auxiliary endpoints");
            // Retain one host submission record per already admitted protocol
            // slot. Every graph family, native event and GPU allocation is still
            // owned once by its original endpoint cache; these are borrowed handles.
            submissions_.resize(state_.slotCount());
            const auto count = original_->local_participants_.size();
            for (auto &slot : submissions_)
            {
                slot.submission = std::make_unique<MappedFollowerTransaction>(
                    *original_->main_graph_, MoEOverlayInferenceTransactionTicket{});
                slot.submission->armed.reserve(count);
                slot.submission->submitted_boundaries.reserve(count);
                slot.submission->prepared_gpu_submissions.reserve(count);
                slot.submission->submitted_gpu_terminals.reserve(count);
                slot.submission->pending_gpu_enqueues.reserve(count);
            }
        }
        /** @brief Abandonment still owns every partial launch until exact native proof. */
        ~RankLocalInferencePublisher() override
        {
            if (!state_.active())
                return;
            std::string ignored;
            (void)abort(state_.activeTicket()->placement_epoch, 1, &ignored);
            if (state_.active())
                std::terminate();
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        bool beginCommand(const MoEOverlayInferenceCommandIdentity &command, std::string *error) override
        {
            return state_.begin(command, error);
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        MoEOverlayPublishedInferenceTransaction publish(
            const MoEOverlayInferenceExecutionDescriptor &descriptor) override
        {
            auto result = state_.publishOriginalFollowers(descriptor, [&](const auto &ticket, std::string *error)
            {
                auto &slot = slotFor(ticket);
                slot.submission->reset(*original_->main_graph_, ticket);
                if (original_->original_native_slot_geometry_)
                    slot.submission->original_slot = state_.originalSubmittedSlot(ticket);
                slot.submission->original_source_cancellation = original_->original_pooled_source_cancellation_;
                const auto selection = original_->selectRetainedGraph(ticket, error);
                if (!selection)
                    return false;
                slot.submission->cached = selection->graph;
                slot.selection = *selection;
                return original_->submitMappedFollowerTransaction(*slot.submission,error);
            }, [&](std::string* error) {
                // This consumes only native SDK enqueue futures. The captured
                // expert/storage bodies remain asynchronous; actual GPU/event
                // terminal proof belongs to retire(), never publication.
                const auto *active = state_.activeTicket();
                if (!active)
                    throw std::logic_error("Original local enqueue lost its protocol slot");
                auto &slot = slotFor(*active);
                const auto enqueues = original_->joinMappedFollowerEnqueues(*slot.submission);
                if (!enqueues.successful || enqueues.missing_terminal)
                {
                    if (error)
                        *error = enqueues.first_error.empty()
                            ? "Original expert host enqueue lacks its exact terminal publication"
                            : enqueues.first_error;
                    return false;
                }
                return true;
            });
            if (!result.ok && state_.active())
                cancelOriginalLanes();
            return result;
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        bool retire(const MoEOverlayPublishedInferenceTransaction &transaction, std::string *error) override
        {
            const bool retired = state_.retire(transaction, [&](std::string *diagnostic)
            {
                if (slotFor(transaction.ticket).submission->original_slot)
                {
                    if (!state_.observeOriginalActive([&](const auto &handle, std::string *detail)
                        {
                            return original_->retireMappedFollowerTransaction(
                                *slotFor(handle.ticket).submission, detail,
                                MappedFollowerRetirementPhase::NativeProofOnly);
                        }, diagnostic) ||
                        !state_.observeOriginalActive([&](const auto &handle, std::string *detail)
                        {
                            return original_->observeOriginalNativePacketReceipts(
                                *slotFor(handle.ticket).submission, detail);
                        }, diagnostic) ||
                        !state_.observeOriginalActive([&](const auto &handle, std::string *detail)
                        {
                            return original_->clearOriginalNativePacketReceipts(
                                *slotFor(handle.ticket).submission, detail);
                        }, diagnostic))
                        return false;
                }
                return original_->retireMappedFollowerTransaction(*slotFor(transaction.ticket).submission, diagnostic);
            }, error);
            if (retired)
            {
                auto &slot = submissions_.at(transaction.slot_index);
                if (!slot.selection)
                    throw std::logic_error("Retired local expert transaction lost its original family selection");
                original_->recordSuccessfulRetainedTransaction(transaction.ticket, *slot.selection);
                slot.selection.reset();
            }
            return retired;
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        bool complete(std::uint64_t epoch, std::uint64_t progress, std::string *error) override
        {
            return state_.complete(epoch, progress, error);
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        bool abort(std::uint64_t epoch, int code, std::string *error) override
        {
            return state_.abort(epoch, code, [&](std::string *diagnostic)
            {
                cancelOriginalLanes();
                // Existing receipts retire in original transaction order, even
                // when the protocol ring wraps. Attempt every physical drain
                // and preserve its first failure before releasing any handles.
                bool complete = true;
                std::string first_error;
                std::uint64_t previous = 0u;
                for (std::size_t visited = 0; visited < submissions_.size(); ++visited)
                {
                    const MoEOverlayPublishedInferenceTransaction *next = nullptr;
                    for (std::size_t index = 0; index < submissions_.size(); ++index)
                    {
                        const auto *handle = state_.activeHandle(index);
                        if (handle && handle->ticket.transaction_ordinal > previous &&
                            (!next || handle->ticket.transaction_ordinal < next->ticket.transaction_ordinal))
                            next = handle;
                    }
                    if (!next)
                        break;
                    previous = next->ticket.transaction_ordinal;
                    auto &submission = *submissions_[next->slot_index].submission;
                    std::string current;
                    try
                    {
                        if (!drainOriginalSubmissionCancellation(submission.native_terminals_joined,
                                submission.source_cancellation_failure_mutex, submission.source_cancellation_failure,
                                [&](std::string *diagnostic)
                                {
                                    return original_->drainCancelledMappedFollowerTransaction(submission, diagnostic);
                                }, &current))
                        {
                            complete = false;
                            if (first_error.empty())
                                first_error = current;
                        }
                    }
                    catch (const std::exception &failure)
                    {
                        complete = false;
                        if (first_error.empty())
                            first_error = failure.what();
                    }
                    catch (...)
                    {
                        complete = false;
                        if (first_error.empty())
                            first_error = "Nonstandard original native slot drain failure";
                    }
                }
                if (!complete && diagnostic)
                    *diagnostic = first_error;
                return complete;
            }, error);
        }
        /** @inheritdoc IMoEOverlayInferenceTransactionPublisher */
        const MoEOverlayInferenceTopologyIdentity &topologyIdentity() const noexcept override
        {
            return state_.topology();
        }
    private:
        /** @brief Signal exact original lanes; existing endpoint errors remain absorbing. */
        void cancelOriginalLanes() noexcept
        {
            for (std::size_t index = 0; index < submissions_.size(); ++index)
                if (state_.activeHandle(index))
                    original_->cancelMappedFollowerTransaction(*submissions_[index].submission);
        }
        std::shared_ptr<MoEOverlayParticipantGraphRunner> original_;
        RankLocalPublicationState &state_;
        /** @brief One cold host record borrows native resources for one original protocol slot. */
        struct SubmissionSlot {
            std::unique_ptr<MappedFollowerTransaction> submission;
            std::optional<RetainedGraphSelection> selection;
        };
        /**
         * @param ticket Exact original active protocol identity.
         * @return Its retained host record borrowing the original native resources.
         * @throws std::logic_error When no original protocol slot owns the ticket.
         */
        SubmissionSlot &slotFor(const MoEOverlayInferenceTransactionTicket &ticket)
        {
            const auto index = state_.activeSlot(ticket);
            if (!index)
                throw std::logic_error("Local expert native submission lost its exact original protocol slot");
            return submissions_.at(*index);
        }
        std::vector<SubmissionSlot> submissions_;
    };

    MoEOverlayParticipantGraphRunner::RankLocalPublicationState &
    MoEOverlayParticipantGraphRunner::rankLocalPublicationState() const
    {
        if (!rank_local_publication_state_ ||
            config_.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
            throw std::logic_error("Local expert publisher lost its original cold protocol domain");
        return *rank_local_publication_state_;
    }

    std::shared_ptr<IMoEOverlayInferenceTransactionPublisher>
    MoEOverlayParticipantGraphRunner::createRankLocalInferencePublisher(
        std::shared_ptr<MoEOverlayParticipantGraphRunner> original)
    {
        if (!original)
            throw std::invalid_argument("Local expert publisher requires its original runner owner");
        auto publisher = std::make_shared<RankLocalInferencePublisher>(original);
        original->serving_graph_family_lifecycle_ = ServingGraphFamilyLifecycle::ExecutionAdmitted;
        return publisher;
    }

    std::optional<MoEOverlayParticipantGraphRunner::RetainedGraphSelection>
    MoEOverlayParticipantGraphRunner::selectRetainedGraph(
        const MoEOverlayInferenceTransactionTicket &ticket, std::string *error)
    {
        const auto fail = [error](const char *message)
            -> std::optional<RetainedGraphSelection>
        {
            if (error)
                *error = message;
            return std::nullopt;
        };
        if (!ticket.valid() ||
            ticket.action != MoEOverlayInferenceTransactionAction::Execute)
        {
            return fail(
                "executor requires one valid retained-graph execution ticket");
        }
        if (ticket.target_world_rank != config_.mpi_context->rank() ||
            ticket.source_world_rank !=
                execution_plan_->continuation_root_rank)
        {
            return fail(
                "ticket source/target ranks do not match this participant graph");
        }

        const uint64_t request_count =
            static_cast<uint64_t>(ticket.request_count);
        const uint64_t logical_rows =
            request_count *
            static_cast<uint64_t>(ticket.logical_rows_per_request);
        const uint64_t physical_rows =
            request_count *
            static_cast<uint64_t>(ticket.physical_rows_per_request);
        if (logical_rows == 0 || logical_rows > physical_rows ||
            physical_rows > static_cast<uint64_t>(
                                config_.max_graph_activation_rows))
        {
            return fail(
                "ticket row geometry exceeds the setup-owned participant family");
        }

        CachedParticipantGraph *cached = nullptr;
        SparseTransactionPhase phase = SparseTransactionPhase::Decode;
        int mtp_graph_depth = -1;
        const char *role_name = "invalid";
        switch (ticket.graph_role)
        {
        case MoEOverlayInferenceGraphRole::MainPrefill:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::Prefill;
            role_name = "main_prefill";
            break;
        case MoEOverlayInferenceGraphRole::MainDecode:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::Decode;
            role_name = "main_decode";
            break;
        case MoEOverlayInferenceGraphRole::MTPGroupedVerifier:
            cached = main_graph_.get();
            phase = SparseTransactionPhase::GroupedVerifier;
            mtp_graph_depth = ticket.draft_depth;
            role_name = "mtp_grouped_verifier";
            break;
        case MoEOverlayInferenceGraphRole::MTPDraft:
            if (mtp_sidecar_graphs_.size() != 1u)
            {
                return fail(
                    "recurrent MTP tickets require exactly one learned retained sidecar graph");
            }
            cached = mtp_sidecar_graphs_.front().get();
            phase = SparseTransactionPhase::MTPDraft;
            mtp_graph_depth = cached ? cached->mtp_graph_depth : -1;
            role_name = "mtp_draft";
            break;
        case MoEOverlayInferenceGraphRole::MTPConditionSpeculativeContinuation:
        case MoEOverlayInferenceGraphRole::MTPConditionCommittedSerialToken:
            return fail("Pipeline condition tickets require their original stage-cohort executor");
        case MoEOverlayInferenceGraphRole::None:
            return fail("terminal graph role reached the execution authority");
        }

        if (!cached || cached->admitted_empty)
            return fail("Ticket has no admitted expert packet program");
        return RetainedGraphSelection{
            .graph = cached, .phase = phase, .mtp_graph_depth = mtp_graph_depth,
            .role_name = role_name, .logical_rows = logical_rows,
            .physical_rows = physical_rows,
        };
    }

    /** @brief Mirror only successfully retired original family/ticket evidence, independent of owner role. */
    void MoEOverlayParticipantGraphRunner::recordSuccessfulRetainedTransaction(
        const MoEOverlayInferenceTransactionTicket &ticket, const RetainedGraphSelection &selection)
    {
        const auto phase = selection.phase;
        const auto mtp_graph_depth = selection.mtp_graph_depth;
        const auto *role_name = selection.role_name;
        const auto logical_rows = selection.logical_rows;
        const auto physical_rows = selection.physical_rows;
        auto *const cached = selection.graph;
        const bool mapped_followers = cached->usesMappedActivationEpochs();
        const char *const evidence_phase =
            phase == SparseTransactionPhase::Prefill
                ? "prefill"
                : (phase == SparseTransactionPhase::Decode
                       ? "decode"
                       : role_name);
        PerfStatsCollector::recordOrderedSequenceStep(
            "forward_graph",
            "moe_overlay_collective_transaction_sequence",
            {ticket.request_generation,
             ticket.logical_step_id,
             logical_rows,
             physical_rows},
            evidence_phase,
            participantDeviceList(local_participants_),
            {{"role", "expert_participant_graph"},
             {"identity_source", "orchestration_request_and_chunk"},
             {"logical_step_semantics", "monotonic_transaction"}});
        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "ticket_selected_graphs",
            1.0,
            role_name,
            participantDeviceList(local_participants_),
            {{"logical_step_semantics", "monotonic_transaction"},
             {"logical_rows", std::to_string(logical_rows)},
             {"physical_rows", std::to_string(physical_rows)},
             {"draft_depth", std::to_string(ticket.draft_depth)},
             {"sidecar_ordinal", std::to_string(ticket.sidecar_depth)},
             {"mtp_graph_depth", std::to_string(mtp_graph_depth)},
             {"parallel_gpu_waves",
              mapped_followers
                  ? "device_owned_epoch"
                  : std::to_string(
                        cached->execution_plan.concurrent_gpu_wave_count)},
             {"max_wave_width",
              mapped_followers
                  ? std::to_string(
                        cached->mappedShapeForRows(
                            static_cast<int>(physical_rows))
                            ->endpoints.size())
                  : std::to_string(
                        cached->execution_plan.max_wave_width)},
             {"transport_path",
              mapped_followers
                  ? (cached->mapped_cpu_endpoints.empty()
                         ? "node_local_mapped_device_epoch"
                         : "node_local_mapped_gpu_epoch_cpu_boundary")
                  : "host_scheduled_rank_batch"},
             {"mapped_gpu_endpoints",
              mapped_followers
                  ? std::to_string(
                        cached->mappedShapeForRows(
                            static_cast<int>(physical_rows))
                            ->endpoints.size())
                  : "0"},
             {"mapped_cpu_endpoints",
              mapped_followers
                  ? std::to_string(cached->mapped_cpu_endpoints.size())
                  : "0"},
             {"gpu_host_layer_dispatches", "0"},
             {"position_mutated", "false"}});
    }

    /**
     * @brief Select and execute one retained graph from an authenticated ticket.
     *
     * Qwen's one learned NextN block is recurrent across speculative positions:
     * ticket `sidecar_depth` names that position, not another set of weights.
     * The retained graph therefore stays at manifest graph depth zero while
     * the transaction sequence carries positions 0..draft_depth-1. A future
     * model with several learned sidecar blocks must extend the ticket ABI with
     * a distinct graph-depth field instead of guessing from that ordinal.
     */
    bool MoEOverlayParticipantGraphRunner::
        executeMoEOverlayInferenceTransaction(
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error)
    {
        using Clock = std::chrono::steady_clock;
        const bool timing_enabled =
            PerfStatsCollector::isDomainEnabled(
                "moe_overlay_participant_graph");
        const auto selection_begin =
            timing_enabled ? Clock::now() : Clock::time_point{};
        if (error)
            error->clear();
        std::optional<MoEOverlayInferenceInterferenceScope>
            interference_scope;
        const auto fail = [&](std::string diagnostic)
        {
            if (interference_scope)
                interference_scope->discard();
            if (error)
                *error = diagnostic;
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transaction execution rejected: "
                << diagnostic << " (" << ticket.toString() << ")");
            return false;
        };

        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
            return fail("Rank-local expert endpoints require their original continuation coordinator");

        const auto selection = selectRetainedGraph(ticket, error);
        if (!selection)
            return fail(error ? *error : "Retained graph selection failed");
        auto *cached = selection->graph;
        const auto phase = selection->phase;
        const int mtp_graph_depth = selection->mtp_graph_depth;
        const char *role_name = selection->role_name;
        const std::uint64_t logical_rows = selection->logical_rows;
        const std::uint64_t physical_rows = selection->physical_rows;

        ExpertHistogramSource calibration_source =
            ExpertHistogramSource::SyntheticTest;
        int calibration_depth = 0;
        if (ticket.graph_role == MoEOverlayInferenceGraphRole::MainPrefill)
            calibration_source = ExpertHistogramSource::PrefillChunk;
        else if (ticket.graph_role == MoEOverlayInferenceGraphRole::MainDecode)
            calibration_source = ExpertHistogramSource::DecodeToken;
        else if (ticket.graph_role ==
                 MoEOverlayInferenceGraphRole::MTPGroupedVerifier)
        {
            calibration_source = ExpertHistogramSource::GroupedVerifier;
            calibration_depth = ticket.draft_depth;
        }
        if (calibration_source != ExpertHistogramSource::SyntheticTest)
        {
            interference_scope.emplace(
                interference_probe_.get(),
                makeMoEOverlayInferenceWorkloadIdentity(
                    calibration_source,
                    static_cast<int>(logical_rows),
                    static_cast<int>(physical_rows),
                    /*transaction_count=*/1,
                    calibration_depth));
            if (interference_scope->active())
            {
                LOG_DEBUG(
                    "[ExpertOverlay][Calibration] Follower rank "
                    << config_.mpi_context->rank()
                    << " claimed baseline/concurrent sample logical_step="
                    << ticket.logical_step_id << " rows=" << logical_rows
                    << '/' << physical_rows);
            }
        }

        if (!cached || cached->row_capacity <= 0 ||
            (!cached->graph && !cached->usesMappedActivationEpochs()) ||
            physical_rows > static_cast<uint64_t>(cached->row_capacity))
        {
            return fail(
                "selected retained graph is absent or smaller than ticket geometry");
        }
        const auto selection_end =
            timing_enabled ? Clock::now() : Clock::time_point{};
        const bool mapped_followers = cached->usesMappedActivationEpochs();
        if (!mapped_followers &&
            !stampMoEOverlayCollectiveRuntime(
                *cached,
                ticket.request_generation,
                ticket.logical_step_id,
                phase,
                mtp_graph_depth,
                ticket.placement_epoch))
        {
            return fail(
                "selected retained graph rejected its runtime wire namespace");
        }
        const auto stamp_end =
            timing_enabled ? Clock::now() : Clock::time_point{};
        std::string execution_error;
        const bool executed = mapped_followers
                                  ? executeMappedFollowerTransaction(
                                        *cached,
                                        ticket,
                                        &execution_error)
                                  : executor_.executeRetainedMultiDevice(
                                        cached->execution_plan,
                                        &execution_error);
        if (!executed)
        {
            std::string diagnostic =
                "selected retained participant graph execution failed: " +
                execution_error;
            if (config_.device_controller_fabric)
            {
                /*
                 * The continuation rank owns the mapped LocalTP epoch barrier.
                 * If its retained parent stalls before publishing layer zero,
                 * expose each device arrival lane at this terminal boundary;
                 * the remote follower cannot inspect this rank-local fabric.
                 */
                diagnostic += ' ';
                diagnostic += config_.device_controller_fabric
                                  ->describeInferenceEpochBarrier();
            }
            return fail(std::move(diagnostic));
        }
        if (interference_scope && interference_scope->active())
        {
            LOG_DEBUG(
                "[ExpertOverlay][Calibration] Follower rank "
                << config_.mpi_context->rank()
                << " reached exact retained-graph terminal logical_step="
                << ticket.logical_step_id << " rows=" << logical_rows
                << '/' << physical_rows);
        }
        const auto execution_end =
            timing_enabled ? Clock::now() : Clock::time_point{};

        if (timing_enabled)
        {
            const PerfStatsCollector::Tags tags{
                {"logical_rows", std::to_string(logical_rows)},
                {"physical_rows", std::to_string(physical_rows)},
                {"draft_depth", std::to_string(ticket.draft_depth)},
                {"sidecar_ordinal", std::to_string(ticket.sidecar_depth)},
                {"parallel_gpu_waves",
                 mapped_followers
                     ? "device_owned_epoch"
                     : std::to_string(
                           cached->execution_plan.concurrent_gpu_wave_count)},
                {"max_wave_width",
                 mapped_followers
                     ? std::to_string(
                           cached->mappedShapeForRows(
                               static_cast<int>(physical_rows))
                               ->endpoints.size())
                     : std::to_string(
                           cached->execution_plan.max_wave_width)},
                {"mapped_gpu_endpoints",
                 mapped_followers
                     ? std::to_string(
                           cached->mappedShapeForRows(
                               static_cast<int>(physical_rows))
                               ->endpoints.size())
                     : "0"},
                {"mapped_cpu_endpoints",
                 mapped_followers
                     ? std::to_string(
                           cached->mapped_cpu_endpoints.size())
                     : "0"},
                {"gpu_host_layer_dispatches", "0"},
            };
            const auto record = [&](const char* name, auto begin, auto end)
            {
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end - begin)
                        .count();
                PerfStatsCollector::recordTimingNs(
                    "moe_overlay_participant_graph",
                    name,
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, elapsed)),
                    role_name,
                    participantDeviceList(local_participants_),
                    tags);
            };
            record(
                "ticket_selection_and_validation",
                selection_begin,
                selection_end);
            record(
                mapped_followers
                    ? "mapped_epoch_selection"
                    : "runtime_sparse_stage_stamping",
                selection_end,
                stamp_end);
            record(
                mapped_followers
                    ? "mapped_device_owned_epoch_execution"
                    : "multi_device_graph_execution",
                stamp_end,
                execution_end);
        }

        recordSuccessfulRetainedTransaction(ticket, *selection);
        return true;
    }

    /**
     * @brief Validate that a bounded shared schedule can fit this remote graph.
     *
     * The root passes the exact schedule into @ref forwardPrefillChunkSchedule.
     * This capability check rejects requests outside the full request context
     * or an unplanned compact-buffer family before an MPI collective can begin.
     */
    bool MoEOverlayParticipantGraphRunner::supportsPrefillChunkSchedule(
        int seq_len) const
    {
        return seq_len > 1 && config_.max_seq_len > 0 &&
               seq_len <= config_.max_seq_len &&
               config_.max_graph_activation_rows > 0;
    }

    /**
     * @brief Execute the continuation root's schedule with live sparse shapes.
     *
     * Padded continuation buckets are a GPU capture concern.  Sparse dispatch
     * packets carry only real rows, so this endpoint must build/replay its
     * graph at @c chunk.real_count rather than padding expert traffic or
     * changing the collective record order.  The root-published contract has
     * already limited bucket capacity to every endpoint's planner admission;
     * checks here keep a corrupted or stale caller from violating that promise.
     */
    bool MoEOverlayParticipantGraphRunner::forwardPrefillChunkSchedule(
        const int *tokens,
        int seq_len,
        const PrefillChunkSchedulerPolicy &policy,
        int pad_token_id,
        bool allow_padded_execution)
    {
        (void)pad_token_id;
        (void)allow_padded_execution;
        if (!tokens || !supportsPrefillChunkSchedule(seq_len) ||
            policy.real_token_count != seq_len)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Invalid shared prefill "
                "chunk schedule input");
            return false;
        }

        const PrefillChunkSchedule schedule = planPrefillChunkSchedule(policy);
        if (!schedule || schedule.chunks.empty())
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Could not plan shared "
                "prefill chunks: " << schedule.error);
            return false;
        }
        if (position_ != policy.real_token_start)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Shared prefill starts "
                "at a cursor not owned by this participant runner "
                "schedule_start=" << policy.real_token_start
                << " local_position=" << position_);
            return false;
        }

        uint64_t executed_real_rows = 0;
        for (const PrefillChunkPlan &chunk : schedule.chunks)
        {
            const int relative_offset =
                chunk.token_offset - policy.real_token_start;
            if (chunk.real_count <= 0 || relative_offset < 0 ||
                relative_offset > seq_len ||
                chunk.real_count > seq_len - relative_offset ||
                chunk.bucket_seq_len <= 0 ||
                chunk.bucket_seq_len > config_.max_graph_activation_rows ||
                chunk.real_count > config_.max_graph_activation_rows)
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Shared prefill chunk "
                    "exceeds this participant's immutable graph capacity"
                    << " chunk=" << chunk.chunk_index
                    << " real_rows=" << chunk.real_count
                    << " bucket_rows=" << chunk.bucket_seq_len
                    << " capacity=" << config_.max_graph_activation_rows);
                return false;
            }

            if (!executeAtLogicalStep(
                    tokens + relative_offset,
                    chunk.real_count,
                    chunk.token_offset,
                    SparseTransactionPhase::Prefill,
                    chunk.bucket_seq_len))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Shared prefill chunk "
                    << chunk.chunk_index << " failed");
                return false;
            }
            executed_real_rows += static_cast<uint64_t>(chunk.real_count);
        }

        if (executed_real_rows != static_cast<uint64_t>(seq_len))
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Shared prefill chunks did "
                "not cover the requested real-token range");
            return false;
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_participant_prefill_chunk_schedules",
            static_cast<double>(schedule.chunks.size()),
            "prefill",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"logical_rows", std::to_string(seq_len)},
                {"bucket_rows", std::to_string(schedule.chunks.front().bucket_seq_len)},
                {"capacity", std::to_string(config_.max_graph_activation_rows)},
                {"live_rows_only", "true"},
            });
        return true;
    }

    const float *MoEOverlayParticipantGraphRunner::logits() const
    {
        return nullptr;
    }

    int MoEOverlayParticipantGraphRunner::vocab_size() const
    {
        return 0;
    }

    void MoEOverlayParticipantGraphRunner::clear_cache()
    {
        position_ = 0;
    }

    bool MoEOverlayParticipantGraphRunner::purgePrefixCache()
    {
        return true;
    }

    bool MoEOverlayParticipantGraphRunner::admitSparseQueryExecutionProgram(const DeviceGenerationAdmissionRequest &request)
    {
        // This participant owns expert-only projections, never an attention/indexer graph.
        // Validate the complete admission even though no query program is consumed here.
        if (!request.valid()) return false;
        (void)resolveSparseQueryExecutionProgram(request);
        return true;
    }

    void MoEOverlayParticipantGraphRunner::endSparseQueryExecutionProgram() noexcept
    {
        // Expert-only participants retain no sparse-query topology or response ledger.
    }

    int MoEOverlayParticipantGraphRunner::get_position() const
    {
        return position_;
    }

    ExecutionPath MoEOverlayParticipantGraphRunner::executionPath() const
    {
        return ExecutionPath::GRAPH;
    }

    const char *MoEOverlayParticipantGraphRunner::architecture() const
    {
        return architecture_.c_str();
    }

    std::vector<int>
    MoEOverlayParticipantGraphRunner::participantIds() const
    {
        std::vector<int> ids;
        ids.reserve(local_participants_.size());
        for (const auto *participant : local_participants_)
        {
            if (participant)
                ids.push_back(participant->participant_id);
        }
        return ids;
    }

    std::vector<DeviceId>
    MoEOverlayParticipantGraphRunner::localDevices() const
    {
        return participantDevices(local_participants_);
    }

    size_t MoEOverlayParticipantGraphRunner::cachedGraphCount() const noexcept
    {
        return (main_graph_ ? 1u : 0u) + mtp_sidecar_graphs_.size();
    }

    std::vector<MoEOverlayDeviceControllerRuntimeBinding>
    MoEOverlayParticipantGraphRunner::
        moeOverlayDeviceControllerRuntimeBindings() const
    {
        std::vector<int> participant_ids;
        participant_ids.reserve(participant_gpu_runtimes_.size());
        for (const auto &[participant_id, runtime] :
             participant_gpu_runtimes_)
        {
            if (!runtime)
            {
                throw std::logic_error(
                    "Mapped follower retained a null device-controller runtime");
            }
            participant_ids.push_back(participant_id);
        }
        std::sort(participant_ids.begin(), participant_ids.end());

        std::vector<MoEOverlayDeviceControllerRuntimeBinding> bindings;
        bindings.reserve(participant_ids.size());
        for (const int participant_id : participant_ids)
        {
            const auto &runtime =
                participantGpuRuntimeForParticipant(participant_id);
            if (!runtime.runtime_table || !runtime.epoch_arena ||
                runtime.runtime_table->layerCount() <= 0 ||
                runtime.initialized_layers.size() !=
                    static_cast<std::size_t>(
                        runtime.runtime_table->layerCount()) ||
                std::any_of(
                    runtime.initialized_layers.begin(),
                    runtime.initialized_layers.end(),
                    [](std::uint8_t initialized)
                    {
                        return initialized == 0u;
                    }))
            {
                throw std::logic_error(
                    "Mapped follower controller binding was requested before every runtime layer became durable");
            }

            MoEOverlayDeviceControllerRuntimeBinding binding{
                .device = runtime.device,
                .runtime_layers_device =
                    runtime.runtime_table->deviceLayerState(runtime.runtime_table->firstModelLayer()),
                .runtime_table_host = runtime.runtime_table.get(),
                .service_telemetry_device =
                    runtime.runtime_table
                        ->deviceOverlayServiceTelemetry(),
                .service_samples_device =
                    runtime.runtime_table
                        ->deviceOverlayServiceTelemetrySample(runtime.runtime_table->firstModelLayer()),
                .overlay_participant_id = runtime.participant_id,
                .domain_participant_id =
                    runtime.domain_participant_id,
                .domain_participant_count =
                    runtime.domain_participant_count,
                .layer_count = static_cast<std::uint32_t>(
                    runtime.runtime_table->layerCount()),
                .expert_count = static_cast<std::uint32_t>(
                    runtime.runtime_table->expertCount()),
                .top_k = static_cast<std::uint32_t>(
                    runtime.runtime_table->topK()),
                .epoch_control =
                    runtime.epoch_arena->deviceControlAddress(),
                .maintenance_epoch =
                    runtime.epoch_arena
                        ->deviceMaintenanceEpochAddress(),
                .maintenance_status =
                    runtime.epoch_arena
                        ->deviceMaintenanceStatusAddress(),
                .inference_boundary = const_cast<
                    ParticipantGpuRuntime *>(&runtime),
                .initial_runtime_publisher = const_cast<
                    ParticipantGpuRuntime *>(&runtime),
                .first_model_layer = runtime.runtime_table->firstModelLayer(),
            };
            if (!binding.backgroundPublicationValid() ||
                !binding.initialRuntimePublicationValid())
            {
                throw std::logic_error(
                    "Mapped follower produced an incomplete background-publication binding");
            }
            bindings.push_back(binding);
        }
        return bindings;
    }

    MoEOverlayInferenceTransactionProtocol::Config
    MoEOverlayParticipantGraphRunner::
        inferenceTransactionProtocolConfig() const
    {
        if (!transaction_topology_identity_.valid())
        {
            throw std::logic_error(
                "Participant retained graph family has no transaction topology identity");
        }
        return {
            .topology = transaction_topology_identity_,
            .slot_count = moeOverlayInferenceTransactionSlotCount(
                config_.max_mtp_draft_depth),
            .max_request_count = config_.max_request_count,
            .max_rows_per_request =
                config_.max_graph_activation_rows,
            .max_mtp_draft_depth =
                config_.max_mtp_draft_depth,
        };
    }

    std::unique_ptr<IInferenceRunner>
    createMoEOverlayParticipantGraphRunner(
        MoEOverlayParticipantGraphRunnerConfig config)
    {
        try
        {
            return std::make_unique<MoEOverlayParticipantGraphRunner>(
                std::move(config));
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Construction failed: "
                << error.what());
            return nullptr;
        }
    }

} // namespace llaminar2
