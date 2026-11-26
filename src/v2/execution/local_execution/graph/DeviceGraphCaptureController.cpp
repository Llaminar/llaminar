/**
 * @file DeviceGraphCaptureController.cpp
 * @brief Capture plan building and segment replay execution.
 */
#include "DeviceGraphCaptureControllerInternal.h"

namespace llaminar2
{


    const char *DeviceGraphCaptureController::phaseName(Phase phase)
    {
        switch (phase)
        {
        case Phase::Capture:
            return "capture";
        case Phase::Replay:
            return "replay";
        }

        throw std::logic_error("Unknown GPU graph replay phase");
    }

    bool DeviceGraphCaptureController::publishKernelInventory(
        const IGPUGraphCapture &capture,
        std::string_view fragment,
        std::optional<int> transaction_depth,
        DeviceId device,
        std::string &error)
    {
        if (!debugEnv().runtime_debug.gpu_graph_kernel_inventory)
            return true;

        const std::string fragment_name(fragment);
        const std::string depth_name = transaction_depth
                                           ? std::to_string(*transaction_depth)
                                           : "not_applicable";
        std::vector<GPUGraphKernelNodeInfo> kernel_nodes;
        std::string inspection_error;
        if (!capture.inspectKernelNodes(kernel_nodes, &inspection_error))
        {
            error = "kernel inventory failed for fragment '" +
                    fragment_name + "': " + inspection_error;
            return false;
        }

        size_t unresolved_names = 0u;
        for (const GPUGraphKernelNodeInfo &node : kernel_nodes)
        {
            if (!node.valid())
            {
                error =
                    "kernel inventory returned invalid launch geometry for fragment '" +
                    fragment_name + "' at " + node.graph_path;
                return false;
            }
            unresolved_names += node.name_resolved ? 0u : 1u;

            const std::string grid =
                std::to_string(node.grid_x) + "x" +
                std::to_string(node.grid_y) + "x" +
                std::to_string(node.grid_z);
            const std::string block =
                std::to_string(node.block_x) + "x" +
                std::to_string(node.block_y) + "x" +
                std::to_string(node.block_z);
            PerfStatsCollector::Tags tags{
                {"backend", capture.backendName()},
                {"fragment", fragment_name},
                {"transaction_depth", depth_name},
                {"kernel", node.name},
                {"grid", grid},
                {"block", block},
                {"dynamic_smem_bytes",
                 std::to_string(node.dynamic_shared_memory_bytes)},
                {"static_smem_bytes",
                 std::to_string(node.static_shared_memory_bytes)},
                {"local_bytes_per_thread",
                 std::to_string(node.local_memory_bytes_per_thread)},
                {"registers_per_thread",
                 std::to_string(node.registers_per_thread)},
                {"max_threads_per_block",
                 std::to_string(node.max_threads_per_block)},
                {"max_active_blocks_per_sm",
                 std::to_string(node.max_active_blocks_per_sm)},
                {"nesting_depth", std::to_string(node.nesting_depth)},
                {"name_resolved", node.name_resolved ? "true" : "false"},
            };
            if (!node.name_resolved)
            {
                tags.emplace(
                    "function_identity",
                    std::to_string(node.function_identity));
            }
            PerfStatsCollector::addCounter(
                "gpu_graph_inventory",
                "kernel_nodes",
                1.0,
                "graph_setup",
                device.toString(),
                std::move(tags));
        }

        const PerfStatsCollector::Tags summary_tags{
            {"backend", capture.backendName()},
            {"fragment", fragment_name},
            {"transaction_depth", depth_name},
        };
        auto kernel_summary_tags = summary_tags;
        kernel_summary_tags.emplace(
            "unresolved_names", std::to_string(unresolved_names));
        PerfStatsCollector::addCounter(
            "gpu_graph_inventory",
            "fragment_kernel_nodes",
            static_cast<double>(kernel_nodes.size()),
            "graph_setup",
            device.toString(),
            std::move(kernel_summary_tags));
        PerfStatsCollector::addCounter(
            "gpu_graph_inventory",
            "fragment_native_nodes",
            static_cast<double>(capture.nodeCount()),
            "graph_setup",
            device.toString(),
            summary_tags);
        return true;
    }

    DeviceGraphCaptureController::Transition DeviceGraphCaptureController::beginStep(
        bool initialized,
        bool &needs_capture,
        uint64_t &decode_step)
    {
        // Cached graph replay uses a monotonic step so we can reason about which
        // segments were executed in each phase.
        ++decode_step;

        if (!initialized)
        {
            return {Phase::Capture, decode_step};
        }

        if (needs_capture)
        {
            throw std::logic_error(
                "GPU graph cache exposed initialized-without-executable state");
        }

        return {Phase::Replay, decode_step};
    }

    void DeviceGraphCaptureController::prepareDeviceForGraphCapture(IDeviceContext *ctx)
    {
        if (ctx)
            ctx->activateDevice();
    }

    uint64_t DeviceGraphCaptureController::computeCaptureVariantSignature(ComputeGraph &graph)
    {
        constexpr uint64_t kFnvOffset = 1469598103934665603ull;
        constexpr uint64_t kFnvPrime = 1099511628211ull;

        uint64_t h = kFnvOffset;
        bool has_variant = graph.readAheadCaptureIdentity()!=0;
        if(has_variant){h ^= graph.readAheadCaptureIdentity();h *= kFnvPrime;h ^= graph.topologyGeneration();h *= kFnvPrime;}
        if(const auto& body=graph.retainedParticipantAuxiliaryRange()) {
            has_variant=true;h^=body->identity;h*=kFnvPrime;
            h^=static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(body->original_execution.get()));h*=kFnvPrime;
            h^=static_cast<std::uint64_t>(body->role);h*=kFnvPrime;
            if(body->lifecycle!=RetainedGraphParticipantAuxiliaryLifecycle::CurrentDemand){
                h^=static_cast<std::uint64_t>(body->lifecycle);h*=kFnvPrime;
                if(body->local_consumption){h^=body->local_consumption->captureIdentity();h*=kFnvPrime;
                    h^=reinterpret_cast<std::uintptr_t>(body->local_arena);h*=kFnvPrime;}
            }
            h^=graph.topologyGeneration();h*=kFnvPrime;
        }
        if(const auto participants=graph.retainedParticipantSourcePlan()){
            has_variant=true;
            h^=participants->captureIdentity();h*=kFnvPrime;
            h^=graph.topologyGeneration();h*=kFnvPrime;
        }
        if(const auto& domain=graph.retainedParticipantDefinitionDomain()){
            has_variant=true;
            for(const auto word:{static_cast<uint64_t>(domain->source.type),static_cast<uint64_t>(domain->source.ordinal),
                static_cast<uint64_t>(reinterpret_cast<std::uintptr_t>(domain->authority))}){h^=word;h*=kFnvPrime;}
        }
        if (const auto &transaction = graph.nativeTransactionCaptureContract()) {
            has_variant = true;
            for (const auto *word : {&transaction->identity, &transaction->terminal_node}) {
                for (const unsigned char byte : *word) { h ^= byte; h *= kFnvPrime; }
                h ^= 0xffu; h *= kFnvPrime;
            }
            h ^= static_cast<std::uint64_t>(graph.nativeCaptureEnvelope()); h *= kFnvPrime;
            h ^= graph.topologyGeneration(); h *= kFnvPrime;
        }
        const auto &order = graph.getExecutionOrder();
        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                continue;
            }

            const uint64_t stage_variant = node->stage->graphCaptureVariantSignature();
            const auto authority=static_cast<uint64_t>(node->stage->collectiveGraphCaptureAuthority());
            if (stage_variant == 0 && authority==0)
            {
                continue;
            }

            has_variant = true;
            const uint64_t name_hash = static_cast<uint64_t>(std::hash<std::string>{}(name));
            const uint64_t type_hash = static_cast<uint64_t>(static_cast<int>(node->stage->type()));
            for (uint64_t part : {name_hash, type_hash, stage_variant, authority})
            {
                h ^= part;
                h *= kFnvPrime;
            }
        }

        return has_variant ? h : 0;
    }

    const char *DeviceGraphCaptureController::replayModeName(
        const DeviceGraphExecutor::GraphSegmentCache &segment_cache)
    {
        return replayTopologyTag(segment_cache);
    }

    bool DeviceGraphCaptureController::constrainReplayPolicyToNativeEnvelope(
        GraphNativeCaptureEnvelope envelope,
        DeviceGraphExecutor::DecodeCapturePolicy &policy,
        std::string *error)
    {
        if (error)
            error->clear();

        const auto reject = [&](std::string reason)
        {
            if (error)
                *error = std::move(reason);
            return false;
        };

        switch (envelope)
        {
        case GraphNativeCaptureEnvelope::Ordinary:
            return true;
        case GraphNativeCaptureEnvelope::DeviceOwnedTimelineTransaction:
            if (!policy.allow_cached_graph_replay)
            {
                return reject(
                    "device-owned timeline requires mandatory native graph replay");
            }
            policy.graph_replay_plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph;
            policy.retained_parent_composer = {};
            policy.defer_final_sync = true;
            return true;
        case GraphNativeCaptureEnvelope::
            HeterogeneousTicketAuthorityTransaction:
        case GraphNativeCaptureEnvelope::
            HeterogeneousTicketFollowerTransaction:
            if (!policy.allow_cached_graph_replay ||
                !policy.heterogeneous_segmented_enabled)
            {
                return reject(
                    "heterogeneous ticket transaction requires topology-admitted segmented native replay");
            }
            policy.graph_replay_plan_policy =
                DeviceGraphExecutor::GraphReplayPlanPolicy::
                    AllowHeterogeneousBoundarySegmentation;
            policy.retained_parent_composer = {};
            policy.defer_final_sync = true;
            return true;
        }

        return reject("unknown graph-native capture envelope");
    }

    bool DeviceGraphCaptureController::prepareReplayGpuTiming(
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IWorkerGPUContext *gpu_ctx,
        const std::string &device_name)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled())
            return true;
        if (!gpu_ctx || !segment_cache.capture_stream ||
            segment_cache.segments.empty())
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Replay GPU timing preparation "
                "requires a complete replay plan, context, and capture stream");
            return false;
        }

        if (!segment_cache.replay_gpu_timing_slots.empty())
        {
            if (segment_cache.replay_gpu_timing_device_name != device_name)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Replay GPU timing ring was "
                    "prepared for a different device label");
                return false;
            }
            return true;
        }

        const GraphReplayCaptureMode replay_mode =
            captureModeForCache(segment_cache);
        const size_t simultaneous_intervals =
            replay_mode == GraphReplayCaptureMode::FullGraph
                ? 1
                : segment_cache.segments.size() + 1;
        const size_t slot_capacity = std::max(
            simultaneous_intervals,
            replay_mode == GraphReplayCaptureMode::FullGraph
                ? kDeferredReplayGpuTimingSlotCapacity
                : simultaneous_intervals);

        segment_cache.replay_gpu_timing_device_name = device_name;
        segment_cache.replay_gpu_timing_slots.resize(slot_capacity);
        for (auto &slot : segment_cache.replay_gpu_timing_slots)
        {
            slot.start_event = gpu_ctx->createEvent(GPUEventPurpose::Timing);
            slot.stop_event = gpu_ctx->createEvent(GPUEventPurpose::Timing);
            if (!slot.start_event || !slot.stop_event)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Failed to preallocate the "
                    "replay GPU timing event ring");
                segment_cache.destroyReplayGpuTimingEvents();
                return false;
            }
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "replay_timing_event_slots_preallocated",
            static_cast<double>(slot_capacity),
            "capture",
            device_name,
            {{"context", segment_cache.perf_context},
             {"replay_mode", captureModeTag(replay_mode)}});
        return true;
    }

    bool DeviceGraphCaptureController::reclaimReplayGpuTimingNonblocking(
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IWorkerGPUContext *gpu_ctx)
    {
        if (!PerfStatsCollector::gpuStageEventTimingEnabled() ||
            segment_cache.replay_gpu_timing_slots.empty())
        {
            return true;
        }
        if (!gpu_ctx)
            return false;
        return reclaimReplayGpuTimingNonblockingImpl(segment_cache, gpu_ctx);
    }

    void DeviceGraphCaptureController::buildCapturePlan(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        const std::unordered_set<std::string> *collective_nodes,
        bool has_collective_nodes,
        bool collectives_graph_capturable,
        DeviceGraphExecutor::GraphReplayPlanPolicy plan_policy)
    {
        segment_cache.segments.clear();
        segment_cache.graph_replay_plan_policy = plan_policy;
        segment_cache.runtime_collective_capture_requirement=DeviceGraphExecutor::GraphSegmentCache::RuntimeCollectiveCaptureRequirement::NotRequired;
        // One setup-only identity joins capture, transaction zero and replay.
        // Context names are semantic labels and can name several unused/live
        // variants; pointers are unsuitable because allocator reuse aliases them.
        static std::atomic<uint64_t> next_evidence_family{1};
        segment_cache.evidence_family = next_evidence_family.fetch_add(1, std::memory_order_relaxed);
        if (segment_cache.evidence_family == 0)
            throw std::overflow_error("Graph executable evidence identity exhausted");

        const auto &order = graph.getExecutionOrder();
        const GraphNativeCaptureEnvelope native_capture_envelope =
            graph.nativeCaptureEnvelope();
        const bool device_owned_timeline_transaction =
            requiresSingleNativeExecutable(native_capture_envelope);
        const bool heterogeneous_ticket_transaction =
            requiresHeterogeneousTicketSegmentation(
                native_capture_envelope);
        const bool heterogeneous_ticket_authority =
            ownsHeterogeneousTicketBoundary(native_capture_envelope);
        const bool heterogeneous_ticket_follower =
            followsHeterogeneousTicketBoundary(native_capture_envelope);
        const bool retained_parent_composition =
            DeviceGraphExecutor::isRetainedParentPlanPolicy(plan_policy);
        const bool concurrent_ticket_service =
            DeviceGraphExecutor::hasConcurrentTicketService(plan_policy);
        const bool retained_heterogeneous_authority =
            heterogeneous_ticket_authority && concurrent_ticket_service;
        const bool retained_heterogeneous_follower =
            heterogeneous_ticket_follower && retained_parent_composition &&
            !concurrent_ticket_service;
        const bool native_auxiliary_parent = device_owned_timeline_transaction &&
            graph.retainedParticipantAuxiliaryRange().has_value() &&
            retained_parent_composition && !concurrent_ticket_service;
        if (native_auxiliary_parent)
            graph.requireRetainedParticipantNativeAuxiliaryParent();
        if (device_owned_timeline_transaction && !native_auxiliary_parent &&
            plan_policy !=
                DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph)
        {
            throw std::logic_error(
                "A device-owned timeline transaction requires the full-graph replay policy");
        }
        if (heterogeneous_ticket_transaction &&
            plan_policy !=
                DeviceGraphExecutor::GraphReplayPlanPolicy::
                    AllowHeterogeneousBoundarySegmentation &&
            !retained_heterogeneous_authority &&
            !retained_heterogeneous_follower)
        {
            throw std::logic_error(
                "A heterogeneous ticket transaction requires explicit boundary segmentation, an authority parent with concurrent ticket service, or a graph-only follower parent");
        }
        const auto &segmented_collective_capture_allow =
            debugEnv().execution.gpu_graph_collective_segmented_capture_allow;

        auto stage_in_collective_allowlist = [&](const std::string &stage_name) -> bool
        {
            if (segmented_collective_capture_allow.empty())
            {
                return false;
            }
            for (const auto &needle : segmented_collective_capture_allow)
            {
                if (stage_name.find(needle) != std::string::npos)
                {
                    return true;
                }
            }
            return false;
        };

        graph.requireRetainedParticipantAuxiliaryRange();
        if(graph.retainedParticipantAuxiliaryRange() && !retained_parent_composition)
            throw std::invalid_argument("Participant auxiliary body requires its complete retained native parent");

        bool current_capturable = false;
        bool first = true;
        bool force_new_segment = false;
        size_t heterogeneous_ticket_unit_boundaries = 0u;
        size_t heterogeneous_ticket_terminal_units = 0u;

        for (const auto &name : order)
        {
            auto *node = graph.getNode(name);
            if (!node || !node->stage)
            {
                continue;
            }

            // A finite body is recorded separately and attached once to this
            // same complete parent. Never replay it serially as a model unit.
            if(graph.isRetainedParticipantAuxiliaryNode(name))continue;

            const auto *const wave_contract =
                native_capture_envelope ==
                            GraphNativeCaptureEnvelope::Ordinary &&
                        node->graph_capture_wave
                    ? &*node->graph_capture_wave
                    : nullptr;
            const auto *const ticket_unit_contract =
                node->heterogeneous_ticket_unit_contract
                    ? &*node->heterogeneous_ticket_unit_contract
                    : nullptr;
            if (ticket_unit_contract &&
                !heterogeneous_ticket_transaction)
            {
                throw std::logic_error(
                    "Graph node '" + name +
                    "' declares a heterogeneous ticket-unit boundary outside a heterogeneous ticket envelope");
            }
            if (wave_contract &&
                wave_contract->participation ==
                    GraphCaptureWaveParticipation::Passive)
            {
                if (!node->stage->isPassiveGraphCaptureNoOp())
                {
                    throw std::logic_error(
                        "Graph node '" + name + "' declares passive capture "
                        "participation, but its stage no longer certifies an "
                        "immutable no-op role");
                }
                if (first || segment_cache.segments.empty() ||
                    segment_cache.segments.back().stage_names.empty())
                {
                    throw std::logic_error(
                        "Passive capture node '" + name + "' has no preceding "
                        "execution segment to anchor its rendezvous");
                }

                auto &anchor = segment_cache.segments.back();
                const auto append_passive_wave =
                    [&](const std::string &identity,
                        const std::string *declarative_noop_stage)
                {
                    const auto duplicate = std::find_if(
                        anchor.passive_capture_waves_after.begin(),
                        anchor.passive_capture_waves_after.end(),
                        [&](const DeviceGraphExecutor::GraphSegment::
                                PassiveCaptureWave &wave)
                        {
                            return wave.identity == identity;
                        });
                    if (duplicate !=
                        anchor.passive_capture_waves_after.end())
                    {
                        throw std::logic_error(
                            "Captured segment repeats passive wave identity '" +
                            identity + "'");
                    }

                    DeviceGraphExecutor::GraphSegment::PassiveCaptureWave wave{
                        .ordinal = 0,
                        .identity = identity,
                    };
                    if (declarative_noop_stage)
                    {
                        wave.declarative_noop_stage_names.push_back(
                            *declarative_noop_stage);
                    }
                    anchor.passive_capture_waves_after.push_back(
                        std::move(wave));
                };

                append_passive_wave(wave_contract->identity, &name);
                for (const auto &following_identity :
                     wave_contract->passive_following_identities)
                {
                    append_passive_wave(following_identity, nullptr);
                }

                /*
                 * A passive wave is an intervening domain lifecycle boundary.
                 * The next local device operation must start a new executable;
                 * fusing it into the anchor would move that work ahead of the
                 * sibling wave this participant just joined.
                 */
                force_new_segment = true;
                continue;
            }

            // Start from stage capability, then layer on safety gates.
            bool stage_capturable = node->stage->isGraphCapturable();
            const bool preparation_dependent_capture =
                !stage_capturable && node->stage->supportsGraphCaptureAfterLaunchPreparation();
            if (preparation_dependent_capture)
            {
                stage_capturable = true;
            }
            const bool stage_capture_capable = stage_capturable;
            const bool collective_by_name = (collective_nodes && collective_nodes->count(name));
            const bool native_transport=node->stage->isParticipantLocalCapturedTransfer();
            const bool requires_runtime_capture=node->stage->requiresCollectiveRuntimeCapture(collective_by_name);
            if(requires_runtime_capture)segment_cache.runtime_collective_capture_requirement=DeviceGraphExecutor::GraphSegmentCache::RuntimeCollectiveCaptureRequirement::Required;

            /*
             * A runtime collective needs both stage readiness and its backend's
             * stream-ordered capture support. An actual participant-local native
             * transport owns its separate immutable captured binding, while
             * retaining the SAME readiness AND and semantic collective identity.
             * Backend capability must never override a stage-level rejection
             * caused by invalid geometry, missing workspace, or stale state.
             */
            if (requires_runtime_capture)
            {
                stage_capturable =
                    stage_capturable && collectives_graph_capturable;
            }
            const bool capture_wave_dormant_by_collective_policy =
                wave_contract &&
                wave_contract->participation ==
                    GraphCaptureWaveParticipation::Active &&
                stage_capture_capable &&
                requires_runtime_capture &&
                !collectives_graph_capturable;

            if (has_collective_nodes && !segmented_collective_capture_allow.empty())
            {
                // Explicit allowlist mode: only allowlisted stages are capturable
                stage_capturable = stage_capturable && stage_in_collective_allowlist(name);
            }
            if(native_transport && !stage_capturable)
                throw std::invalid_argument("Participant-local captured transfer is not capture-ready; manual ticket execution is forbidden: "+name);
            // Otherwise: trust each stage's isGraphCapturable() declaration.
            // Stages that need per-step updates either return false or report
            // segment boundaries around themselves when they can be captured
            // alone but cannot safely be fused with adjacent captured work.
            // A collective remains inside the surrounding captured segment
            // when both capture contracts pass. Only an explicitly enabled
            // compatibility policy may leave it as a manual segment.
            const bool boundary_before =
                stage_capturable &&
                node->stage->requiresGraphCaptureSegmentBoundaryBefore() &&
                !segment_cache.segments.empty() &&
                !segment_cache.segments.back().stage_names.empty();
            force_new_segment = force_new_segment || boundary_before;

            /*
             * An explicit identity is a logical capture-unit boundary. Two
             * adjacent captured nodes with different identities cannot share a
             * native executable because sibling graphs may participate in only
             * one of those waves. An unannotated prefix may join the first
             * explicit identity in its segment; that is how the continuation
             * compute and root-only ticket publication form one active wave.
             */
            if (stage_capturable && wave_contract &&
                !segment_cache.segments.empty() &&
                segment_cache.segments.back().capturable &&
                !segment_cache.segments.back().stage_names.empty() &&
                !segment_cache.segments.back().capture_wave_identity.empty() &&
                segment_cache.segments.back().capture_wave_identity !=
                    wave_contract->identity)
            {
                force_new_segment = true;
            }

            if (first || force_new_segment || stage_capturable != current_capturable)
            {
                // Create a new segment whenever capturable/manual mode changes
                // so each segment has uniform execution semantics.
                segment_cache.segments.emplace_back();
                segment_cache.segments.back().capturable = stage_capturable;
                current_capturable = stage_capturable;
                first = false;
                force_new_segment = false;
            }

            segment_cache.segments.back().stage_names.push_back(name);
            if (wave_contract)
            {
                if (!stage_capturable)
                {
                    /*
                     * An active collective wave can be declared once in the
                     * model graph even when an explicit diagnostic policy runs
                     * collectives as manual heterogeneous segments.  In that
                     * mode the wave is dormant: the manual collective remains
                     * in the plan and consumes no native-capture ordinal. A
                     * stage that is intrinsically incapable of capture still
                     * fails, so this cannot hide stale preparation or geometry.
                     */
                    if (capture_wave_dormant_by_collective_policy)
                    {
                        force_new_segment = false;
                        continue;
                    }
                    throw std::logic_error(
                        "Graph node '" + name + "' declares capture wave '" +
                        wave_contract->identity +
                        "' but is not graph-capturable");
                }

                auto &segment = segment_cache.segments.back();
                if (segment.capture_wave_identity.empty())
                {
                    segment.capture_wave_identity =
                        wave_contract->identity;
                }
                else if (segment.capture_wave_identity !=
                         wave_contract->identity)
                {
                    throw std::logic_error(
                        "Captured segment combines incompatible wave identities '" +
                        segment.capture_wave_identity + "' and '" +
                        wave_contract->identity + "'");
                }

                for (const auto &passive_identity :
                     wave_contract->passive_following_identities)
                {
                    const auto duplicate = std::find_if(
                        segment.passive_capture_waves_after.begin(),
                        segment.passive_capture_waves_after.end(),
                        [&](const DeviceGraphExecutor::GraphSegment::PassiveCaptureWave &wave)
                        {
                            return wave.identity == passive_identity;
                        });
                    if (duplicate != segment.passive_capture_waves_after.end())
                    {
                        throw std::logic_error(
                            "Captured segment repeats passive wave identity '" +
                            passive_identity + "'");
                    }
                    segment.passive_capture_waves_after.push_back(
                        {.ordinal = 0, .identity = passive_identity});
                }
            }
            if (ticket_unit_contract)
            {
                if (!stage_capturable ||
                    segment_cache.segments.empty() ||
                    !segment_cache.segments.back().capturable)
                {
                    throw std::logic_error(
                        "Heterogeneous ticket-unit boundary '" +
                        ticket_unit_contract->identity + "' on node '" +
                        name + "' does not close captured device work");
                }
                auto &segment = segment_cache.segments.back();
                if (!segment.capture_wave_identity.empty() &&
                    segment.capture_wave_identity !=
                        ticket_unit_contract->identity)
                {
                    throw std::logic_error(
                        "Captured segment combines heterogeneous ticket-unit identity '" +
                        ticket_unit_contract->identity +
                        "' with capture-wave identity '" +
                        segment.capture_wave_identity + "'");
                }
                segment.capture_wave_identity =
                    ticket_unit_contract->identity;
                switch (ticket_unit_contract->disposition)
                {
                case GraphHeterogeneousTicketUnitDisposition::
                    BeforeManualBoundary:
                    ++heterogeneous_ticket_unit_boundaries;
                    /* The next participant-local operation belongs to the
                     * unit after the authority's CPU work, even on a follower
                     * graph with no manual stage of its own. */
                    force_new_segment = true;
                    break;
                case GraphHeterogeneousTicketUnitDisposition::
                    TransactionTerminal:
                    if (graph.terminalNode() != name)
                    {
                        throw std::logic_error(
                            "Heterogeneous ticket terminal-unit identity '" +
                            ticket_unit_contract->identity + "' is attached "
                            "to non-terminal node '" + name + "'");
                    }
                    ++heterogeneous_ticket_terminal_units;
                    force_new_segment = false;
                    break;
                }
            }
            else
            {
                force_new_segment =
                    stage_capturable &&
                    node->stage->requiresGraphCaptureSegmentBoundaryAfter();
            }
        }

        const int max_stages = debugEnv().execution.gpu_graph_max_stages;
        if (max_stages > 0 && heterogeneous_ticket_transaction &&
            retained_parent_composition)
        {
            throw std::runtime_error(
                "gpu_graph_max_stages cannot override typed retained heterogeneous ticket compilation units");
        }
        if (max_stages > 0)
        {
            std::vector<DeviceGraphExecutor::GraphSegment> split_segments;
            for (auto &seg : segment_cache.segments)
            {
                if (seg.capturable && static_cast<int>(seg.stage_names.size()) > max_stages)
                {
                    if (device_owned_timeline_transaction)
                    {
                        throw std::runtime_error(
                            "gpu_graph_max_stages cannot split a device-owned timeline transaction");
                    }
                    if (!seg.capture_wave_identity.empty() ||
                        !seg.passive_capture_waves_after.empty())
                    {
                        throw std::runtime_error(
                            "gpu_graph_max_stages would split explicit capture wave '" +
                            (seg.capture_wave_identity.empty()
                                 ? std::string("<implicit>")
                                 : seg.capture_wave_identity) +
                            "'; synchronized role-asymmetric LocalTP waves may not be split independently");
                    }
                    for (size_t i = 0; i < seg.stage_names.size(); i += max_stages)
                    {
                        DeviceGraphExecutor::GraphSegment sub;
                        sub.capturable = true;
                        size_t end = std::min(i + static_cast<size_t>(max_stages), seg.stage_names.size());
                        for (size_t j = i; j < end; j++)
                        {
                            sub.stage_names.push_back(seg.stage_names[j]);
                        }
                        split_segments.push_back(std::move(sub));
                    }
                }
                else
                {
                    split_segments.push_back(std::move(seg));
                }
            }
            segment_cache.segments = std::move(split_segments);
        }

        /*
         * First prove the declarative captured/manual/captured lifecycle. Only
         * then lower it to bounded graph-only compiler shards and, on the
         * authority, one ordered CPU ticket service. The retained-parent
         * composer imports every captured shard into one executable, while the
         * mapped publication/wait kernels preserve every original cutpoint.
         */
        if (heterogeneous_ticket_authority)
        {
            validateHeterogeneousTicketSegmentLifecycle(
                graph,
                segment_cache.segments,
                HeterogeneousTicketSegmentRole::Authority,
                heterogeneous_ticket_unit_boundaries,
                heterogeneous_ticket_terminal_units);
        }

        if (heterogeneous_ticket_follower)
        {
            validateHeterogeneousTicketSegmentLifecycle(
                graph,
                segment_cache.segments,
                HeterogeneousTicketSegmentRole::Follower,
                heterogeneous_ticket_unit_boundaries,
                heterogeneous_ticket_terminal_units);
        }

        if (heterogeneous_ticket_transaction &&
            retained_parent_composition)
        {
            segment_cache.segments = lowerHeterogeneousTicketCompilationPlan(
                std::move(segment_cache.segments),
                heterogeneous_ticket_authority);
        }

        exposeParticipantAuxiliaryProbeFrontier(graph,segment_cache.segments);

        if (const auto &transaction = graph.nativeTransactionCaptureContract())
        {
            if (heterogeneous_ticket_transaction || segment_cache.segments.empty() ||
                (!native_auxiliary_parent && segment_cache.segments.size() != 1u) ||
                graph.terminalNode() != transaction->terminal_node || !graph.getNode(transaction->terminal_node) ||
                std::find(segment_cache.segments.back().stage_names.begin(),
                    segment_cache.segments.back().stage_names.end(), transaction->terminal_node) ==
                    segment_cache.segments.back().stage_names.end())
                throw std::logic_error("Whole native transaction capture rejected a conflicting or partial plan");
            for (auto& segment : segment_cache.segments)
            {
                if (!segment.capturable || (!native_auxiliary_parent && !segment.capture_wave_identity.empty() &&
                    segment.capture_wave_identity != transaction->identity))
                    throw std::logic_error("Whole native transaction capture rejected a manual or foreign child");
                // These are graph-only compilation fragments, never independently
                // replayable transactions. Final original-parent attachment and
                // complete ordered coverage remain mandatory before instantiation.
                if (!native_auxiliary_parent)
                    segment.capture_wave_identity = transaction->identity;
            }
            // Probe-prefix identity is the existing participant rendezvous and
            // must stay intact. The terminal fragment retains the original
            // whole-transaction label; only their final parent is executable.
            if (native_auxiliary_parent && segment_cache.segments.back().capture_wave_identity.empty())
                segment_cache.segments.back().capture_wave_identity = transaction->identity;
        }

        if (retained_parent_composition)
        {
            /*
             * Child captures share stable arena storage but do not execute
             * while they are recorded. A read crossing a child boundary is
             * therefore neither globally authoritative nor graph-internal to
             * the child. Freeze the exact BufferId proof here, while the whole
             * retained-parent topology is still visible. The composer later
             * clones these children in this same order and supplies the actual
             * device-side producer/consumer edge.
             *
             * Writes become eligible only after the complete current segment,
             * so an ordinary producer within this child remains an internal
             * dependency rather than masquerading as a parent import.
             */
            std::unordered_set<BufferId> prior_child_writes;
            if (segment_cache.prior_child_input_authority)
            {
                // Only the closed original transport composer can mint this proof.
                // The local consumer contract remains unchanged and all other
                // inputs still need a producer in this graph or live ready bytes.
                const auto &proof = *segment_cache.prior_child_input_authority;
                proof.require(graph, segment_cache.capture_device, segment_cache.capture_stream);
                prior_child_writes.insert(proof.inputs().begin(), proof.inputs().end());
            }
            for (auto &segment : segment_cache.segments)
            {
                if (!segment.capturable)
                {
                    if (!concurrent_ticket_service)
                    {
                        throw std::runtime_error(
                            "Retained parent composition forbids manual replay units");
                    }

                    size_t device_ingress_publishers = 0u;
                    for (const std::string &stage_name : segment.stage_names)
                    {
                        const ComputeNode *const node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                        {
                            throw std::runtime_error(
                                "Concurrent ticket-service planning cannot resolve stage '" +
                                stage_name + "'");
                        }
                        const StageBufferContract contract =
                            node->stage->bufferContract();
                        if (!node->stage->isManualGraphBoundary() ||
                            node->stage->manualGraphBoundaryScheduling() !=
                                ManualGraphBoundaryScheduling::
                                    ConcurrentTicketService ||
                            !contract.allArenaReads().empty() ||
                            !contract.allWrites().empty())
                        {
                            throw std::runtime_error(
                                "Concurrent ticket-service stage '" + stage_name +
                                "' lacks a self-contained mapped-ticket boundary contract");
                        }
                        if (node->stage->concurrentManualFailureRole() ==
                            ConcurrentManualFailureRole::DeviceIngressPublisher)
                        {
                            ++device_ingress_publishers;
                        }
                    }
                    if (device_ingress_publishers == 0u)
                    {
                        throw std::runtime_error(
                            "Concurrent ticket-service manual unit has no authenticated device-ingress publisher for failure drainage");
                    }
                    continue;
                }

                std::unordered_set<BufferId> segment_parent_inputs;
                std::unordered_set<BufferId> segment_writes;
                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        throw std::runtime_error(
                            "Retained parent dependency planning cannot resolve stage '" +
                            stage_name + "'");
                    }

                    const StageBufferContract contract =
                        node->stage->bufferContract();
                    for (const auto &binding : contract.allArenaReads())
                    {
                        if (prior_child_writes.contains(binding.id))
                            segment_parent_inputs.insert(binding.id);
                    }
                    for (const auto &binding : contract.allWrites())
                        segment_writes.insert(binding.id);
                }

                segment.retained_parent_input_ids.assign(
                    segment_parent_inputs.begin(),
                    segment_parent_inputs.end());
                std::sort(
                    segment.retained_parent_input_ids.begin(),
                    segment.retained_parent_input_ids.end());
                prior_child_writes.insert(
                    segment_writes.begin(), segment_writes.end());
            }
        }

        /*
         * Ordinals count domain lifecycle waves, not local replay segments.
         * Manual sparse work exists only on the continuation root and therefore
         * must not perturb the number used by the strict LocalTP rendezvous.
         * A passive follower consumes one ordinal exactly as an active sibling
         * segment does, keeping every later common capture wave aligned.
         */
        size_t next_capture_wave_ordinal = 0;
        for (auto &segment : segment_cache.segments)
        {
            if (segment.capturable)
                segment.capture_wave_ordinal = next_capture_wave_ordinal++;
            for (auto &passive_wave : segment.passive_capture_waves_after)
                passive_wave.ordinal = next_capture_wave_ordinal++;
        }

        size_t capturable_segments = 0, manual_segments = 0;
        size_t capturable_stages = 0, manual_stages = 0;
        size_t max_capturable_segment_stages = 0;
        size_t max_manual_segment_stages = 0;
        std::map<std::string, size_t> capturable_stage_types;
        std::map<std::string, size_t> manual_stage_types;
        for (const auto &seg : segment_cache.segments)
        {
            if (seg.capturable)
            {
                capturable_segments++;
                capturable_stages += seg.stage_names.size();
                max_capturable_segment_stages =
                    std::max(max_capturable_segment_stages, seg.stage_names.size());
            }
            else
            {
                manual_segments++;
                manual_stages += seg.stage_names.size();
                max_manual_segment_stages =
                    std::max(max_manual_segment_stages, seg.stage_names.size());
            }

            auto &stage_types = seg.capturable ? capturable_stage_types : manual_stage_types;
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    continue;
                }
                stage_types[computeStageTypeName(node->stage->type())]++;
            }
        }

        if (device_owned_timeline_transaction &&
            (manual_segments != 0u || capturable_segments == 0u ||
             capturable_segments != segment_cache.segments.size() ||
             (!native_auxiliary_parent && segment_cache.segments.size() != 1u) ||
             std::any_of(segment_cache.segments.begin(), segment_cache.segments.end(),
                 [](const auto& segment) { return !segment.passive_capture_waves_after.empty(); })))
        {
            std::ostringstream diagnostic;
            diagnostic
                << "Device-owned timeline capture did not lower to exactly "
                   "one native executable"
                << ": segments=" << segment_cache.segments.size()
                << " capturable_segments=" << capturable_segments
                << " manual_segments=" << manual_segments
                << " capturable_stages=" << capturable_stages
                << " manual_stages=" << manual_stages;
            for (std::size_t segment_index = 0u;
                 segment_index < segment_cache.segments.size();
                 ++segment_index)
            {
                const auto &segment =
                    segment_cache.segments[segment_index];
                diagnostic << " segment[" << segment_index
                           << "]={capturable="
                           << (segment.capturable ? "true" : "false")
                           << ",stages=" << segment.stage_names.size()
                           << ",passive_waves="
                           << segment.passive_capture_waves_after.size();
                if (!segment.capturable)
                {
                    diagnostic << ",manual_frontier=[";
                    bool first_stage = true;
                    for (const auto &stage_name : segment.stage_names)
                    {
                        auto *const node = graph.getNode(stage_name);
                        if (!first_stage)
                            diagnostic << ';';
                        first_stage = false;
                        diagnostic << stage_name;
                        if (node && node->stage)
                        {
                            diagnostic << "(type="
                                       << computeStageTypeName(
                                              node->stage->type())
                                       << ",readiness="
                                       << node->stage
                                              ->graphCaptureReadinessDebugString()
                                       << ')';
                        }
                    }
                    diagnostic << ']';
                }
                diagnostic << '}';
            }
            throw std::runtime_error(diagnostic.str());
        }

        if (retained_parent_composition)
        {
            if (capturable_segments == 0u ||
                (!concurrent_ticket_service && manual_segments != 0u))
            {
                throw std::runtime_error(
                    "Retained parent composition requires one or more native child graphs and forbids manual replay units");
            }
            if (concurrent_ticket_service &&
                !heterogeneous_ticket_authority)
            {
                throw std::runtime_error(
                    "Concurrent ticket service is legal only for a typed heterogeneous ticket authority");
            }
            for (const auto &segment : segment_cache.segments)
            {
                if (!segment.capturable)
                    continue;
                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        (!concurrent_ticket_service &&
                         node->stage->graphLaunchPreparationPolicy() ==
                             GraphLaunchPreparationPolicy::CaptureAndReplay))
                    {
                        throw std::runtime_error(
                            "Retained parent composition requires immutable replay metadata for stage '" +
                            stage_name + "'");
                    }
                }
            }
        }

        const GraphReplayCaptureMode plan_mode =
            captureModeForPlan(segment_cache.segments.size(), capturable_segments, manual_segments);

        if (plan_mode == GraphReplayCaptureMode::FullGraph)
        {
            LOG_DEBUG("[DeviceGraphExecutor] Full graph capture plan: "
                      << capturable_stages << " capturable stages, no manual segments");
        }
        else
        {
            LOG_DEBUG("[DeviceGraphExecutor] Segmented graph capture plan: "
                      << capturable_segments << " capturable segments ("
                      << capturable_stages << " stages) + "
                      << manual_segments << " manual segments ("
                      << manual_stages << " stages)");
        }

        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(segment_cache.segments.size()),
                "decode",
                "",
                {{"type", "total"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(capturable_segments),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planUnitMetricName(plan_mode),
                static_cast<double>(manual_segments),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMetricName(plan_mode, "stages"),
                static_cast<double>(capturable_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMetricName(plan_mode, "stages"),
                static_cast<double>(manual_stages),
                "decode",
                "",
                {{"type", "manual"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMaxUnitStagesMetricName(plan_mode),
                static_cast<double>(max_capturable_segment_stages),
                "decode",
                "",
                {{"type", "capturable"}});
            PerfStatsCollector::addCounter(
                "forward_graph",
                planMaxUnitStagesMetricName(plan_mode),
                static_cast<double>(max_manual_segment_stages),
                "decode",
                "",
                {{"type", "manual"}});

            for (const auto &[type_name, count] : capturable_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    planMetricName(plan_mode, "stage_types"),
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{planUnitTypeTagName(plan_mode), "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    planMetricName(plan_mode, "stage_types"),
                    static_cast<double>(count),
                    "decode",
                    "",
                    {{planUnitTypeTagName(plan_mode), "manual"}, {"stage_type", type_name}});
            }

            auto record_stage_gpu_plan = [&](const char *name,
                                             double value,
                                             PerfStatsCollector::Tags tags)
            {
                addReplayWorkloadTags(
                    tags,
                    segment_cache.replay_workload);
                PerfStatsCollector::addCounter(
                    "stage_gpu",
                    name,
                    value,
                    "decode",
                    "",
                    graphReplayMetadataTags(std::move(tags), segment_cache.perf_context, plan_mode));
            };

            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(segment_cache.segments.size()),
                {{"type", "total"}});
            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(capturable_segments),
                {{"type", "capturable"}});
            record_stage_gpu_plan(
                stageGpuPlanUnitMetricName(plan_mode),
                static_cast<double>(manual_segments),
                {{"type", "manual"}});
            record_stage_gpu_plan(
                "graph_replay_plan_stages",
                static_cast<double>(capturable_stages),
                {{"type", "capturable"}});
            record_stage_gpu_plan(
                "graph_replay_plan_stages",
                static_cast<double>(manual_stages),
                {{"type", "manual"}});

            for (const auto &[type_name, count] : capturable_stage_types)
            {
                record_stage_gpu_plan(
                    "graph_replay_plan_stage_types",
                    static_cast<double>(count),
                    {{planUnitTypeTagName(plan_mode), "capturable"}, {"stage_type", type_name}});
            }
            for (const auto &[type_name, count] : manual_stage_types)
            {
                record_stage_gpu_plan(
                    "graph_replay_plan_stage_types",
                    static_cast<double>(count),
                    {{planUnitTypeTagName(plan_mode), "manual"}, {"stage_type", type_name}});
            }
        }

        if (plan_mode == GraphReplayCaptureMode::Segmented)
        {
            std::vector<std::string> undeclared_manual_boundaries;
            for (const auto &segment : segment_cache.segments)
            {
                if (segment.capturable)
                    continue;
                for (const auto &stage_name : segment.stage_names)
                {
                    const auto *node = graph.getNode(stage_name);
                    if (!node || !node->stage ||
                        (!node->stage->isCollectiveStage() &&
                         !node->stage->isManualGraphBoundary()))
                    {
                        undeclared_manual_boundaries.push_back(stage_name);
                    }
                }
            }

            const bool heterogeneous_boundary_segmentation_admitted =
                plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        AllowHeterogeneousBoundarySegmentation &&
                undeclared_manual_boundaries.empty();
            const bool retained_parent_composition_admitted =
                retained_parent_composition &&
                (manual_segments == 0u || concurrent_ticket_service) &&
                undeclared_manual_boundaries.empty();

            if (!heterogeneous_boundary_segmentation_admitted &&
                !retained_parent_composition_admitted)
            {
                std::ostringstream detail;
                detail
                    << "GPU replay planning produced "
                    << capturable_segments << " capturable segment(s) and "
                    << manual_segments << " manual segment(s), but this "
                       "execution domain requires one fully captured graph";
                if (plan_policy ==
                    DeviceGraphExecutor::GraphReplayPlanPolicy::RequireFullGraph)
                {
                    detail
                        << "; the execution topology did not admit a "
                           "heterogeneous manual boundary";
                }
                else if (retained_parent_composition)
                {
                    detail << (concurrent_ticket_service
                                   ? "; retained parent concurrent service admits only mapped-ticket manual units with an authenticated abort publisher"
                                   : "; retained parent composition admits native child graphs only and cannot contain a manual unit");
                }
                if (!undeclared_manual_boundaries.empty())
                {
                    detail << "; uncapturable stages without an explicit "
                              "collective/manual-boundary contract:";
                    for (const auto &stage_name : undeclared_manual_boundaries)
                        detail << ' ' << stage_name;
                }
                detail << ". Manual stage types:";
                if (manual_stage_types.empty())
                {
                    detail << " <none>";
                }
                else
                {
                    for (const auto &[type_name, count] : manual_stage_types)
                    {
                        detail << ' ' << type_name << '=' << count;
                    }
                }
                detail << ". Manual stages:";
                bool described_manual_stage = false;
                for (const auto &segment : segment_cache.segments)
                {
                    if (segment.capturable)
                        continue;

                    for (const auto &stage_name : segment.stage_names)
                    {
                        auto *node = graph.getNode(stage_name);
                        if (!node || !node->stage)
                        {
                            detail << " [" << stage_name << ": missing-node]";
                            described_manual_stage = true;
                            continue;
                        }

                        const std::string readiness =
                            node->stage->graphCaptureReadinessDebugString();
                        detail
                            << " [" << stage_name
                            << ": type="
                            << computeStageTypeName(node->stage->type())
                            << ", launch_preparation="
                            << (node->stage->supportsGraphCaptureAfterLaunchPreparation()
                                    ? "true"
                                    : "false");
                        if (!readiness.empty())
                            detail << ", " << readiness;
                        detail << ']';
                        described_manual_stage = true;
                    }
                }
                if (!described_manual_stage)
                    detail << " <none>";

                LOG_ERROR("[DeviceGraphCaptureController] " << detail.str());
                throw std::runtime_error(detail.str());
            }
        }

        for (auto &seg : segment_cache.segments)
        {
            seg.last_executed_step = 0;
            seg.evidence_family = segment_cache.evidence_family;
        }
    }

    bool DeviceGraphCaptureController::executeStreamOnlyReplay(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegmentCache &segment_cache,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool use_default_stream)
    {
        if(graph.readAheadBranch())throw std::logic_error("Read-ahead requires an unchanged complete captured parent; eager/manual/recapture verification is forbidden");
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay missing context");
            return false;
        }

        // Always use an explicit stream object — never nullptr (hipStream_t(0)).
        // AMDDeviceContext creates a real hipStream_t via hipStreamCreateWithFlags,
        // so defaultStream() is a distinct object from hipStream_t(0). Using nullptr
        // would target the HIP legacy null stream, causing stream identity mismatches
        // with event-based synchronization.
        void *use_stream = use_default_stream ? gpu_ctx->defaultStream() : capture_stream;
        for (auto &seg : segment_cache.segments)
        {
            for (const auto &stage_name : seg.stage_names)
            {
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay missing stage: " << stage_name);
                    return false;
                }

                node->stage->setGPUStream(use_stream);
                if (!node->stage->execute(ctx))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only stage failed: " << stage_name);
                    return false;
                }
                if (node->stage->isManualGraphBoundary() &&
                    !node->stage->manualGraphBoundaryComplete())
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Stream-only manual boundary incomplete: "
                              << stage_name);
                    return false;
                }
                graph.markCompleted(stage_name);
            }
        }

        if (!gpu_ctx->synchronizeStreamChecked(use_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Stream-only replay sync failed");
            return false;
        }
        return true;
    }

    bool DeviceGraphCaptureController::segmentHasNonIdempotentStage(
        ComputeGraph &graph,
        const DeviceGraphExecutor::GraphSegment &segment)
    {
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                continue;
            }

            if (node->stage->type() == ComputeStageType::ADD_RESIDUAL)
            {
                return true;
            }
        }
        return false;
    }

    bool DeviceGraphCaptureController::executeManualReplaySegment(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool has_collective_nodes,
        bool needs_segment_sync,
        uint64_t current_step,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb)
    {
        if(graph.readAheadBranch())throw std::logic_error("Read-ahead requires an unchanged complete captured parent; eager/manual/recapture verification is forbidden");
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Manual replay missing context");
            return false;
        }

        bool manual_had_collective = false;
        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual segment missing stage: " << stage_name);
                return false;
            }

            const auto stage_type = node->stage->type();
            if(node->stage->isParticipantLocalCapturedTransfer()){
                LOG_ERROR("[DeviceGraphCaptureController] Native captured transfer cannot execute as a manual service: "<<stage_name);
                return false;
            }
            const bool is_collective = node->stage->isCollectiveStage();

            if (is_collective)
            {
                // Collective stages (allreduce, allgather) run on their own
                // internal streams (RCCL/NCCL). The preceding captured graph
                // segment was replayed on capture_stream, so we MUST ensure
                // graph compute completes before the collective reads GPU data.
                //
                // We use GPU-side event dependencies instead of host-blocking
                // synchronizeStream() + hipDeviceSynchronize().  Host-level
                // blocking can deadlock multi-device TP: device A blocks in
                // hipDeviceSynchronize (waiting for RCCL, which needs device B)
                // while device B blocks in synchronizeStream or its own
                // hipDeviceSynchronize from a different segment.
                //
                // GPU-side events let both device threads queue ALL segments
                // without blocking, so all RCCL calls are enqueued promptly
                // and the only host sync is the final stream-level sync
                // at the end of executeReplayPhase().
                manual_had_collective = true;

                // Use the ACTUAL default stream, not nullptr (hipStream_t(0)).
                // AMDDeviceContext creates a real hipStream_t via
                // hipStreamCreateWithFlags, so defaultStream() != nullptr.
                // The RCCL coordinator's pre/post sync events are recorded
                // on this same stream (registered via setComputeStreams()),
                // so the event chain MUST target the same stream object.
                // Using nullptr would target the HIP null stream which is
                // a DIFFERENT stream — event dependencies wouldn't chain
                // with the allreduce at all, causing intermittent stale
                // reads and deadlocks.
                void *compute_stream = gpu_ctx->defaultStream();

                if (capture_stream)
                {
                    // GPU-side: compute_stream waits for capture_stream.
                    // The allreduce's pre-sync event (hipEventRecord on
                    // compute_stream) will then chain after graph
                    // completion, ensuring RCCL reads committed data.
                    if (!gpu_ctx->insertStreamDependency(
                            compute_stream,
                            capture_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                                  "capture-to-collective stream dependency for "
                                  << stage_name);
                        return false;
                    }
                }

                node->stage->setGPUStream(compute_stream);
                if (!execute_node_cb(*node))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual collective stage failed on replay: " << stage_name);
                    return false;
                }

                if (capture_stream)
                {
                    // GPU-side: capture_stream waits for compute_stream.
                    // compute_stream already has a dependency on RCCL
                    // completion (via the allreduce's internal post-sync
                    // event), so capture_stream chains after RCCL finishes.
                    if (!gpu_ctx->insertStreamDependency(
                            capture_stream,
                            compute_stream))
                    {
                        LOG_ERROR("[DeviceGraphCaptureController] Failed to publish "
                                  "collective-to-capture stream dependency for "
                                  << stage_name);
                        return false;
                    }
                }

            }
            else
            {
                node->stage->setGPUStream(capture_stream);
                const bool stage_ok = execute_node_cb
                                          ? execute_node_cb(*node)
                                          : node->stage->execute(ctx);
                if (!stage_ok)
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual stage failed on replay: " << stage_name);
                    return false;
                }
            }

            if (record_snapshot_copies_cb &&
                !record_snapshot_copies_cb(*node, capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual replay snapshot copy failed: "
                          << stage_name);
                return false;
            }

            if (node->stage->isManualGraphBoundary() &&
                !node->stage->manualGraphBoundaryComplete())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Manual boundary incomplete on replay: "
                          << stage_name);
                return false;
            }
        }

        segment.last_executed_step = current_step;
        if (needs_segment_sync)
        {
            if (manual_had_collective)
            {
                // Collective stages ran on compute_stream (defaultStream),
                // non-collective stages ran on capture_stream. Sync both
                // instead of using device-wide hipDeviceSynchronize.
                if (!gpu_ctx->synchronizeStreamChecked(gpu_ctx->defaultStream()))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay default-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay capture-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
            }
            else
            {
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Manual replay capture-stream sync failed after segment starting at "
                              << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                    return false;
                }
            }
        }

        return true;
    }

    bool DeviceGraphCaptureController::executeCapturedReplaySegmentNormal(
        DeviceGraphExecutor::GraphSegment &segment,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
        bool full_graph_replay,
        const std::string &perf_context,
        const std::string &device_name,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        if (!gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Normal replay missing GPU context");
            return false;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Normal replay missing segment capture");
            return false;
        }
        if (!capture_stream ||
            segment.capture->executionStream() != capture_stream)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Replay executable does not own "
                "the declared graph-cache producer stream"
                << " declared_stream=" << capture_stream
                << " executable_stream="
                << segment.capture->executionStream());
            return false;
        }

        const bool profiling =
            KernelProfiler::isEnabled() ||
            PerfStatsCollector::isDomainEnabled("forward_pass");
        const GraphReplayCaptureMode replay_mode = full_graph_replay
                                                       ? GraphReplayCaptureMode::FullGraph
                                                       : GraphReplayCaptureMode::Segmented;
#ifdef HAVE_ROCM
        if (perf_context == "main_decode" && device_name == "ROCm:0")
            RocmDecodeTimelineProbe::instance().maybeDumpOnReplay();
#endif

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::SteadyReplay,
                capture_stream))
        {
            return false;
        }

        // Time the graph launch itself
        auto launch_t0 = std::chrono::high_resolution_clock::now();
        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Segment graph launch failed on replay");
            return false;
        }
        if (profiling)
        {
            auto launch_t1 = std::chrono::high_resolution_clock::now();
            ForwardPassProfiler::addReplayLaunchNs(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count()));
        }
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            auto launch_t1 = std::chrono::high_resolution_clock::now();
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                replayMetricName(replay_mode, "graph_launch"),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(launch_t1 - launch_t0).count()),
                "decode",
                device_name,
                replaySegmentTags(segment, perf_context));
        }

        // Time post-launch output-coherence publication.
        auto post_t0 = std::chrono::high_resolution_clock::now();
        post_launch_cb(segment, capture_stream);
        if (profiling)
        {
            auto post_t1 = std::chrono::high_resolution_clock::now();
            ForwardPassProfiler::addReplayPostLaunchNs(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(post_t1 - post_t0).count()));
        }
        if (PerfStatsCollector::isDomainEnabled("forward_graph"))
        {
            auto post_t1 = std::chrono::high_resolution_clock::now();
            PerfStatsCollector::recordTimingNs(
                "forward_graph",
                replayMetricName(replay_mode, "post_launch"),
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(post_t1 - post_t0).count()),
                "decode",
                device_name,
                replaySegmentTags(segment, perf_context));
        }

        if (needs_segment_sync)
        {
            if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Captured replay stream sync failed after segment starting at "
                          << (segment.stage_names.empty() ? std::string("<empty>") : segment.stage_names.front()));
                return false;
            }
        }

        return true;
    }

    bool DeviceGraphCaptureController::executeCapturedReplaySegmentRecapture(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        int segment_index,
        uint64_t current_step,
        const std::string &perf_context,
        const std::function<bool(const DeviceGraphExecutor::GraphSegment &)> &cohere_inputs_cb,
        const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
            const DeviceGraphExecutor::GraphSegment &)> &plan_capture_dependencies_cb,
        const std::function<bool(ComputeNode &)> &execute_node_cb,
        const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary_cb,
        IGraphCaptureAuxiliaryBranch *auxiliary_branch,
        const std::function<bool(ComputeNode &, void *)> &record_snapshot_copies_cb,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        if(graph.readAheadBranch())throw std::logic_error("Read-ahead requires an unchanged complete captured parent; eager/manual/recapture verification is forbidden");
        if (auxiliary_branch)
        {
            LOG_ERROR("[DeviceGraphCaptureController] A sealed auxiliary authority cannot be diagnostically recaptured");
            return false;
        }
        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture missing context");
            return false;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture missing segment capture");
            return false;
        }
        if (!execute_node_cb)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Re-capture requires the canonical "
                "executor stage hook");
            return false;
        }

        for (const auto &stage_name : segment.stage_names)
        {
            auto *node = graph.getNode(stage_name);
            if (node && node->stage)
            {
                node->stage->setGPUStream(capture_stream);
            }
        }

        if (!prepareGraphLaunchMetadata(
                graph,
                segment,
                ctx,
                capture_stream,
                GraphLaunchPreparationPhase::Capture))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture metadata preparation failed, seg "
                      << segment_index);
            return false;
        }
        /*
         * Launch metadata preparation can publish new device writes. Join those
         * exact event generations to the capture stream only after preparation,
         * but before the domain rendezvous and beginCapture(). Stage-level
         * requirePreparedInput() calls inside capture then prove this prejoin
         * instead of attempting an illegal external-event import.
         */
        if (!cohere_inputs_cb)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture requires a pre-capture input coherence hook, seg "
                      << segment_index);
            return false;
        }
        if (!cohere_inputs_cb(segment))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture input preflight failed, seg "
                      << segment_index);
            return false;
        }

        bool exec_ok = true;
        std::string failed_stage_name;
        auto dependency_ledger = planCaptureDependencies(
            graph,
            segment,
            ctx->deviceId(),
            capture_stream,
            plan_capture_dependencies_cb,
            "segment_recapture_" + std::to_string(segment_index));
        if (!dependency_ledger)
            return false;
        std::vector<IComputeStage *> capture_stages;
        capture_stages.reserve(segment.stage_names.size());
        for (const auto &stage_name : segment.stage_names)
        {
            ComputeNode *node = graph.getNode(stage_name);
            capture_stages.push_back(
                node && node->stage ? node->stage.get() : nullptr);
        }
        {
            if (capture_boundary_cb)
            {
                const std::string boundary_name =
                    graphCaptureBoundaryName(
                        "recapture_begin",
                        current_step,
                        segment.capture_wave_ordinal,
                        segment.capture_wave_identity,
                        &segment,
                        perf_context);
                if (!capture_boundary_cb(boundary_name, capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Re-capture begin boundary rendezvous failed, seg "
                              << segment_index << " boundary=" << boundary_name);
                    return false;
                }
            }

            ScopedGraphCaptureStageActivity stage_capture_activity(
                capture_stages,
                ctx,
                capture_stream,
                "segment recapture index=" +
                    std::to_string(segment_index));
            ScopedBackendGraphCapture capture_transaction(
                *gpu_ctx,
                *segment.capture,
                "segment recapture index=" +
                    std::to_string(segment_index),
                dependency_ledger.get());
            if (!stage_capture_activity.begin())
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Re-capture stage activity admission failed, seg "
                    << segment_index);
                return false;
            }
            if (!capture_transaction.begin())
            {
                (void)stage_capture_activity.finish(
                    GraphCaptureActivityTransition::Aborted);
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture beginCapture failed, seg " << segment_index);
                return false;
            }

            for (const auto &stage_name : segment.stage_names)
            {
                if (!exec_ok)
                    break;
                auto *node = graph.getNode(stage_name);
                if (!node || !node->stage || !execute_node_cb(*node))
                {
                    exec_ok = false;
                    failed_stage_name = stage_name;
                    break;
                }
                if (record_snapshot_copies_cb &&
                    !record_snapshot_copies_cb(*node, capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Re-capture snapshot copy failed: "
                              << stage_name);
                    exec_ok = false;
                    failed_stage_name = stage_name;
                    break;
                }
            }

            capture_transaction.finish();
            if (!stage_capture_activity.finish(
                    exec_ok
                        ? GraphCaptureActivityTransition::Completed
                        : GraphCaptureActivityTransition::Aborted))
            {
                if (exec_ok)
                    failed_stage_name = "<stage_capture_activity>";
                exec_ok = false;
            }
        }

        /*
         * endCapture() is participant-local. A faster LocalTP participant must
         * not proceed to a manual NCCL/RCCL segment while a sibling remains in
         * global stream-capture mode, because CUDA/HIP then rejects the
         * collective stream dependency. Pair every begin rendezvous with this
         * exit rendezvous before graph instantiation or manual execution.
         */
        if (capture_boundary_cb)
        {
            const std::string boundary_name =
                graphCaptureBoundaryName(
                    exec_ok ? "recapture_end" : "recapture_end_failed",
                    current_step,
                    segment.capture_wave_ordinal,
                    segment.capture_wave_identity,
                    &segment,
                    perf_context);
            if (!capture_boundary_cb(boundary_name, capture_stream))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture end boundary rendezvous failed, seg "
                          << segment_index << " boundary=" << boundary_name);
                return false;
            }
        }

        if (!exec_ok)
        {
            LOG_ERROR(
                "[DeviceGraphCaptureController] Re-capture stage execution "
                "failed; refusing eager recovery, seg "
                << segment_index
                << " stage="
                << (failed_stage_name.empty()
                        ? std::string("<unknown>")
                        : failed_stage_name));
            return false;
        }

        /*
         * Executable publication is selected by the graph implementation's
         * typed capability, not by device-name knowledge in the orchestrator.
         * A backend without in-place update support replaces its executable
         * directly. Runtime update failures are therefore genuine failures,
         * never a feature-probe or an invitation to clear errors and retry.
         */
        std::string memory_contract_error;
        if (!segment.capture->validateExecutableMemoryClass(
                graph.executableMemoryClass(), &memory_contract_error))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture memory contract: " << memory_contract_error);
            return false;
        }
        GraphUpdateResult update_result = GraphUpdateResult::NeedsReinstantiate;
        if (segment.capture->supportsExecutableUpdate())
        {
            update_result = segment.capture->tryUpdate();
            if (update_result == GraphUpdateResult::Failed)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Re-capture graph update failed; "
                    "refusing to discard the backend error and instantiate a "
                    "replacement executable");
                return false;
            }
        }
        if (update_result == GraphUpdateResult::NeedsReinstantiate)
        {
            if (!segment.capture->instantiate())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Re-capture instantiate failed");
                return false;
            }
        }

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticRecapture,
                capture_stream))
        {
            return false;
        }

        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Re-capture launch failed");
            return false;
        }

        post_launch_cb(segment, capture_stream);

        for (const auto &stage_name : segment.stage_names)
        {
            graph.markCompleted(stage_name);
        }

        return true;
    }

    DeviceGraphCaptureController::VerifyReplayResult DeviceGraphCaptureController::executeCapturedReplaySegmentVerify(
        ComputeGraph &graph,
        DeviceGraphExecutor::GraphSegment &segment,
        IDeviceContext *ctx,
        IWorkerGPUContext *gpu_ctx,
        void *capture_stream,
        bool needs_segment_sync,
        int segment_index,
        const DeviceGraphExecutor::GraphLaunchDependencyHook &launch_dependency_cb,
        const std::function<void(DeviceGraphExecutor::GraphSegment &, void *)> &post_launch_cb)
    {
        if(graph.readAheadBranch())throw std::logic_error("Read-ahead requires an unchanged complete captured parent; eager/manual/recapture verification is forbidden");
        VerifyReplayResult result{false, false};

        if (!ctx || !gpu_ctx)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify replay missing context");
            return result;
        }
        if (!segment.capture)
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify replay missing segment capture");
            return result;
        }

        if (segmentHasNonIdempotentStage(graph, segment))
        {
            // Verify mode compares "captured replay" against "direct execute".
            // Non-idempotent stages (e.g. residual accumulation) can legitimately
            // diverge on re-execution, so we skip comparison but still run replay.
            if (!applyGraphLaunchDependency(
                    launch_dependency_cb,
                    DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticVerification,
                    capture_stream))
            {
                return result;
            }
            if (!segment.capture->launch())
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify-skip: graph launch failed, seg " << segment_index);
                return result;
            }

            post_launch_cb(segment, capture_stream);

            if (needs_segment_sync)
            {
                if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
                {
                    LOG_ERROR("[DeviceGraphCaptureController] Verify-skip stream sync failed after seg " << segment_index);
                    return result;
                }
            }

            fprintf(stderr,
                    "[GRAPH_VERIFY] seg %d skipped (non-idempotent stage detected)\n",
                    segment_index);

            for (const auto &stage_name : segment.stage_names)
            {
                graph.markCompleted(stage_name);
            }

            result.success = true;
            result.skipped_non_idempotent = true;
            return result;
        }

        if (!applyGraphLaunchDependency(
                launch_dependency_cb,
                DeviceGraphExecutor::GraphExecutableLaunchPhase::DiagnosticVerification,
                capture_stream))
        {
            return result;
        }
        if (!segment.capture->launch())
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify: graph launch failed, seg " << segment_index);
            return result;
        }
        post_launch_cb(segment, capture_stream);
        if (!gpu_ctx->synchronizeStreamChecked(capture_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify stream sync failed after seg " << segment_index);
            return result;
        }

        void *default_stream = gpu_ctx->defaultStream();

        struct StageOutput
        {
            std::string name;
            float values[8] = {};
            size_t count = 0;
            bool has_gpu_ptr = false;
        };

        std::vector<StageOutput> graph_outputs(segment.stage_names.size());
        // Pass 1: collect outputs from captured replay launch.
        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            auto *node = graph.getNode(segment.stage_names[s]);
            graph_outputs[s].name = segment.stage_names[s];
            if (node && node->stage)
            {
                StageDumpInfo dump_info = node->stage->getDumpInfoSnapshot();
                if (!dump_info.outputs.empty())
                {
                    const auto &out = dump_info.outputs[0];
                    if (out.tensor)
                    {
                        auto *base = dynamic_cast<TensorBase *>(out.tensor);
                        if (base)
                        {
                            const void *gpu_ptr = base->gpu_data_ptr();
                            if (gpu_ptr)
                            {
                                graph_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                graph_outputs[s].has_gpu_ptr =
                                    ctx->copyToHost(graph_outputs[s].values, gpu_ptr,
                                                    graph_outputs[s].count * sizeof(float),
                                                    capture_stream);
                            }
                        }
                    }
                }
            }
        }

        for (const auto &stage_name : segment.stage_names)
        {
            // Pass 2: execute the same stages directly (no graph launch) to
            // produce a reference output for comparison.
            auto *node = graph.getNode(stage_name);
            if (!node || !node->stage)
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify: missing stage during direct exec: " << stage_name);
                return result;
            }
            node->stage->setGPUStream(default_stream);
            if (!node->stage->execute(ctx))
            {
                LOG_ERROR("[DeviceGraphCaptureController] Verify: direct exec failed: " << stage_name);
                return result;
            }
        }
        if (!gpu_ctx->synchronizeStreamChecked(default_stream))
        {
            LOG_ERROR("[DeviceGraphCaptureController] Verify direct-exec stream sync failed");
            return result;
        }

        std::vector<StageOutput> direct_outputs(segment.stage_names.size());
        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            auto *node = graph.getNode(segment.stage_names[s]);
            direct_outputs[s].name = segment.stage_names[s];
            if (node && node->stage)
            {
                StageDumpInfo dump_info = node->stage->getDumpInfoSnapshot();
                if (!dump_info.outputs.empty())
                {
                    const auto &out = dump_info.outputs[0];
                    if (out.tensor)
                    {
                        auto *base = dynamic_cast<TensorBase *>(out.tensor);
                        if (base)
                        {
                            const void *gpu_ptr = base->gpu_data_ptr();
                            if (gpu_ptr)
                            {
                                direct_outputs[s].count = std::min<size_t>(8, out.rows * out.cols);
                                direct_outputs[s].has_gpu_ptr =
                                    ctx->copyToHost(direct_outputs[s].values, gpu_ptr,
                                                    direct_outputs[s].count * sizeof(float),
                                                    default_stream);
                            }
                        }
                    }
                }
            }
        }

        FILE *f = fopen("/tmp/graph_phase3.log", "a");
        float seg_max_diff = 0;

        for (size_t s = 0; s < segment.stage_names.size(); s++)
        {
            if (!graph_outputs[s].has_gpu_ptr || !direct_outputs[s].has_gpu_ptr)
            {
                continue;
            }

            float max_diff = 0;
            size_t n = std::min(graph_outputs[s].count, direct_outputs[s].count);
            for (size_t i = 0; i < n; i++)
            {
                max_diff = std::max(max_diff, std::abs(graph_outputs[s].values[i] - direct_outputs[s].values[i]));
            }
            seg_max_diff = std::max(seg_max_diff, max_diff);

            char buf[1024];
            snprintf(buf, sizeof(buf),
                     "[STAGE_VERIFY] seg %d stage %zu/%zu (%s) max_diff=%.6e",
                     segment_index, s, segment.stage_names.size(), segment.stage_names[s].c_str(), max_diff);

            if (f)
            {
                fprintf(f, "%s\n", buf);
                if (max_diff > 1e-6f)
                {
                    fprintf(f, "  GRAPH : ");
                    for (size_t i = 0; i < n; i++)
                    {
                        fprintf(f, "%.6f%s", graph_outputs[s].values[i], i < n - 1 ? ", " : "");
                    }
                    fprintf(f, "\n  DIRECT: ");
                    for (size_t i = 0; i < n; i++)
                    {
                        fprintf(f, "%.6f%s", direct_outputs[s].values[i], i < n - 1 ? ", " : "");
                    }
                    fprintf(f, "\n");
                }
                fflush(f);
            }
        }

        {
            char buf[512];
            snprintf(buf, sizeof(buf),
                     "[GRAPH_VERIFY] seg %d (%zu stages, %zu nodes, last=%s) seg_max_diff=%.6e",
                     segment_index, segment.stage_names.size(), segment.capture->nodeCount(),
                     segment.stage_names.back().c_str(), seg_max_diff);
            fprintf(stderr, "%s\n", buf);
            if (f)
            {
                fprintf(f, "%s\n\n", buf);
                fflush(f);
            }
        }

        if (f)
        {
            fclose(f);
        }

        for (const auto &stage_name : segment.stage_names)
        {
            graph.markCompleted(stage_name);
        }

        result.success = true;
        return result;
    }

} // namespace llaminar2
