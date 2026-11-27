/**
 * @file Qwen35MoEGraphFFN.cpp
 * @brief Primary FFN graph construction entry point for Qwen35MoEGraph.
 */

#include "Qwen35MoEGraphInternal.h"

namespace llaminar2
{
    ComputeGraph Qwen35MoEGraph::buildFFNGraphForLayer(
        const LayerWeights &layer, ActivationBuffers &buffers,
        const LayerGraphSourceIdentity &identity, int seq_len, int batch_size,
        DeviceId device, void *device_state_publication_stream,
        const int32_t *sequence_lengths_device, const int32_t *absolute_position_ids_device,
        const Qwen4ExpPredictorResidualBinding *predictor)
    {
        const int layer_idx = identity.sourceLayer();
        if (seq_len <= 0 || batch_size <= 0 || batch_size > std::numeric_limits<int>::max() / seq_len)
            throw std::invalid_argument("FFN graph requires positive representable query geometry");
        if ((identity.sourcePolicy() == GraphLayerSourcePolicy::Qwen4ExpSharedPredictor) != (predictor != nullptr))
            throw std::invalid_argument("Shared QFN FFN identity requires its explicit frozen residual construction");
        if (predictor) {
            predictor->requireInvocation(Qwen4ExpPredictorInvocation::Draft);
            Qwen4ExpPredictorResidualBinding::requireFrozenSource(identity, predictor->weights());
            predictor->requireLayerSource(layer);
            predictor->rows().requireCompactCPUQuery();
            if (batch_size != 1 || predictor->rows().tokens().physicalRows() != seq_len ||
                predictor->rows().device() != device ||
                (device.is_gpu() && predictor->rows().tokens().countOwner() != sequence_lengths_device))
                throw std::invalid_argument("Dense0 FFN must retain its exact original row/count/device owner");
        }
        (void)projectionArena();
        if (device.is_gpu() && !device_state_publication_stream)
        {
            throw std::invalid_argument(
                "[Qwen35MoEGraph::buildFFNGraph] GPU graph construction "
                "requires an explicit non-null device-state publication stream");
        }

        const bool has_overlay_authority =
            config_.moe.expert_overlay_residency_authority != nullptr;
        if (has_overlay_authority !=
            config_.moe.usesExpertOverlayDurableResidencyAuthority())
        {
            throw std::logic_error(
                "Qwen35 MoE graph has an inconsistent ExpertOverlayRCU "
                "durable residency authority binding");
        }
        ++routed_layer_specialization_attempts_;
        ComputeGraph graph;
        const bool mtp_sidecar_context = identity.workload() == GraphLayerWorkload::MTPDepth;
        const int mtp_depth_idx = identity.predictorDepth().value_or(-1);
        const std::string prefix = identity.prefix();
        std::string ffn_terminal;
        int total_tokens = batch_size * seq_len;
        const bool main_condition =
            !mtp_sidecar_context && config_.live_mtp_request_batch_condition;
        // Resident MTP condition publishes the absolute target-cache length
        // for attention and the pipeline handoff. A scalar condition already
        // owns exactly one FFN activation. Verifier and predictor counts still
        // govern their captured padding, including empty replays.
        const int32_t *const routed_live_rows = main_condition && total_tokens == 1
            ? nullptr : sequence_lengths_device;
        /*
         * Every lowering of this routed layer—direct continuation compute,
         * retained remote GPU replay, and serial CPU service—shares one
         * semantic graph identity. Compute that identity once so the direct
         * fast path cannot silently classify a one-row MTP predictor as
         * ordinary decode while endpoint paths publish grouped-verifier
         * evidence.
         */
        const MoEOverlayServicePhaseHint routed_service_phase =
            mtp_sidecar_context || config_.grouped_mtp_verifier ||
                    config_.live_mtp_request_batch_condition
                ? MoEOverlayServicePhaseHint::GroupedVerifier
                : (total_tokens == 1
                       ? MoEOverlayServicePhaseHint::Decode
                       : MoEOverlayServicePhaseHint::Prefill);
        // Demand and measured service prices describe the same graph phase.
        // In particular, predictor M=1 is MTP work, not ordinary decode.
        const ExpertHistogramSource routed_histogram_source =
            routed_service_phase == MoEOverlayServicePhaseHint::GroupedVerifier
                ? ExpertHistogramSource::GroupedVerifier
                : routed_service_phase == MoEOverlayServicePhaseHint::Prefill
                    ? ExpertHistogramSource::PrefillChunk
                    : ExpertHistogramSource::DecodeToken;
        if (layer_idx == 0 && mirroredLayerDiagnosticsEnabled())
        {
            LOG_INFO(
                "[MirroredLayerDiagnostics] graph_build device="
                << device.toString()
                << " total_tokens=" << total_tokens
                << " mtp_sidecar=" << (mtp_sidecar_context ? "true" : "false")
                << " grouped_mtp_verifier="
                << (config_.grouped_mtp_verifier ? "true" : "false")
                << " all_position_logits="
                << (config_.compute_all_position_logits ? "true" : "false")
                << " live_request_batch_condition="
                << (config_.live_mtp_request_batch_condition ? "true" : "false")
                << " checkpoint_graph="
                << (!mtp_sidecar_context &&
                            !config_.grouped_mtp_verifier
                        ? "true"
                        : "false"));
        }
        auto overlay_runtime_plan = routedForwardRecipe(config_.default_device).runtime;
        const auto &overlay_plan = overlay_runtime_plan
                                       ? overlay_runtime_plan->sourcePlanPtr()
                                       : config_.moe.routed_expert_plan;
        auto domainContainsDevice = [](const MoEOverlayRuntimeDomain &domain, DeviceId candidate)
        {
            return std::any_of(domain.participants.begin(), domain.participants.end(),
                               [&](const MoEOverlayDomainParticipant &participant)
                               {
                                   return participant.local_device == candidate;
                               });
        };

        if (overlay_runtime_plan)
        {
            const auto &continuation_domain = overlay_runtime_plan->continuationDomain();
            DeviceId continuation_device = overlay_runtime_plan->continuationDevice();
            if (continuation_device.is_valid() && continuation_device != device &&
                !domainContainsDevice(continuation_domain, device))
            {
                LOG_DEBUG("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " using MoE overlay continuation_domain root device "
                                                    << continuation_device.to_string()
                                                    << " instead of caller device " << device.to_string());
                device = continuation_device;
            }
        }

        const bool overlay_requested =
            overlay_plan && overlay_plan->usesExpertOverlayAuthority();
        const RoutedExpertLayerPlacement *overlay_placement = overlay_requested
                                                            ? findExpertOverlayPlacement(*overlay_plan, layer_idx)
                                                            : nullptr;
        const bool use_expert_overlay = overlay_requested && overlay_placement &&
                                        isUsableExpertOverlayPlacement(*overlay_placement,
                                                                       *overlay_plan,
                                                                       config_.moe.num_experts,
                                                                       layer_idx);

        if (overlay_requested && !use_expert_overlay)
        {
            throw std::runtime_error(
                "Qwen35 MoE expert overlay was requested but no usable placement exists for layer " +
                std::to_string(layer_idx) +
                "; refusing to lower the request through the non-overlay routed path");
        }

        const bool mapped_intermediate_shards = intermediateShardArena() && overlay_plan &&
            hasMappedGPUCPUIntermediateShardTopology(*overlay_plan);
        const auto mapped_shard_phase = mapped_intermediate_shards &&
                !mtp_sidecar_context && !config_.grouped_mtp_verifier && !config_.live_mtp_request_batch_condition &&
                forward_execution_phase_ == ForwardExecutionPhase::Prefill
            ? MoEOverlayServicePhaseHint::Prefill : routed_service_phase;
        const bool mapped_shard_grouped = mapped_intermediate_shards &&
            (total_tokens != 1 || mapped_shard_phase != MoEOverlayServicePhaseHint::Decode);
        const bool cuda_apportioned_retained_m1 = config_.flash_next &&
            device.is_cuda() && !mapped_intermediate_shards && use_expert_overlay &&
            overlay_plan && hasCUDAMappedRetainedM1OverlayTopology(*overlay_plan) &&
            total_tokens == 1 && !mtp_sidecar_context &&
            !config_.grouped_mtp_verifier && !config_.live_mtp_request_batch_condition &&
            forward_input_construction_ &&
            useQwen4ExpRetainedM1HCFFN(qfnCUDAMappedRetainedM1HCFFNCandidate(),
                forward_input_construction_->execution_phase, forward_input_construction_->execution_role,
                total_tokens, batch_size);
        const bool mapped_cuda_retained_m1 = cuda_apportioned_retained_m1 || (config_.flash_next &&
            mapped_intermediate_shards && !mapped_shard_grouped && !mtp_sidecar_context &&
            forward_input_construction_ &&
            ((device.is_cuda() && useQwen4ExpRetainedM1HCFFN(qfnCUDAMappedRetainedM1HCFFNCandidate(),
                forward_input_construction_->execution_phase, forward_input_construction_->execution_role,
                total_tokens, batch_size)) ||
             (device.is_rocm() && overlay_plan->usesGFX906MI50Kernels() &&
                useQwen4ExpRetainedM1HCFFN(qfnROCmMappedRetainedM1HCFFNCandidate(),
                forward_input_construction_->execution_phase, forward_input_construction_->execution_role,
                total_tokens, batch_size))));
        const bool mapped_decode_shared_overlap = [&] {
            if (!config_.flash_next || !mapped_intermediate_shards || mapped_shard_grouped || mapped_cuda_retained_m1)
                return false;
            // Both measured native backends use the same graph ordering at
            // the existing heterogeneous ticket boundary.
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                const auto &hardware = cudaHardwareMetadata().at(device.ordinal);
                return hardware.sm_major == 8 && hardware.sm_minor == 6;
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                const auto &target = rocmHardwareMetadata().at(device.ordinal).isa_target;
                return target == "gfx906" || target.starts_with("gfx906:");
            }
#endif
            return false;
        }();
        if (intermediateShardArena() && !mapped_intermediate_shards)
        {
            if (!use_expert_overlay || batch_size != 1)
                throw std::invalid_argument("GPU F-shard model lowering requires its original overlay and one device-owned row prefix; request-major rows are not wired");
            auto phase = routed_service_phase;
            if (!mtp_sidecar_context && !config_.grouped_mtp_verifier && !config_.live_mtp_request_batch_condition)
            {
                if (!forward_execution_phase_)
                    throw std::invalid_argument("GPU F-shard model lowering requires an explicit forward arithmetic phase");
                phase = *forward_execution_phase_ == ForwardExecutionPhase::Prefill
                    ? MoEOverlayServicePhaseHint::Prefill : MoEOverlayServicePhaseHint::Decode;
            }
            if (main_condition &&
                (total_tokens != 1 || forward_execution_phase_ != ForwardExecutionPhase::Decode ||
                 config_.grouped_mtp_verifier))
                throw std::invalid_argument("GPU F-shard MTP condition requires one fixed main decode row");
            return buildIntermediateShardFFNGraphForLayer(layer, buffers, identity, total_tokens, device,
                device_state_publication_stream, routed_live_rows, absolute_position_ids_device, predictor, phase);
        }

        const bool cpu_homogeneous_intermediate_shards = [&]() -> bool {
            if (!device.is_cpu() || !use_expert_overlay || !overlay_plan || !layer.moe_gate || batch_size != 1)
                return false;
            if (overlay_plan->domains.size() != 1u || overlay_plan->routed_tiers.size() != 1u)
                return false;
            const auto &domain = overlay_plan->domains.front();
            if (domain.routed_compute_policy != RoutedExpertComputePolicy::TensorSharded ||
                domain.participants.size() < 2u)
                return false;
            if (overlay_plan->continuation_domain != domain.name ||
                overlay_plan->routed_tiers.front().domain != domain.name ||
                overlay_plan->shared_expert_domain != domain.name)
                return false;
            if (!std::all_of(domain.participants.begin(), domain.participants.end(),
                    [](const GlobalDeviceAddress &participant) { return participant.isCPU(); }))
                return false;
            auto *gtp = dynamic_cast<IGlobalTPContext *>(config_.tp_ctx);
            if (!gtp || gtp->degree() != static_cast<int>(domain.participants.size()))
                return false;
            const auto &authority = config_.moe.expert_overlay_residency_authority;
            const auto &residency = config_.moe.expert_overlay_participant_residency;
            if (!residency || !prepared_weight_store_)
                return false;
            const bool stationary =
                config_.moe.rebalance_config.mode == MoERebalanceRuntimeMode::Off ||
                (config_.moe.rebalance_config.mode == MoERebalanceRuntimeMode::Dynamic &&
                 authority && (authority->stationaryTensorShardResidency() || authority->noPeerResidency()));
            if (!stationary)
                return false;
            return true;
        }();
        if (cpu_homogeneous_intermediate_shards)
        {
            auto phase = routed_service_phase;
            if (!mtp_sidecar_context && !config_.grouped_mtp_verifier && !config_.live_mtp_request_batch_condition &&
                forward_execution_phase_)
            {
                phase = *forward_execution_phase_ == ForwardExecutionPhase::Prefill
                    ? MoEOverlayServicePhaseHint::Prefill : MoEOverlayServicePhaseHint::Decode;
            }
            return buildCPUIntermediateShardFFNGraphForLayer(
                layer, buffers, identity, total_tokens, device, predictor, phase);
        }

        /*
         * Tiered ExpertOverlay routes every expert parent through the prepared
         * participant registry. A continuation rank may own no local expert
         * slice at all, so a raw 3-D parent is intentionally absent from its
         * model binding view. The router remains a normal immutable tensor;
         * only the routed GEMM parents are registry-owned.
         */
        if (!layer.moe_gate || (!layer.moe_gate_exps && !use_expert_overlay && !device.is_cpu()))
        {
            return Qwen35Graph::buildFFNGraph(
                layer,
                buffers,
                layer_idx,
                seq_len,
                batch_size,
                device,
                device_state_publication_stream,
                sequence_lengths_device,
                absolute_position_ids_device);
        }

        /*
         * Verifier batches are tiny (draft depth + bonus row), but the main-model
         * verifier is stateful: its row logits must match serial decode and the
         * accepted row may later publish KV/GDN/conv state.  The verifier path
         * treats M=1 as a real verifier bucket, but M=1 has no batching economy:
         * it uses the explicit decode-equivalent oracle while multi-row GPU verifier batches use
         * the grouped-prefill machinery.  The router still emits serial-decode
         * top-k rows, and the shared expert uses GEMV verifier-row hooks rather
         * than the older MoE grouped prefill helper.
         */
        auto forceGpuSmallMMoESidecarPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   mtp_sidecar_context;
        };
        auto forceGpuSmallMMainVerifierPrefill = [&](DeviceId candidate)
        {
            return (candidate.is_cuda() || candidate.is_rocm()) &&
                   total_tokens > 1 &&
                   config_.grouped_mtp_verifier &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceGroupedMoEVerifierPrefill = [&](DeviceId candidate)
        {
            return forceGpuSmallMMoESidecarPrefill(candidate) ||
                   forceGpuSmallMMainVerifierPrefill(candidate);
        };
        auto forceDecodeEquivalentMoERouting = [&](DeviceId candidate)
        {
            /*
             * The main verifier compares grouped all-position rows against
             * serial decode rows.  GPU M=1 has no grouped economy to recover,
             * so it stays on ordinary production decode routing. Multi-row GPU verifier batches
             * still uses the decode-equivalent grouped router so the economical
             * expert kernels receive exactly the serial top-k rows.
            */
            const bool gpu_grouped_verifier_rows =
                (candidate.is_cuda() || candidate.is_rocm()) && total_tokens > 1;
            return (candidate.is_cpu() || gpu_grouped_verifier_rows) &&
                   total_tokens >= 1 &&
                   config_.grouped_mtp_verifier &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        auto forceDecodeEquivalentMoEVerifier = [&](DeviceId candidate)
        {
            return candidate.is_cpu() &&
                   total_tokens >= 1 &&
                   config_.grouped_mtp_verifier &&
                   config_.compute_all_position_logits &&
                   !mtp_sidecar_context;
        };
        LayerWeightBindings layer_bindings = predictor ? predictor->weights().fa_block
                                                       : layerWeightBindingsForGraph(layer_idx);

        IMoERuntimeTable *moe_runtime_table = nullptr;
        const auto &rocm_env = debugEnv().rocm;
        /*
         * Grouped MoE verifier rows are now a required production path rather
         * than an advertised optional capability.  Keep backend/shape preflight
         * in the stages, but do not let the old grouped-prefill debug kill switch
         * route publishable verifier rows through a non-grouped substitute.
         */
        auto forceGroupedSharedMoEVerifierPrefill = [&](DeviceId candidate)
        {
            return forceGroupedMoEVerifierPrefill(candidate);
        };
        /*
         * MTP sidecars need their own persistent MoE metadata even when their
         * logical layer index aliases a main-model layer.  The runtime table
         * stores device-resident top-k route IDs, route weights, placement
         * banks, and prefill scratch pointers that graph replay can capture by
         * address.  Sharing those slots with ordinary decode would make an
         * accepted-state publication boundary able to invalidate a warm sidecar
         * graph without changing any graph shape.  Depth-scoped tables match the
         * vLLM-style contract: sidecar metadata is stable, persistent, and owned
         * by the sidecar graph fragment.
         */
        const bool use_mtp_runtime_table = mtp_sidecar_context;
        if (layer_idx < config_.pp_layer_offset || layer_idx == std::numeric_limits<int>::max())
            throw std::invalid_argument("Qwen35 MoE graph layer is outside its runtime stage interval");
        const int runtime_table_layers = use_mtp_runtime_table
                                             ? std::max(config_.n_layers, layer_idx - config_.pp_layer_offset + 1)
                                             : config_.n_layers;
        /*
         * Static placement has no histogram consumer. Keep its captured
         * router numerically identical while omitting the global counter
         * atomics and producer-stream publication that only Observe/Dynamic
         * maintenance can use. Device-resident Dynamic remains enabled here:
         * it collects evidence on device even though it deliberately does not
         * register a host drain callback below.
         */
        const bool collect_runtime_histogram =
            !use_mtp_runtime_table &&
            config_.moe.rebalance_config.mode !=
                MoERebalanceRuntimeMode::Off;
        const bool register_runtime_histogram = collect_runtime_histogram;
        const bool local_decode_layer =
            !mtp_sidecar_context &&
            total_tokens == 1 &&
            layer_idx >= config_.pp_layer_offset &&
            layer_idx < config_.pp_layer_offset + config_.n_layers &&
            !config_.compute_all_position_logits;
        /*
         * MTP may execute the main model exclusively through a grouped
         * all-position verifier.  That graph is not prefill: each row is a
         * serial-decode-equivalent candidate and its route kernels own the
         * production decode histograms.  Treat it as a decode maintenance
         * producer so the device-side controller never waits for an M=1 graph
         * that this request is not required to build.
         */
        const bool grouped_main_verifier_layer =
            !mtp_sidecar_context &&
            total_tokens > 1 &&
            config_.grouped_mtp_verifier &&
            config_.compute_all_position_logits &&
            layer_idx >= config_.pp_layer_offset &&
            layer_idx < config_.pp_layer_offset + config_.n_layers;
        const RoutedExpertTier *resolved_local_tp_replicated_tier = nullptr;
        const bool resolved_local_tp_replicated_fast_candidate =
            use_expert_overlay &&
            canUseLocalTPReplicatedFastPath(
                *overlay_plan,
                device,
                &resolved_local_tp_replicated_tier);
        const bool ordinary_prefill_graph =
            total_tokens > 1 &&
            !mtp_sidecar_context &&
            !config_.compute_all_position_logits;
        /* Grouped MTP verification and sidecars are decode work even when
         * their graph has multiple rows. Resolve transport from semantics,
         * never from tensor shape alone. */
        const MoEOverlayNodeLocalRoutePhase node_local_route_phase =
            ordinary_prefill_graph
                ? MoEOverlayNodeLocalRoutePhase::OrdinaryPrefill
                : MoEOverlayNodeLocalRoutePhase::Decode;
        const MoEOverlayNodeLocalRouteTransport node_local_route_transport =
            config_.moe.node_local_route_transport_policy.forPhase(
                node_local_route_phase);
        const bool phase_split_local_tp_apportioned_gpu_prefill =
            resolved_local_tp_replicated_fast_candidate &&
            resolved_local_tp_replicated_tier &&
            ordinary_prefill_graph &&
            isPrefillApportionedDecodeReplicatedTier(
                *overlay_plan,
                *resolved_local_tp_replicated_tier);
        // A singleton overlay is a complete local native computation, not a
        // one-member collective. Authenticate the original immutable owner map
        // before reusing the prepared local-expert lowering. GPU route state
        // remains in the overlay epoch/runtime table; there is no host ticket.
        const bool single_participant_overlay_local = [&]()
        {
            if (!device.is_gpu() || !use_expert_overlay || !overlay_plan ||
                !MoEOverlayNoPeerResidency::hasNoPeerTopology(*overlay_plan))
                return false;
            const auto &authority = config_.moe.expert_overlay_residency_authority;
            if (!authority || !authority->noPeerResidency())
                throw std::logic_error("Qwen35 MoE singleton native lowering requires its sealed no-peer residency authority");
            const auto &proof = *authority->noPeerResidency();
            const auto snapshot = authority->snapshot();
            if (!snapshot || !snapshot->valid())
                throw std::logic_error("Qwen35 MoE singleton native lowering has no original residency snapshot");
            const auto &original = *snapshot->placement_plan;
            if (!MoEOverlayNoPeerResidency::hasNoPeerTopology(original))
                throw std::logic_error("Qwen35 MoE singleton native lowering lost its original topology");
            const auto &domain = overlay_plan->domains.front();
            const auto &original_domain = original.domains.front();
            if (domain.name != original_domain.name || domain.scope != original_domain.scope ||
                domain.backend != original_domain.backend || domain.participants != original_domain.participants ||
                domain.world_ranks != original_domain.world_ranks || domain.owner_rank != original_domain.owner_rank ||
                domain.routed_compute_policy != original_domain.routed_compute_policy ||
                domain.routed_phase_policy != original_domain.routed_phase_policy ||
                domain.routed_decode_assignment_policy != original_domain.routed_decode_assignment_policy ||
                domain.routed_prefill_assignment_policy != original_domain.routed_prefill_assignment_policy ||
                domain.weights != original_domain.weights ||
                overlay_plan->authority_execution != original.authority_execution ||
                overlay_plan->residency_policy != original.residency_policy ||
                overlay_plan->owner_order != original.owner_order ||
                overlay_plan->routed_tiers.front().name != original.routed_tiers.front().name ||
                overlay_plan->routed_tiers.front().domain != original.routed_tiers.front().domain ||
                !std::equal(overlay_plan->placements.begin(), overlay_plan->placements.end(),
                    original.placements.begin(), original.placements.end(), [](const auto &a, const auto &b)
                    { return a.layer == b.layer && a.routed_expert_tier == b.routed_expert_tier; }))
                throw std::logic_error("Qwen35 MoE singleton native lowering has foreign domain, participant or placement identity");
            const auto &ownership = snapshot->layered_ownership;
            proof.requireDestination(device, ownership.firstModelLayer(),
                ownership.layerCount(), config_.moe.num_experts);
            if (overlay_plan->first_model_layer != ownership.firstModelLayer() ||
                overlay_plan->placements.size() != static_cast<std::size_t>(ownership.layerCount()) ||
                layer_idx < ownership.firstModelLayer() ||
                layer_idx - ownership.firstModelLayer() >= ownership.layerCount() ||
                overlay_plan->continuation_domain != overlay_plan->routed_tiers.front().domain ||
                overlay_plan->shared_expert_domain != overlay_plan->routed_tiers.front().domain ||
                overlay_plan->domains.front().primaryWorldRank() != std::optional<int>{proof.worldRank()} ||
                (config_.moe.overlay_mpi_ctx && config_.moe.overlay_mpi_ctx->rank() != proof.worldRank()) ||
                proof.authorityExecution() != config_.moe.authority_execution)
                throw std::logic_error("Qwen35 MoE singleton native lowering differs from its complete admitted stage ownership");
            return true;
        }();
        const bool resolved_fully_replicated_local_rows =
            single_participant_overlay_local ||
            (resolved_local_tp_replicated_fast_candidate &&
             !phase_split_local_tp_apportioned_gpu_prefill) ||
            (!use_expert_overlay &&
             config_.moe.routed_compute_policy ==
                 RoutedExpertComputePolicy::Replicated);
        const RoutedExpertRowExecutionPolicy routed_row_execution_policy =
            mapped_intermediate_shards ? RoutedExpertRowExecutionPolicy::IntermediateShardLocal
                : resolved_fully_replicated_local_rows
                    ? RoutedExpertRowExecutionPolicy::FullyReplicatedLocal
                    : RoutedExpertRowExecutionPolicy::ParticipantAssigned;
        const bool device_rebalance_decode_layer =
            local_decode_layer || grouped_main_verifier_layer;
        const bool first_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset;
        const bool last_device_rebalance_decode_layer =
            device_rebalance_decode_layer &&
            layer_idx == config_.pp_layer_offset + config_.n_layers - 1;
        auto *local_tp_ctx = dynamic_cast<ILocalTPContext *>(config_.tp_ctx);
        auto *global_tp_ctx = dynamic_cast<IGlobalTPContext *>(config_.tp_ctx);
        const bool rank_local_expert_endpoints = use_expert_overlay && config_.moe.overlay_mpi_ctx &&
            MoEOverlayHostServiceContract::resolveAdmitted(*overlay_plan,
                *routedForwardRecipe(device).owners, *config_.moe.overlay_mpi_ctx)
                .requiresRankLocalInferencePublisher();
        const bool distributed_expert_overlay = use_expert_overlay && config_.moe.overlay_mpi_ctx &&
            (config_.moe.overlay_mpi_ctx->world_size() > 1 || rank_local_expert_endpoints);
        const RoutedContinuationTopology routed_continuation_topology =
            use_expert_overlay ? routedForwardRecipe(device).continuation : RoutedContinuationTopology{};
        const bool captured_overlay_continuation =
            routed_continuation_topology
                .captured_continuation_device;
        const bool captured_local_tp_continuation =
            routed_continuation_topology
                .captured_local_tp;
        if (mapped_intermediate_shards && (!use_expert_overlay || batch_size != 1 ||
            !captured_overlay_continuation || !captured_local_tp_continuation || !local_tp_ctx ||
            config_.moe.authority_execution != MoEOverlayAuthorityExecutionKind::HostResident ||
            node_local_route_transport != MoEOverlayNodeLocalRouteTransport::MappedSparse))
            throw std::invalid_argument("Mapped intermediate shards require the original hosted multi-tier LocalTP continuation");
        int hot_replica_cap = config_.moe.hot_expert_cache.resolveCap(
            config_.moe.num_experts,
            /*dynamic_rebalance_enabled=*/true);
        const auto &env = debugEnv();
        const bool prefix_runtime_device_rehydration =
            prefixRuntimeRehydrationGraphActive() &&
            !mtp_sidecar_context &&
            device.is_gpu();
        if (prefix_runtime_device_rehydration &&
            layer_idx == config_.pp_layer_offset)
        {
            LOG_INFO("[Qwen35MoEGraph] Building dedicated prefix-runtime "
                     "device rehydration graph"
                     << " device=" << device.to_string()
                     << " total_tokens=" << total_tokens
                     << " first_layer=" << layer_idx);
        }
        const bool llep_prefill_requested =
            ordinary_prefill_graph &&
            routed_row_execution_policy ==
                RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
            config_.moe.routed_prefill_assignment_policy ==
                RoutedExpertAssignmentPolicy::LeastLoadedResident;
        const uint64_t llep_prefill_routed_rows =
            llep_prefill_requested
                ? static_cast<uint64_t>(std::max(0, total_tokens)) *
                      static_cast<uint64_t>(std::max(0, config_.moe.top_k))
                : 0ULL;
        const uint64_t llep_prefill_min_routed_rows =
            config_.moe.routed_prefill_config
                .least_loaded_min_routed_rows;
        const bool llep_prefill_cost_gate_passed =
            !llep_prefill_requested ||
            llep_prefill_min_routed_rows == 0ULL ||
            llep_prefill_routed_rows >= llep_prefill_min_routed_rows;
        const RoutedExpertDomain *cpu_llep_overlay_domain =
            use_expert_overlay && overlay_plan &&
                    overlay_plan->routed_tiers.size() == 1u
                ? expertDomainForTier(
                      *overlay_plan,
                      overlay_plan->routed_tiers.front())
                : nullptr;
        const bool cpu_llep_overlay_topology_supported =
            cpu_llep_overlay_domain != nullptr &&
            overlay_plan->routed_tiers.front().domain ==
                overlay_plan->continuation_domain &&
            static_cast<int>(cpu_llep_overlay_domain->participants.size()) ==
                (global_tp_ctx ? global_tp_ctx->degree() : 0) &&
            std::all_of(
                cpu_llep_overlay_domain->participants.begin(),
                cpu_llep_overlay_domain->participants.end(),
                [](const GlobalDeviceAddress &participant)
                { return participant.isCPU(); });
        const bool cpu_llep_prefill_transport_supported =
            llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            device.is_cpu() &&
            use_expert_overlay &&
            cpu_llep_overlay_topology_supported &&
            config_.moe.expert_overlay_residency_authority != nullptr &&
            config_.moe.expert_overlay_participant_residency != nullptr &&
            config_.moe.overlay_mpi_ctx != nullptr &&
            config_.moe.overlay_mpi_ctx->world_size() > 1 &&
            global_tp_ctx != nullptr &&
            global_tp_ctx->degree() > 1 &&
            global_tp_ctx->myIndex() >= 0 &&
            global_tp_ctx->myIndex() < global_tp_ctx->degree() &&
            cpu_current_batch_llep_executor_ != nullptr;
        const bool gpu_llep_prefill_candidate =
            llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            local_tp_ctx != nullptr &&
            device.is_gpu();
        const bool llep_prefill_enabled =
            cpu_llep_prefill_transport_supported ||
            gpu_llep_prefill_candidate;
        const bool llep_prefill_transport_supported =
            gpu_llep_prefill_candidate &&
            local_tp_ctx != nullptr &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        const bool prefix_runtime_rehydration_transport_supported =
            prefix_runtime_device_rehydration &&
            local_tp_ctx &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        if (gpu_llep_prefill_candidate &&
            !llep_prefill_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE least-loaded prefill requires "
                "a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string());
        }
        if (llep_prefill_requested &&
            llep_prefill_cost_gate_passed &&
            !llep_prefill_enabled)
        {
            throw std::runtime_error(
                "Qwen35 MoE least-loaded prefill has no production "
                "transaction authority for " + device.to_string());
        }
        if (prefix_runtime_device_rehydration &&
            !prefix_runtime_rehydration_transport_supported)
        {
            throw std::runtime_error(
                "Qwen35 MoE prefix-runtime payload rehydration requires a homogeneous graph-capturable NCCL/RCCL LocalTP domain for " +
                device.to_string());
        }
        const RoutedExpertAssignmentPolicy prefill_routed_expert_assignment_policy =
            (routed_row_execution_policy ==
                 RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
             total_tokens > 1 &&
             !llep_prefill_enabled)
                ? RoutedExpertAssignmentPolicy::StaticOwner
                : (grouped_main_verifier_layer || total_tokens == 1
                       ? config_.moe.routed_decode_assignment_policy
                       : config_.moe.routed_prefill_assignment_policy);
        if (env.presence.has("LLAMINAR_MOE_REBALANCE_REPLICAS"))
            hot_replica_cap = std::max(0, env.moe_rebalance.max_replicas);
        hot_replica_cap = std::min(hot_replica_cap, config_.moe.num_experts);
        // The replica request is only an upper bound. Consume the immutable
        // physical-admission grant, including after retained-runner reuse;
        // graph construction must not reserve the original larger directory.
        if (overlay_plan && overlay_plan->replica_cache_capacity)
            hot_replica_cap = overlay_plan->replica_cache_capacity->resolve(hot_replica_cap);
        const bool dynamic_overlay_residency =
            use_expert_overlay &&
            config_.moe.routed_expert_plan &&
            config_.moe.rebalance_config.mode ==
                MoERebalanceRuntimeMode::Dynamic;
        const bool device_resident_authority =
            dynamic_overlay_residency &&
            config_.moe.usesExpertOverlayDurableResidencyAuthority() &&
            config_.moe.authority_execution ==
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident;
        std::size_t authority_participant_count = 0u;
        if (device_resident_authority && overlay_plan &&
            overlay_plan->routed_tiers.size() == 1u)
        {
            const std::string &authority_domain =
                overlay_plan->routed_tiers.front().domain;
            const auto domain = std::find_if(
                overlay_plan->domains.begin(),
                overlay_plan->domains.end(),
                [&](const RoutedExpertDomain &candidate)
                {
                    return candidate.name == authority_domain;
                });
            if (domain != overlay_plan->domains.end())
                authority_participant_count = domain->participants.size();
        }
        /*
         * A mapped topology-wide controller is the sole authority across all
         * homogeneous groups.  The older domain-local NCCL/RCCL controller is
         * retained only for the single-group special case; composing both
         * would create two policy writers for the same runtime table.  Mapped
         * participants still register their device histograms below so the
         * topology-wide authority consumes the real production route counts.
         */
        const bool device_side_graph_rebalance_candidate =
            device_resident_authority &&
            !config_.moe.device_controller_fabric &&
            device.is_gpu() &&
            local_tp_ctx &&
            isHomogeneousGpuLocalTPRebalanceDomain(
                *local_tp_ctx,
                device,
                config_.tp_device_idx);
        if (device_resident_authority &&
            !config_.moe.device_controller_fabric &&
            device.is_gpu() &&
            device_rebalance_decode_layer &&
            authority_participant_count > 1u &&
            !device_side_graph_rebalance_candidate)
        {
            throw std::runtime_error(
                "Qwen35 MoE single-domain native device-resident ExpertOverlay authority requires one graph-capturable NCCL/RCCL LocalTP maintenance family on " +
                device.to_string());
        }
        /*
         * Current-batch migration is an amortized long-prefill policy. Grouped
         * MTP verifier rows instead split work only across the active bank's
         * resident participants; importing a multi-megabyte expert payload for
         * a handful of speculative rows is categorically uneconomical.
         *
         * Prefix restore remains an independent transport transaction. A
         * verifier graph may therefore own compact transport resources for
         * prefix rehydration while its current-batch assignment mode remains
         * resident-only.
         */
        const bool current_batch_llep_transfer_candidate =
            llep_prefill_transport_supported;
        const bool prefill_llep_transfer_candidate =
            current_batch_llep_transfer_candidate ||
            prefix_runtime_rehydration_transport_supported;
        const GraphSideRebalanceBindingRole prefill_transfer_binding_role =
            current_batch_llep_transfer_candidate
                ? GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer
                : GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer;
        const bool graph_rebalance_transport_candidate =
            device_side_graph_rebalance_candidate ||
            prefill_llep_transfer_candidate;
        const bool register_runtime_histogram_for_decode =
            register_runtime_histogram &&
            !device_side_graph_rebalance_candidate;
        const MoERuntimeHistogramWorkload runtime_histogram_workload =
            local_decode_layer
                ? MoERuntimeHistogramWorkload::SerialDecode
                : grouped_main_verifier_layer
                      ? MoERuntimeHistogramWorkload::GroupedMainVerifier
                      : MoERuntimeHistogramWorkload::NonDecode;
        const MoERuntimeHistogramProducerRole runtime_histogram_producer_role =
            selectMoERuntimeHistogramProducerRole(
                runtime_histogram_workload,
                register_runtime_histogram_for_decode);
        const auto has_static_full_local_expert_ownership = [&]()
        {
            /*
             * Device-routed one-token MoE decode captures a full layer-local
             * runtime table: router output, local-compute mask, and prepared
             * gate/up/down descriptors for every logical expert.  LocalTP
             * Expert-id-apportioned runners own only a contiguous expert range, so
             * handing them this table would make the expert stage fail at graph
             * build or, worse, capture a single-device contract for a sharded
             * topology. Tiered overlays and graph-side rebalance use masked
             * runtime tables instead; the full table is only for full-owner
             * lanes.
             */
            if (use_expert_overlay)
                return false;

            if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
            {
                if (config_.moe.owner_participant_count > 1)
                {
                    return false;
                }
            }

            if (device.is_gpu() && debugEnv().moe_rebalance.gpu_cache_experts_per_layer > 0)
                return false;

            return true;
        };
        const bool static_full_local_expert_ownership =
            has_static_full_local_expert_ownership();
        const bool masked_local_tp_overlay_decode_runtime_table =
            device.is_gpu() &&
            (total_tokens == 1 || forceGroupedMoEVerifierPrefill(device)) &&
            use_expert_overlay &&
            overlay_plan &&
            canUseLocalTPExpertIdApportionedFastPath(*overlay_plan, device);
        /**
         * @brief Invocation that owns a captured heterogeneous route ledger.
         *
         * The mapped sparse reducer consumes one final domain-local
         * participant for every original router slot.  That publication is a
         * transport invariant, not a Dynamic-only optimization: Static must
         * publish its fixed owner just as Dynamic publishes the owner selected
         * from the request-pinned placement epoch.  Giving each captured shape
         * an explicit role prevents an ordinary prefill graph from binding the
         * ledger pointer while accidentally skipping its producer.
         */
        enum class CapturedOverlayRouteLedgerWorkload : std::uint8_t
        {
            Unbound,
            SerialDecode,
            GroupedVerifier,
            OrdinaryPrefill,
        };

        CapturedOverlayRouteLedgerWorkload
            captured_overlay_route_ledger_workload =
                CapturedOverlayRouteLedgerWorkload::Unbound;
        if (device.is_gpu() && captured_overlay_continuation)
        {
            if (total_tokens == 1 && !mapped_shard_grouped)
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::SerialDecode;
            }
            else if (mapped_intermediate_shards
                        ? mapped_shard_phase != MoEOverlayServicePhaseHint::Prefill
                        : forceGroupedMoEVerifierPrefill(device))
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::GroupedVerifier;
            }
            else
            {
                captured_overlay_route_ledger_workload =
                    CapturedOverlayRouteLedgerWorkload::OrdinaryPrefill;
            }
        }
        const bool captured_distributed_overlay_runtime_table =
            captured_overlay_route_ledger_workload !=
            CapturedOverlayRouteLedgerWorkload::Unbound;
        const bool captured_overlay_route_ledger_uses_grouped_publication =
            captured_overlay_route_ledger_workload ==
                CapturedOverlayRouteLedgerWorkload::GroupedVerifier ||
            captured_overlay_route_ledger_workload ==
                CapturedOverlayRouteLedgerWorkload::OrdinaryPrefill;
        const MoERuntimeRouteWeightProjection
            captured_overlay_route_weight_projection =
                captured_overlay_route_ledger_workload ==
                        CapturedOverlayRouteLedgerWorkload::SerialDecode
                    ? MoERuntimeRouteWeightProjection::DecodeTopK
                : captured_overlay_route_ledger_uses_grouped_publication
                    ? MoERuntimeRouteWeightProjection::GroupedRouteSlots
                    : MoERuntimeRouteWeightProjection::Unspecified;
        /*
         * A retained heterogeneous Dynamic graph can receive a completely new
         * expert descriptor through the durable ExpertOverlay controller even
         * though it does not use the homogeneous graph-local rebalance
         * transport.  Its capture-time descriptor table is therefore only a
         * workspace seed, never an inference authority.  Decode and grouped
         * verification must materialize descriptors from the request-pinned
         * runtime bank after every admitted epoch, or a correctly promoted
         * expert remains invisible to the continuation-local GEMM.
         */
        const bool dynamic_distributed_overlay_uses_mutable_descriptors =
            dynamic_overlay_residency &&
            captured_distributed_overlay_runtime_table;
        const bool full_local_tp_replicated_overlay_decode_runtime_table =
            device.is_gpu() &&
            (total_tokens == 1 || forceGroupedMoEVerifierPrefill(device)) &&
            use_expert_overlay &&
            overlay_plan &&
            canUseLocalTPReplicatedFastPath(*overlay_plan, device);
        const bool masked_local_tp_apportioned_decode_runtime_table =
            (local_decode_layer ||
             grouped_main_verifier_layer ||
             (mtp_sidecar_context && total_tokens == 1)) &&
            device.is_gpu() &&
            !use_expert_overlay &&
            config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned &&
            config_.moe.owner_participant_count > 1 &&
            local_tp_ctx &&
            local_tp_ctx->degree() > 1;
        const bool runtime_table_eligible =
            mapped_intermediate_shards || projection_arena_.has_value() || static_full_local_expert_ownership ||
            single_participant_overlay_local ||
            masked_local_tp_overlay_decode_runtime_table ||
            captured_distributed_overlay_runtime_table ||
            full_local_tp_replicated_overlay_decode_runtime_table ||
            masked_local_tp_apportioned_decode_runtime_table;
        const MoERuntimeTableIdentity runtime_table_identity{
            .role = use_mtp_runtime_table
                        ? MoERuntimeTableRole::MTPDepth
                        : (current_batch_llep_transfer_candidate
                               ? MoERuntimeTableRole::CurrentBatchLLEPPrefill
                               : MoERuntimeTableRole::MainDecodeDurablePlacement),
            .mtp_depth = use_mtp_runtime_table ? mtp_depth_idx : -1,
        };
        /*
         * Every overlay table is a reader of the same request epoch. The main
         * table owns durable placement; MTP and current-batch LLEP are child
         * tables that borrow its banks through the exact same device ticket.
         * LLEP may mutate only its request-local assignment/arrival state.
         */
        const bool bind_overlay_epoch = use_expert_overlay;
        if (total_tokens == 1 &&
            (mapped_intermediate_shards ||
             (rocm_env.moe_grouped_decode && rocm_env.moe_device_routed_decode && runtime_table_eligible)))
        {
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                runtime_table_identity,
                total_tokens,
                runtime_table_layers,
                runtime_histogram_producer_role,
                bind_overlay_epoch);
        }
        else if (total_tokens > 1)
        {
            // Fixed-topology grouped prefill consumes routing tensors directly.
            // Decode-equivalent verifier routing instead uses the same
            // runtime-table router as serial decode, so build the table even
            // if an old environment still tries to disable grouped prefill;
            // otherwise the strict verifier row proof would fail closed before
            // reaching the rows under test.
            // Current-batch LLEP receives a prefill-only typed identity because
            // its transfer-backed placement must not leak into static decode.
            // The typed workload distinguishes ordinary prefill/MTP sidecars
            // from a grouped main verifier. Only the verifier is a production
            // decode-evidence source and therefore registers the host drain.
            moe_runtime_table = moeRuntimeTableForDevice(
                device,
                runtime_table_identity,
                total_tokens,
                runtime_table_layers,
                runtime_histogram_producer_role,
                bind_overlay_epoch);
        }

        /* The canonical ExpertOverlay parent is sized from the complete routed
         * weight manifest, which may include one or more NextN/MTP source
         * layers beyond the main transformer interval. Every maintenance
         * buffer, transfer-format scan, collective identity, and controller
         * bound must use that materialized table capacity. Using n_layers here
         * would let the forward graph bind a larger parent while retaining a
         * smaller maintenance transaction, making request reset fail and
         * leaving the trailing MTP placement outside the sole authority. */
        const int bound_runtime_table_layers =
            moe_runtime_table
                ? moe_runtime_table->layerCount()
                : runtime_table_layers;
        const int bound_runtime_first_layer = moe_runtime_table
            ? moe_runtime_table->firstModelLayer() : config_.pp_layer_offset;

        std::optional<DeviceMoETransferSlotDirectory::FormatProfile>
            graph_rebalance_transfer_profile;
        std::optional<DeviceMoERebalanceTransferMode> graph_rebalance_transfer_mode;
        auto collectGraphRebalanceTransferProfile =
            [&]() -> DeviceMoETransferSlotDirectory::FormatProfile
        {
            // This conditional full-source scan is nested within FFN construction.
            // Reuse the enabled cold GPU setup timer; CPU inference must not
            // acquire clocks or diagnostic string/tag storage on this path.
            std::optional<PerfStatsCollector::ScopedTimer> scan_timing;
            if (device.is_gpu() && PerfStatsCollector::isDomainEnabled("forward_graph"))
                scan_timing.emplace("forward_graph", "cold_graph_build_manifest_scan",
                    "setup", device.to_string());
            if (projection_arena_)
            {
                if (!model_ctx_) throw std::logic_error("Projection transfer profile has no model metadata");
                return DeviceMoETransferSlotDirectory::profileForLayerWeightManifest(
                    buildMoEOverlayLayerWeightManifestFromLoader(model_ctx_->concreteLoader(),
                        bound_runtime_table_layers, config_.moe.num_experts,
                        bound_runtime_first_layer), DeviceMoEProjectionSet::GateUp);
            }
            /*
             * One transfer directory is shared by the complete device-local MoE
             * domain. Its slots retain arrivals from different layers, so sizing
             * it from the current layer would either reject a later, wider
             * codebook or force one full directory allocation per layer. Scan
             * every compute-ready registry descriptor before any directory is
             * created and merge those formats into one immutable capacity
             * profile.
             */
            std::vector<std::vector<DeviceMoETransferSlotDirectory::ProjectionSpec>>
                layer_formats;
            layer_formats.reserve(
                static_cast<size_t>(bound_runtime_table_layers));

            auto weight_mgr =
                model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            if (!weight_mgr)
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph requires a prepared ExpertGemmRegistry "
                    "before sizing transfer slots on " +
                    device.to_string());
            }
            const auto &registry = weight_mgr->expertGemmRegistry();

            for (int scan_layer = bound_runtime_first_layer;
                 scan_layer < bound_runtime_first_layer + bound_runtime_table_layers;
                 ++scan_layer)
            {
                auto append_registered_formats =
                    [&](const std::vector<ITensorGemm *> &gate_engines,
                        const std::vector<ITensorGemm *> &up_engines,
                        const std::vector<ITensorGemm *> &down_engines,
                        const std::string &registry_scope) -> bool
                {
                    if (gate_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts) ||
                        up_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts) ||
                        down_engines.size() !=
                            static_cast<size_t>(config_.moe.num_experts))
                    {
                        return false;
                    }

                    bool found_complete_expert = false;
                    for (int expert = 0;
                         expert < config_.moe.num_experts;
                         ++expert)
                    {
                        ITensorGemm *gate_engine =
                            gate_engines[static_cast<size_t>(expert)];
                        ITensorGemm *up_engine =
                            up_engines[static_cast<size_t>(expert)];
                        ITensorGemm *down_engine =
                            down_engines[static_cast<size_t>(expert)];
                        if (!gate_engine || !up_engine || !down_engine)
                            continue;

                        auto exact_specs =
                            transferSlotSpecsFromPreparedExpertEngines(
                                gate_engine,
                                up_engine,
                                down_engine,
                                config_.d_model,
                                config_.moe.intermediate_size);
                        if (!exact_specs.has_value())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph found an invalid NativeQuantised "
                                "descriptor triple for model layer " +
                                std::to_string(scan_layer) + " expert " +
                                std::to_string(expert) + " in " +
                                registry_scope + " on " +
                                device.to_string());
                        }
                        layer_formats.push_back(std::move(*exact_specs));
                        found_complete_expert = true;
                    }
                    return found_complete_expert;
                };

                bool found_layer_format = false;
                if (overlay_runtime_plan)
                {
                    for (const auto &domain : overlay_runtime_plan->domains())
                    {
                        if (!domainContainsDevice(domain, device))
                            continue;

                        std::vector<ITensorGemm *> gate_engines;
                        std::vector<ITensorGemm *> up_engines;
                        std::vector<ITensorGemm *> down_engines;
                        (void)registry.populateExpertEnginesForDomain(
                            domain.name,
                            device,
                            scan_layer,
                            config_.moe.num_experts,
                            gate_engines,
                            up_engines,
                            down_engines);
                        found_layer_format =
                            append_registered_formats(
                                gate_engines,
                                up_engines,
                                down_engines,
                                "domain '" + domain.name + "'") ||
                            found_layer_format;
                    }
                }
                else
                {
                    std::vector<ITensorGemm *> gate_engines;
                    std::vector<ITensorGemm *> up_engines;
                    std::vector<ITensorGemm *> down_engines;
                    (void)registry.populateExpertEngines(
                        device,
                        scan_layer,
                        config_.moe.num_experts,
                        gate_engines,
                        up_engines,
                        down_engines);
                    found_layer_format =
                        append_registered_formats(
                            gate_engines,
                            up_engines,
                            down_engines,
                            "device registry");
                }

                if (!found_layer_format)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph could not derive NativeQuantised "
                        "transfer-slot specs from prepared engines for model layer " +
                        std::to_string(scan_layer) + " on " + device.to_string());
                }
            }
            return DeviceMoETransferSlotDirectory::profileForLayerFormats(
                layer_formats);
        };
        auto graphRebalanceDomainKey = [&]() -> std::string
        {
            std::ostringstream key;
            key << device.to_string()
                << ":participant=" << config_.tp_device_idx
                << ":first_layer=" << bound_runtime_first_layer
                << ":layers=" << bound_runtime_table_layers
                << ":experts=" << config_.moe.num_experts
                << ":topk=" << config_.moe.top_k;
            return key.str();
        };
        auto graphRebalanceBindingKey = [&]() -> std::string
        {
            std::string key = graphRebalanceDomainKey();
            if (prefill_llep_transfer_candidate)
            {
                /*
                 * Prefix rehydration and current-batch LLEP intentionally
                 * share the immutable transfer directory, but they are
                 * different graph transactions with different evidence and
                 * event lifetimes.  Giving each use an explicit logical key
                 * prevents whichever graph happens to build first from
                 * donating its policy or publication semantics to the other.
                 */
                switch (prefill_transfer_binding_role)
                {
                case GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer:
                    key += ":current_batch_llep_layer=";
                    break;
                case GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer:
                    key += ":prefix_runtime_rehydration_layer=";
                    break;
                case GraphSideRebalanceBindingRole::DecodeMaintenance:
                    throw std::logic_error(
                        "Qwen35 MoE prefill transfer key received decode-maintenance policy");
                }
                key += std::to_string(layer_idx);
            }
            return key;
        };
        auto graphRebalanceCollectiveKey = [&]() -> std::string
        {
            std::ostringstream key;
            key << "backend=" << static_cast<int>(local_tp_ctx ? local_tp_ctx->backend() : CollectiveBackendType::AUTO)
                << ":degree=" << (local_tp_ctx ? local_tp_ctx->degree() : 0)
                << ":first_layer=" << bound_runtime_first_layer
                << ":layers=" << bound_runtime_table_layers
                << ":experts=" << config_.moe.num_experts
                << ":topk=" << config_.moe.top_k;
            if (local_tp_ctx)
            {
                key << ":devices=";
                const auto &devices = local_tp_ctx->devices();
                for (size_t i = 0; i < devices.size(); ++i)
                {
                    if (i > 0)
                        key << ',';
                    key << devices[i].toString();
                }
            }
            return key.str();
        };
        auto graphRebalanceWorkspaceName = [&]() -> std::string
        {
            std::string workspace =
                std::string(prefill_llep_transfer_candidate ? "moe_prefill_llep_" : "moe_device_rebalance_") +
                graphRebalanceCollectiveKey();
            if (prefill_llep_transfer_candidate)
            {
                /*
                 * A transfer-backed prefill stage joins its auxiliary transfer
                 * stream back into the layer's compute stream before expert
                 * execution. Two persistent lanes therefore cover adjacent
                 * layers while allowing layer N+2 to reuse layer N's storage
                 * only after the graph's event edge has completed. Keeping the
                 * lane count fixed is essential: a payload workspace per model
                 * layer consumes several GiB for production expert shapes and
                 * starves the forward graph's activation/KV scratch budget.
                 */
                constexpr int kPrefillLLEPTransferWorkspaceLanes = 2;
                workspace += ":prefill_lane=";
                workspace += std::to_string(
                    layer_idx >= 0 ? (layer_idx % kPrefillLLEPTransferWorkspaceLanes) : 0);
            }
            return workspace;
        };
        auto makeGraphRebalanceConfig =
            [&](GraphSideRebalanceBindingRole purpose)
            -> DeviceMoERebalanceConfig
        {
            const bool current_batch_llep_policy =
                purpose ==
                GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer;
            const bool decode_maintenance_policy =
                purpose == GraphSideRebalanceBindingRole::DecodeMaintenance;
            if (current_batch_llep_policy &&
                config_.moe.routed_prefill_assignment_policy !=
                    RoutedExpertAssignmentPolicy::LeastLoadedResident)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP binding requires the "
                    "declarative least-loaded-resident prefill policy");
            }

            auto deviceRebalanceConfigOrEnv =
                [&](uint32_t config_value, const char *env_name, int env_value) -> uint32_t
            {
                if (env.presence.has(env_name))
                    return static_cast<uint32_t>(std::max(0, env_value));
                return config_value;
            };

            DeviceMoERebalanceConfig rebalance_config;
            rebalance_config.num_layers =
                static_cast<uint32_t>(bound_runtime_table_layers);
            rebalance_config.num_experts = static_cast<uint32_t>(config_.moe.num_experts);
            rebalance_config.top_k = static_cast<uint32_t>(config_.moe.top_k);
            rebalance_config.participant_id = static_cast<uint32_t>(config_.tp_device_idx);
            rebalance_config.participant_count = static_cast<uint32_t>(local_tp_ctx ? local_tp_ctx->degree() : 0);
            rebalance_config.root_participant = static_cast<uint32_t>(
                overlay_plan ? continuationRootParticipant(*overlay_plan) : 0);
            const int current_batch_assignment_window =
                config_.moe.routed_prefill_config.assignment_window_tokens > 0
                    ? config_.moe.routed_prefill_config.assignment_window_tokens
                    : total_tokens;
            rebalance_config.window_size_tokens = static_cast<uint32_t>(
                std::max(
                    1,
                    current_batch_llep_policy
                        ? current_batch_assignment_window
                        : (decode_maintenance_policy
                               ? config_.moe.rebalance_config.window_size
                               : total_tokens)));
            const int maintenance_slack =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                                  .device_maintenance_slack_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_maintenance_slack_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_maintenance_slack_tokens);
            const int requested_maintenance_period =
                std::max(
                    1,
                    static_cast<int>(rebalance_config.window_size_tokens) +
                        maintenance_slack);
            const int configured_minimum_period =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                            .device_min_maintenance_period_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_min_maintenance_period_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_min_maintenance_period_tokens);
            const int maintenance_period =
                configured_minimum_period > 0
                    ? std::max(
                          requested_maintenance_period,
                          configured_minimum_period)
                    : requested_maintenance_period;
            const int configured_initial_period =
                !decode_maintenance_policy
                    ? 0
                    : config_.moe.rebalance_config
                            .device_initial_maintenance_period_tokens >= 0
                    ? std::max(
                          0,
                          config_.moe.rebalance_config
                              .device_initial_maintenance_period_tokens)
                    : std::max(
                          0,
                          env.moe_rebalance
                              .device_rebalance_initial_maintenance_period_tokens);
            rebalance_config.maintenance_period_tokens =
                static_cast<uint32_t>(maintenance_period);
            rebalance_config.initial_maintenance_period_tokens =
                static_cast<uint32_t>(
                    configured_initial_period > 0
                        ? configured_initial_period
                        : maintenance_period);
            rebalance_config.max_hot_replicas_per_participant = static_cast<uint32_t>(
                decode_maintenance_policy
                    ? std::min(hot_replica_cap, config_.moe.num_experts)
                    : 0);
            rebalance_config.dynamic_imbalance_threshold_per_mille =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_imbalance_threshold_per_mille
                    : 0u;
            rebalance_config.dynamic_min_improvement_per_mille =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_min_improvement_per_mille
                    : 0u;
            rebalance_config.dynamic_max_swaps_per_layer =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config.dynamic_max_swaps_per_layer
                    : 0u;
            rebalance_config.dynamic_max_plan_entries_per_wave =
                decode_maintenance_policy
                    ? config_.moe.rebalance_config
                          .dynamic_max_plan_entries_per_wave
                    : 0u;
            rebalance_config.dynamic_min_window_activations =
                static_cast<uint32_t>(std::min<uint64_t>(
                    decode_maintenance_policy
                        ? config_.moe.rebalance_config
                              .dynamic_min_window_activations
                        : 0ULL,
                    static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
            switch (purpose)
            {
            case GraphSideRebalanceBindingRole::DecodeMaintenance:
                rebalance_config.routed_assignment_policy =
                    config_.moe.routed_decode_assignment_policy ==
                            RoutedExpertAssignmentPolicy::LeastLoadedResident
                        ? kDeviceMoERebalanceAssignmentLeastLoadedResident
                        : kDeviceMoERebalanceAssignmentStaticOwner;
                rebalance_config.min_load_spread_improvement =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_load_spread_improvement,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT",
                        env.moe_rebalance
                            .device_rebalance_min_load_spread_improvement);
                rebalance_config.min_load_spread_improvement_divisor =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_load_spread_improvement_divisor,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR",
                        env.moe_rebalance
                            .device_rebalance_min_load_spread_improvement_divisor);
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_wave_spread_improvement_per_payload_slot,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                        env.moe_rebalance
                            .device_rebalance_min_wave_spread_improvement_per_payload_slot);
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot =
                    deviceRebalanceConfigOrEnv(
                        config_.moe.rebalance_config
                            .device_min_foreign_rows_per_critical_path_payload_slot,
                        "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_CRITICAL_PATH_PAYLOAD_SLOT",
                        env.moe_rebalance
                            .device_rebalance_min_foreign_rows_per_critical_path_payload_slot);
                break;
            case GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer:
                /*
                 * Paper-style LLEP owns only this batch's row assignment. Its
                 * alpha/lambda economics are declared by routed_prefill_config;
                 * durable residency-maintenance floors must never suppress or
                 * reshape the current-batch planner. A movement-positive test
                 * can therefore disable the lambda skip without also enabling
                 * Dynamic residency maintenance or mutating unrelated knobs.
                 */
                rebalance_config.routed_assignment_policy =
                    kDeviceMoERebalanceAssignmentLeastLoadedResident;
                rebalance_config.min_load_spread_improvement = 0u;
                rebalance_config.min_load_spread_improvement_divisor = 0u;
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot = 0u;
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot = 0u;
                break;
            case GraphSideRebalanceBindingRole::PrefixRuntimeRehydrationTransfer:
                /*
                 * Prefix restore rehydrates an already selected placement. It
                 * never performs a fresh least-loaded assignment and therefore
                 * carries no decode-maintenance economy thresholds.
                 */
                rebalance_config.routed_assignment_policy =
                    kDeviceMoERebalanceAssignmentStaticOwner;
                rebalance_config.min_load_spread_improvement = 0u;
                rebalance_config.min_load_spread_improvement_divisor = 0u;
                rebalance_config
                    .min_wave_spread_improvement_per_payload_slot = 0u;
                rebalance_config
                    .min_foreign_rows_per_critical_path_payload_slot = 0u;
                break;
            }
            rebalance_config.min_router_spread_improvement_per_payload_slot =
                decode_maintenance_policy
                    ? deviceRebalanceConfigOrEnv(
                          config_.moe.rebalance_config
                              .device_min_router_spread_improvement_per_payload_slot,
                          "LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT",
                          env.moe_rebalance
                              .device_rebalance_min_router_spread_improvement_per_payload_slot)
                    : 0u;
            rebalance_config.max_post_wave_load_spread_per_mille =
                decode_maintenance_policy
                    ? deviceRebalanceConfigOrEnv(
                          config_.moe.rebalance_config
                              .device_max_post_wave_load_spread_per_mille,
                          "LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE",
                          env.moe_rebalance
                              .device_rebalance_max_post_wave_load_spread_per_mille)
                    : 0u;
            if (current_batch_llep_policy)
            {
                rebalance_config.llep_alpha_numerator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_alpha_numerator);
                rebalance_config.llep_alpha_denominator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_alpha_denominator);
                rebalance_config.llep_lambda_numerator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_lambda_numerator);
                rebalance_config.llep_lambda_denominator =
                    std::max<uint32_t>(
                        1u,
                        config_.moe.routed_prefill_config
                            .llep_lambda_denominator);
                rebalance_config.llep_enable_balanced_skip =
                    config_.moe.routed_prefill_config
                            .llep_enable_balanced_skip
                        ? 1u
                        : 0u;
            }
            rebalance_config.flags =
                static_cast<uint32_t>(DeviceMoERebalanceFlags::ResetHistogramsAfterApply);
            if (rebalance_config.max_hot_replicas_per_participant > 0)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::HotReplicaCache);
            }
            if (decode_maintenance_policy &&
                env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply);
            }
            if (env.moe_rebalance.device_rebalance_collect_load_stats ||
                PerfStatsCollector::isDomainEnabled("moe_rebalance"))
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
            }

            if (decode_maintenance_policy &&
                env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                rebalance_config = deviceMoECommittedMainLayerScope(
                    rebalance_config,
                    static_cast<uint32_t>(moe_runtime_table->storageIndexForModelLayer(config_.pp_layer_offset)),
                    static_cast<uint32_t>(config_.n_layers));
            }
            else if (first_device_rebalance_decode_layer &&
                rebalance_config.num_layers > 0)
            {
                const uint32_t current_layer =
                    static_cast<uint32_t>(moe_runtime_table->storageIndexForModelLayer(layer_idx));
                if (current_layer + 1u < rebalance_config.num_layers)
                {
                    rebalance_config.layer_window_start = current_layer + 1u;
                    rebalance_config.layer_window_count =
                        rebalance_config.num_layers - rebalance_config.layer_window_start;
                }
                else if (current_layer < rebalance_config.num_layers)
                {
                    rebalance_config.layer_window_start = current_layer;
                    rebalance_config.layer_window_count = 1u;
                }
            }
            rebalance_config.layer_wave_count =
                static_cast<uint32_t>(
                    std::max(0, env.moe_rebalance.device_rebalance_layer_wave_count));
            return rebalance_config;
        };
        auto graphRebalanceMovesFixedPayloadCapacity =
            [](DeviceMoERebalanceTransferMode mode) -> bool
        {
            return deviceMoERebalanceModeMovesFixedPayloadCapacity(mode);
        };
        auto graphRebalanceTransferModeName =
            [](DeviceMoERebalanceTransferMode mode) -> const char *
        {
            switch (mode)
            {
            case DeviceMoERebalanceTransferMode::ResidentOnly:
                return "resident_only";
            case DeviceMoERebalanceTransferMode::CompactTransferSlots:
                return "compact_transfer_slots";
            case DeviceMoERebalanceTransferMode::CollectiveSidebandPayload:
                return "collective_sideband_payload";
            case DeviceMoERebalanceTransferMode::LegacyCollectiveAllGather:
                return "legacy_collective_allgather";
            }
            return "unknown";
        };
        auto graphRebalanceUsesTransferSlots = [&]() -> bool
        {
            return graph_rebalance_transfer_mode.has_value() &&
                   deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode);
        };
        auto graphRebalanceFixedPayloadTransferEnabled = [&]() -> bool
        {
            return graph_rebalance_transfer_mode.has_value() &&
                   graphRebalanceMovesFixedPayloadCapacity(*graph_rebalance_transfer_mode);
        };
        auto graphRebalanceEnsureTransferMode = [&]() -> bool
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return false;
            }

            if (!graph_rebalance_transfer_mode.has_value())
            {
                /*
                 * Durable same-tier skew correction exchanges paired complete
                 * expert owners even when the optional replica cache is zero.
                 * It therefore requires compact non-empty payload slots; a
                 * ResidentOnly selection would publish ownership metadata for
                 * weights that never arrived.
                 */
                const bool dynamic_ownership_transfers =
                    device_side_graph_rebalance_candidate;
                const bool routed_assignment_payload_transfers =
                    config_.moe.routed_decode_assignment_policy ==
                        RoutedExpertAssignmentPolicy::LeastLoadedResident ||
                    current_batch_llep_transfer_candidate;
                graph_rebalance_transfer_mode =
                    selectGraphRebalanceTransferMode(
                        *local_tp_ctx,
                        device,
                        static_cast<uint32_t>(
                            std::min(hot_replica_cap, config_.moe.num_experts)),
                        dynamic_ownership_transfers || routed_assignment_payload_transfers);
            }
            if (!graph_rebalance_transfer_mode.has_value())
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side rebalance requires graph-captured NCCL/RCCL maintenance-wave transport for " +
                    device.to_string() +
                    "; refusing to fall back to host publish/apply in device-side mode");
            }

            return true;
        };
        auto graphRebalanceDecodeUsesMutableDescriptors = [&]() -> bool
        {
            if (!device_side_graph_rebalance_candidate)
                return false;
            if (!graphRebalanceEnsureTransferMode())
                return false;

            /*
             * ResidentOnly changes runtime top-k/local-compute masks for
             * experts that already have local resident descriptors, so the
             * immutable grouped descriptor tables remain valid. Compact and
             * fixed-payload transfer-slot modes can publish newly-arrived
             * descriptors and must therefore use mutable runtime descriptors.
             */
            return deviceMoERebalanceModeUsesTransferSlots(*graph_rebalance_transfer_mode);
        };
        auto activeRuntimeBankUsesTransientLocalPayload = [&](int table_layer_idx) -> bool
        {
            if (!moe_runtime_table || table_layer_idx < 0)
                return false;
            if (moe_runtime_table->decodeRuntimePublicationRequired(table_layer_idx))
                return false;
            return deviceMoELayerUsesTransientLocalPayload(
                moe_runtime_table->hostLayerState(table_layer_idx));
        };
        auto ensureGraphRebalanceTransferMode = [&]() -> bool
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return false;
            }

            return graphRebalanceEnsureTransferMode();
        };
        auto shouldCollectGraphRebalanceTransferSpecs = [&]() -> bool
        {
            if (!ensureGraphRebalanceTransferMode())
                return false;
            return graphRebalanceUsesTransferSlots();
        };
        auto graphRebalanceTransferDirectoryKey =
            [&](uint32_t transfer_slot_count) -> std::string
        {
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::logic_error(
                    "Qwen35 MoE cannot identify a transfer directory without "
                    "an expert weight format profile");
            }

            /*
             * This key identifies physical storage, not only a logical TP
             * domain. graphRebalanceDomainKey() already includes the exact
             * DeviceId and participant. Keeping all remaining allocation
             * discriminators in this one builder prevents prefill and decode
             * maintenance from silently inventing different cache identities.
             */
            std::ostringstream key;
            key << graphRebalanceDomainKey()
                << ":slots=" << transfer_slot_count;
            for (const auto &spec :
                 graph_rebalance_transfer_profile->allocation_specs)
            {
                key << ':' << spec.label
                    << '=' << spec.N << 'x' << spec.K
                    << ":cb" << static_cast<int>(spec.codebook_id)
                    << ":pb" << spec.payload_bytes_per_block
                    << ":asym" << (spec.is_asymmetric ? 1 : 0)
                    << ":emins" << (spec.has_emins ? 1 : 0)
                    << ":format" << static_cast<int>(spec.format.kind);
            }
            key << ":floating_capacity=" << static_cast<int>(
                graph_rebalance_transfer_profile->floating_allocation_format);
            key << ":wire="
                << graph_rebalance_transfer_profile->max_wire_payload_bytes;
            return key.str();
        };
        auto getOrCreateGraphRebalanceTransferDirectory =
            [&](const DeviceMoERebalanceConfig &rebalance_config,
                DeviceMoETransferSlotDirectory::BufferedCapacity
                    transfer_capacity,
                const char *context)
            -> std::pair<
                std::string,
                std::shared_ptr<DeviceMoETransferSlotDirectory>>
        {
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::logic_error(
                    "Qwen35 MoE transfer-directory creation requires a "
                    "weight format profile");
            }

            const uint32_t transfer_slot_count =
                transfer_capacity.total_slots;

            IBackend *backend = getBackendFor(device);
            const int gpu_ordinal = gpuOrdinalForGraphDevice(device);
            if (!backend || gpu_ordinal < 0)
            {
                throw std::runtime_error(
                    "Qwen35 MoE could not resolve the transfer-directory "
                    "backend for " +
                    device.to_string() +
                    (context ? std::string(" (") + context + ")"
                             : std::string{}));
            }

            const std::string transfer_key =
                graphRebalanceTransferDirectoryKey(transfer_slot_count);
            auto existing =
                moe_transfer_slot_directories_.find(transfer_key);
            if (existing == moe_transfer_slot_directories_.end())
            {
                if (!prepared_weight_store_)
                {
                    throw std::logic_error(
                        "Qwen35 MoE transfer-directory construction has no model-owned PreparedWeightStore");
                }
                const auto memory_authority =
                    prepared_weight_store_->physicalMemoryAuthority();
                if (!memory_authority)
                {
                    throw std::logic_error(
                        "Qwen35 MoE transfer-directory construction reached allocation before physical-memory admission");
                }
                auto directory = DeviceMoETransferSlotDirectory::create(
                    backend,
                    device,
                    gpu_ordinal,
                    rebalance_config.participant_id,
                    transfer_capacity,
                    *graph_rebalance_transfer_profile,
                    memory_authority);
                existing =
                    moe_transfer_slot_directories_
                        .emplace(transfer_key, std::move(directory))
                        .first;
            }
            if (!existing->second)
            {
                throw std::logic_error(
                    "Qwen35 MoE transfer-directory cache contains a null "
                    "physical owner for " +
                    transfer_key);
            }

            /*
             * A permissive peer-memory topology can make an ownership mistake
             * numerically invisible while turning each expert GEMV into
             * fine-grained PCIe traffic. Treat reuse across a physical owner
             * boundary as a fatal graph-construction error.
             */
            existing->second->requirePhysicalOwner(
                device,
                gpu_ordinal,
                rebalance_config.participant_id);
            return {transfer_key, existing->second};
        };
        auto ensureGraphRebalanceTransferBindingOnly =
            [&](const char *context) -> const GraphSideRebalanceBinding *
        {
            if (!graph_rebalance_transport_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return nullptr;
            }

            const std::string binding_key = graphRebalanceBindingKey();
            DeviceMoERebalanceConfig rebalance_config =
                makeGraphRebalanceConfig(prefill_transfer_binding_role);
            if (!validateDeviceMoERebalanceConfig(rebalance_config))
                return nullptr;
            if (!ensureGraphRebalanceTransferMode())
                return nullptr;
            if (!graphRebalanceUsesTransferSlots())
                return nullptr;
            if (prefill_llep_transfer_candidate &&
                graph_rebalance_transfer_mode.value() !=
                    DeviceMoERebalanceTransferMode::CompactTransferSlots)
            {
                throw std::runtime_error(
                    "Qwen35 MoE LLEP prefill requires compact transfer-slot payload movement for " +
                    device.to_string() + "; refusing fixed-payload/host fallback");
            }

            const auto existing =
                moe_graph_rebalance_bindings_.find(binding_key);
            if (existing != moe_graph_rebalance_bindings_.end())
            {
                const auto &binding = existing->second;
                if (binding.role != prefill_transfer_binding_role ||
                    binding.device_id != device ||
                    binding.collective_tp_ctx != local_tp_ctx ||
                    binding.moe_runtime_table != moe_runtime_table ||
                    binding.tp_device_idx != config_.tp_device_idx ||
                    binding.transfer_mode !=
                        graph_rebalance_transfer_mode.value())
                {
                    throw std::logic_error(
                        "Qwen35 MoE prefill transfer binding identity changed "
                        "while reusing " +
                        binding_key);
                }
                return &binding;
            }

            if (deviceMoERebalanceModePlansMissingArrivals(
                    graph_rebalance_transfer_mode.value()))
            {
                rebalance_config.flags |=
                    static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
            }
            if (!graph_rebalance_transfer_profile.has_value())
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side transfer binding requires an expert weight transfer profile for layer " +
                    std::to_string(layer_idx) + " on " + device.to_string() +
                    (context ? std::string(" (") + context + ")" : std::string{}));
            }

            const uint64_t prefill_llep_slots =
                prefill_llep_transfer_candidate
                    ? static_cast<uint64_t>(
                          std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots))
                    : 1ULL;
            const auto transfer_capacity =
                DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                    rebalance_config,
                    prefill_llep_slots,
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_wave_experts)),
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.gpu_direct_transfer_buffers)));
            rebalance_config.active_transfer_slot_capacity =
                transfer_capacity.active_slots;
            rebalance_config.transfer_slot_directory_capacity =
                transfer_capacity.total_slots;
            auto [transfer_key, transfer_directory] =
                getOrCreateGraphRebalanceTransferDirectory(
                    rebalance_config,
                    transfer_capacity,
                    context);

            const std::string rebalance_workspace = graphRebalanceWorkspaceName();
            /*
             * Transfer payloads and event edges are layer-owned; only the
             * auxiliary stream and workspace storage use the bounded rolling
             * lane.  A CUDA/HIP graph may contain many records and waits on the
             * same auxiliary stream, but reusing one event object for layer 0,
             * layer 2, layer 4, ... makes those graph dependencies alias the
             * event's latest record.  That ambiguity used to invalidate
             * long-context capture after several MoE layers.
             *
             * Key the state object by the layer-specific transfer identity so
             * every stage owns a persistent compute-ready/transfer-done event
             * pair. DeviceMoERebalanceTransferState::ensure() still asks the
             * worker context for the auxiliary stream by rebalance_workspace,
             * so all even layers share lane 0 and all odd layers share lane 1.
             * The result is two streams and two workspace lanes, with explicit
             * non-aliasing graph edges for every layer.
             */
            const std::string transfer_state_key =
                binding_key + ":workspace=" + rebalance_workspace;
            auto &state_ref =
                moe_rebalance_transfer_states_[transfer_state_key];
            if (!state_ref)
                state_ref = std::make_shared<DeviceMoERebalanceTransferState>();

            const uint32_t collective_payload_slot_capacity =
                std::min<uint32_t>(
                    transfer_directory->slotCount(),
                    static_cast<uint32_t>(
                        std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots)));
            const uint64_t collective_payload_slot_bytes =
                DeviceMoERebalanceWorkspaceContract::
                    collectivePayloadSlotBytes(
                        transfer_directory->wirePayloadBytes());

            moe_graph_rebalance_bindings_[binding_key] = GraphSideRebalanceBinding{
                .transfer_key = transfer_key,
                .workspace_name = rebalance_workspace,
                .role = prefill_transfer_binding_role,
                .device_id = device,
                .collective_tp_ctx = local_tp_ctx,
                .moe_runtime_table = moe_runtime_table,
                .tp_device_idx = config_.tp_device_idx,
                .config = rebalance_config,
                .local_transfer_slots = transfer_directory->deviceEntries(),
                .local_transfer_slot_count = transfer_directory->slotCount(),
                .transfer_mode = graph_rebalance_transfer_mode.value(),
                .collective_payload_slot_bytes = collective_payload_slot_bytes,
                .collective_payload_slot_capacity = collective_payload_slot_capacity,
                .transfer_state = state_ref,
                .producer_layer_idx = layer_idx};

            return &moe_graph_rebalance_bindings_.find(binding_key)->second;
        };
        auto attachPrefillLLEPTransferBinding =
            [&](MoEExpertComputeStage::Params &expert_params,
                const char *context)
        {
            if (!prefill_llep_transfer_candidate)
                return;
            graph_rebalance_transfer_profile =
                collectGraphRebalanceTransferProfile();
            const auto *binding =
                ensureGraphRebalanceTransferBindingOnly(context);
            if (!binding)
            {
                throw std::runtime_error(
                    "Qwen35 MoE LLEP prefill could not create transfer binding for layer " +
                    std::to_string(layer_idx) + " on " + device.to_string());
            }
            expert_params.prefill_llep_tp_ctx = binding->collective_tp_ctx;
            expert_params.prefill_llep_transfer_slots =
                binding->local_transfer_slots;
            expert_params.prefill_llep_transfer_slot_count =
                binding->local_transfer_slot_count;
            expert_params.prefill_llep_payload_slot_bytes =
                binding->collective_payload_slot_bytes;
            expert_params.prefill_llep_payload_slot_capacity =
                binding->collective_payload_slot_capacity;
            expert_params.prefill_llep_transfer_mode =
                binding->transfer_mode;
            expert_params.prefix_runtime_device_rehydration =
                prefix_runtime_device_rehydration;
            if (prefix_runtime_device_rehydration)
            {
                expert_params.prefix_runtime_rehydration_transfer_state =
                    std::make_shared<DeviceMoERebalanceTransferState>();
            }
            /*
             * The binding owns persistent physical resources.  Window size,
             * maintenance period, and other policy fields belong to this
             * exact graph invocation and may vary with its prefill bucket.
             * Rebuild those fields from the declarative graph policy, then
             * project only the proven physical capacities from the binding.
             */
            DeviceMoERebalanceConfig invocation_config =
                makeGraphRebalanceConfig(prefill_transfer_binding_role);
            invocation_config.active_transfer_slot_capacity =
                binding->config.active_transfer_slot_capacity;
            invocation_config.transfer_slot_directory_capacity =
                binding->config.transfer_slot_directory_capacity;
            if (deviceMoERebalanceModePlansMissingArrivals(
                    binding->transfer_mode))
            {
                invocation_config.flags |= static_cast<uint32_t>(
                    DeviceMoERebalanceFlags::PlanMissingArrivals);
            }
            if (!validateDeviceMoERebalanceConfig(invocation_config))
            {
                throw std::logic_error(
                    "Qwen35 MoE prefill transfer invocation produced an "
                    "invalid graph-owned rebalance policy for layer " +
                    std::to_string(layer_idx) + " on " +
                    device.to_string());
            }
            expert_params.prefill_llep_rebalance_config = invocation_config;
            expert_params.prefill_llep_transfer_state =
                binding->transfer_state;
            expert_params.prefill_llep_workspace_name =
                binding->workspace_name;
        };

        bool graph_rebalance_plan_inserted = false;
        std::string graph_rebalance_apply_node;
        std::string graph_rebalance_collect_node;
        std::string graph_rebalance_plan_after_sideband_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_plan_after_sideband_params;
        std::string graph_rebalance_pack_payload_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_pack_payload_params;
        std::string graph_rebalance_unpack_payload_node;
        std::optional<MoEDeviceRebalanceStage::Params> graph_rebalance_unpack_payload_params;
        bool graph_rebalance_transfer_command_sideband_taken = false;
        bool graph_rebalance_transfer_payload_sideband_taken = false;
        std::string current_batch_llep_plan_node;
        std::string current_batch_llep_apply_node;
        std::string current_batch_llep_expert_node;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params>
            current_batch_llep_plan_params;
        std::optional<MoEGPUCurrentBatchLLEPStage::Params>
            current_batch_llep_apply_params;
        bool current_batch_llep_payload_sideband_taken = false;
        const bool graph_rebalance_can_have_allreduce_anchor =
            config_.tp_ctx && config_.tp_ctx->degree() > 1;

        /**
         * Build the one payload sideband consumed by the shared-expert anchor.
         *
         * PlanAndPack declares and writes the send buffer. The matching apply
         * phase declares the gathered buffer. Returning this binding exactly
         * once makes duplicate collective attachment unrepresentable while
         * leaving the physical pointer resolution to TPAllreduceStage.
         */
        auto takeCurrentBatchLLEPPayloadSideband =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            if (!current_batch_llep_plan_params.has_value() ||
                !current_batch_llep_apply_params.has_value() ||
                current_batch_llep_payload_sideband_taken)
            {
                return {};
            }
            const auto &params = *current_batch_llep_plan_params;
            const size_t payload_bytes =
                MoEGPUCurrentBatchLLEPStage::localPayloadBytes(params);
            if (payload_bytes == 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP sideband has zero payload capacity for layer " +
                    std::to_string(layer_idx));
            }

            TPAllreduceSidebandWorkspaceBinding payload;
            payload.kind = LocalTPCollectiveSidebandKind::Allgather;
            payload.send_buffer_name =
                MoEGPUCurrentBatchLLEPStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                    params.workspace_name);
            payload.recv_buffer_name =
                MoEGPUCurrentBatchLLEPStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                    params.workspace_name);
            payload.element_count = payload_bytes;
            payload.dtype = CollectiveDataType::INT8;
            payload.root_device_index =
                static_cast<int>(params.config.root_participant);
            payload.name = "moe_current_batch_llep_payload";
            current_batch_llep_payload_sideband_taken = true;
            return {std::move(payload)};
        };
        auto makeGraphRebalanceStateSidebands =
            [&](const GraphSideRebalanceBinding &binding)
            -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            if (!binding.collective_tp_ctx ||
                !binding.collective_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture())
            {
                return sidebands;
            }

            const uint32_t window_count =
                binding.config.layer_window_count == 0u
                    ? binding.config.num_layers
                    : std::min(binding.config.layer_window_count,
                               binding.config.num_layers);
            const uint32_t wave_count =
                binding.config.layer_wave_count == 0u
                    ? window_count
                    : std::min(binding.config.layer_wave_count, window_count);
            const size_t local_histogram_entries =
                static_cast<size_t>(std::max<uint32_t>(1u, wave_count)) *
                static_cast<size_t>(binding.config.num_experts);
            if (local_histogram_entries == 0)
                return sidebands;

            TPAllreduceSidebandWorkspaceBinding histogram_sideband;
            histogram_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
            histogram_sideband.send_buffer_name =
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_LOCAL_HISTOGRAM,
                    binding.workspace_name);
            histogram_sideband.recv_buffer_name =
                MoEDeviceRebalanceStage::workspaceBufferName(
                    MoEDeviceRebalanceStage::WS_GATHERED_HISTOGRAM,
                    binding.workspace_name);
            static_assert(sizeof(uint64_t) == 2 * sizeof(int32_t));
            histogram_sideband.element_count = local_histogram_entries * 2u;
            histogram_sideband.dtype = CollectiveDataType::INT32;
            histogram_sideband.root_device_index =
                static_cast<int>(binding.config.root_participant);
            histogram_sideband.name = "moe_rebalance_histogram_sideband";
            sidebands.push_back(std::move(histogram_sideband));

            return sidebands;
        };
        auto takeGraphRebalanceStateSidebands =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            if (graph_rebalance_collect_node.empty())
                return {};

            const auto binding_it = moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};

            auto sidebands = makeGraphRebalanceStateSidebands(binding_it->second);
            if (!sidebands.empty())
                binding_it->second.state_sideband_enabled = true;
            return sidebands;
        };
        auto transferPlanCapacityForBinding =
            [](const GraphSideRebalanceBinding &binding) -> size_t
        {
            return static_cast<size_t>(
                deviceMoERebalanceCommandPlanCapacity(
                    binding.config,
                    binding.transfer_mode));
        };
        auto commandBufferCountForBinding =
            [](const GraphSideRebalanceBinding &binding) -> size_t
        {
            return binding.local_transfer_slot_count > 0 ? 2u : 1u;
        };
        auto makeRebalanceStageParamsFromBinding =
            [&](const GraphSideRebalanceBinding &binding,
                const std::string &stage_name,
                DeviceMoERebalanceStagePhase phase)
            -> MoEDeviceRebalanceStage::Params
        {
            MoEDeviceRebalanceStage::Params params;
            params.device_id = device;
            params.tp_ctx = binding.collective_tp_ctx;
            params.moe_runtime_table = binding.moe_runtime_table;
            params.tp_device_idx = config_.tp_device_idx;
            params.config = binding.config;
            params.local_transfer_slots = binding.local_transfer_slots;
            params.local_transfer_slot_count = binding.local_transfer_slot_count;
            params.collective_payload_slot_bytes = binding.collective_payload_slot_bytes;
            params.collective_payload_slot_capacity = binding.collective_payload_slot_capacity;
            params.stage_name = stage_name;
            params.workspace_name = binding.workspace_name;
            params.phase = phase;
            params.transfer_mode = binding.transfer_mode;
            params.transfer_state = binding.transfer_state;
            params.join_transfer_stream_after_copy = false;
            return params;
        };
        auto takeGraphRebalanceTransferSidebands =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            const auto binding_it = moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};
            const auto &binding = binding_it->second;
            if (!binding.collective_tp_ctx ||
                !binding.collective_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture() ||
                binding.transfer_mode != DeviceMoERebalanceTransferMode::CollectiveSidebandPayload ||
                binding.local_transfer_slot_count == 0 ||
                binding.collective_payload_slot_bytes == 0 ||
                binding.producer_layer_idx < 0)
            {
                return {};
            }

            const int layer_delta = layer_idx - binding.producer_layer_idx;
            const size_t plan_capacity = transferPlanCapacityForBinding(binding);
            const size_t command_buffer_count = commandBufferCountForBinding(binding);
            if (plan_capacity == 0)
                return {};

            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            bool consumed_sideband = false;
            if (layer_delta == 1 && !graph_rebalance_transfer_command_sideband_taken)
            {
                static_assert((sizeof(DeviceMoERebalancePlanEntry) % sizeof(int32_t)) == 0);
                static_assert((sizeof(DeviceMoERebalanceCommandBufferHeader) % sizeof(int32_t)) == 0);
                static_assert((sizeof(DeviceMoERebalanceWaveState) % sizeof(int32_t)) == 0);

                TPAllreduceSidebandWorkspaceBinding plan_sideband;
                plan_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                plan_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_TRANSFER_PLAN,
                        binding.workspace_name);
                plan_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PLAN,
                        binding.workspace_name);
                plan_sideband.element_count =
                    (command_buffer_count * plan_capacity *
                     sizeof(DeviceMoERebalancePlanEntry)) /
                    sizeof(int32_t);
                plan_sideband.dtype = CollectiveDataType::INT32;
                plan_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                plan_sideband.name = "moe_rebalance_transfer_plan_sideband";
                sidebands.push_back(std::move(plan_sideband));

                TPAllreduceSidebandWorkspaceBinding header_sideband;
                header_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                header_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_COMMAND_HEADER,
                        binding.workspace_name);
                header_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_COMMAND_HEADER,
                        binding.workspace_name);
                header_sideband.element_count =
                    (command_buffer_count *
                     sizeof(DeviceMoERebalanceCommandBufferHeader)) /
                    sizeof(int32_t);
                header_sideband.dtype = CollectiveDataType::INT32;
                header_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                header_sideband.name = "moe_rebalance_command_header_sideband";
                sidebands.push_back(std::move(header_sideband));

                TPAllreduceSidebandWorkspaceBinding wave_sideband;
                wave_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                wave_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_WAVE_STATE,
                        binding.workspace_name);
                wave_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_WAVE_STATE,
                        binding.workspace_name);
                wave_sideband.element_count =
                    (command_buffer_count *
                     sizeof(DeviceMoERebalanceWaveState)) /
                    sizeof(int32_t);
                wave_sideband.dtype = CollectiveDataType::INT32;
                wave_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                wave_sideband.name = "moe_rebalance_wave_state_sideband";
                sidebands.push_back(std::move(wave_sideband));

                graph_rebalance_pack_payload_node =
                    prefix + "moe_device_rebalance_pack_payload_after_command_sideband";
                graph_rebalance_pack_payload_params =
                    makeRebalanceStageParamsFromBinding(
                        binding,
                        graph_rebalance_pack_payload_node,
                        DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband);
                graph_rebalance_transfer_command_sideband_taken = true;
                consumed_sideband = true;
            }
            else if (layer_delta == 2 && !graph_rebalance_transfer_payload_sideband_taken)
            {
                const size_t payload_slot_capacity =
                    std::min<size_t>(
                        plan_capacity,
                        binding.collective_payload_slot_capacity == 0
                            ? static_cast<size_t>(binding.local_transfer_slot_count)
                            : std::min<size_t>(
                                  static_cast<size_t>(binding.collective_payload_slot_capacity),
                                  static_cast<size_t>(binding.local_transfer_slot_count)));
                const size_t payload_local_bytes =
                    payload_slot_capacity *
                    (binding.transfer_mode == DeviceMoERebalanceTransferMode::CompactTransferSlots
                         ? 1u
                         : static_cast<size_t>(binding.config.participant_count)) *
                    static_cast<size_t>(binding.collective_payload_slot_bytes);
                if (payload_local_bytes == 0)
                    return {};

                TPAllreduceSidebandWorkspaceBinding payload_sideband;
                payload_sideband.kind = LocalTPCollectiveSidebandKind::Allgather;
                payload_sideband.send_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_LOCAL_TRANSFER_PAYLOAD,
                        binding.workspace_name);
                payload_sideband.recv_buffer_name =
                    MoEDeviceRebalanceStage::workspaceBufferName(
                        MoEDeviceRebalanceStage::WS_GATHERED_TRANSFER_PAYLOAD,
                        binding.workspace_name);
                payload_sideband.element_count = payload_local_bytes;
                payload_sideband.dtype = CollectiveDataType::INT8;
                payload_sideband.root_device_index =
                    static_cast<int>(binding.config.root_participant);
                payload_sideband.name = "moe_rebalance_transfer_payload_sideband";
                sidebands.push_back(std::move(payload_sideband));

                graph_rebalance_unpack_payload_node =
                    prefix + "moe_device_rebalance_unpack_payload_after_payload_sideband";
                graph_rebalance_unpack_payload_params =
                    makeRebalanceStageParamsFromBinding(
                        binding,
                        graph_rebalance_unpack_payload_node,
                        DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband);
                graph_rebalance_transfer_payload_sideband_taken = true;
                consumed_sideband = true;
            }

            if (sidebands.empty() && consumed_sideband)
            {
                graph_rebalance_transfer_command_sideband_taken = false;
                graph_rebalance_transfer_payload_sideband_taken = false;
            }
            return sidebands;
        };
        auto appendRebalanceSidebands =
            [&](std::vector<TPAllreduceSidebandWorkspaceBinding> &sidebands,
                std::vector<TPAllreduceSidebandWorkspaceBinding> extra)
        {
            for (auto &sideband : extra)
                sidebands.push_back(std::move(sideband));
        };
        auto takeGraphRebalanceSidebandsForAllreduce =
            [&]() -> std::vector<TPAllreduceSidebandWorkspaceBinding>
        {
            std::vector<TPAllreduceSidebandWorkspaceBinding> sidebands;
            if (!graph_rebalance_collect_node.empty())
                sidebands = takeGraphRebalanceStateSidebands();
            appendRebalanceSidebands(sidebands, takeGraphRebalanceTransferSidebands());
            return sidebands;
        };
        auto maybeAddGraphRebalancePayloadStageAfterSideband =
            [&](const std::string &anchor_name,
                std::string &terminal_name)
        {
            if (graph_rebalance_pack_payload_params.has_value() &&
                !graph_rebalance_pack_payload_node.empty())
            {
                graph.addNode(
                    graph_rebalance_pack_payload_node,
                    ComputeStageFactory::createMoEDeviceRebalance(
                        *graph_rebalance_pack_payload_params),
                    device);
                graph.addDependency(graph_rebalance_pack_payload_node, anchor_name);
                terminal_name = graph_rebalance_pack_payload_node;
                graph_rebalance_pack_payload_node.clear();
                graph_rebalance_pack_payload_params.reset();
            }
            if (graph_rebalance_unpack_payload_params.has_value() &&
                !graph_rebalance_unpack_payload_node.empty())
            {
                graph.addNode(
                    graph_rebalance_unpack_payload_node,
                    ComputeStageFactory::createMoEDeviceRebalance(
                        *graph_rebalance_unpack_payload_params),
                    device);
                graph.addDependency(graph_rebalance_unpack_payload_node, anchor_name);
                terminal_name = graph_rebalance_unpack_payload_node;
                graph_rebalance_unpack_payload_node.clear();
                graph_rebalance_unpack_payload_params.reset();
            }
        };
        auto maybeInsertGraphSideRebalance =
            [&](const char *context,
                const std::string &graph_rebalance_producer_node_name) -> std::string
        {
            if (!device_side_graph_rebalance_candidate ||
                !moe_runtime_table ||
                !local_tp_ctx)
            {
                return {};
            }

            if (!graph_rebalance_apply_node.empty())
                return graph_rebalance_apply_node;

            DeviceMoERebalanceConfig rebalance_config =
                makeGraphRebalanceConfig(
                    GraphSideRebalanceBindingRole::DecodeMaintenance);
            if (!validateDeviceMoERebalanceConfig(rebalance_config))
            {
                return {};
            }

            const std::string domain_key = graphRebalanceDomainKey();
            if (first_device_rebalance_decode_layer &&
                !graph_rebalance_plan_inserted)
            {
                moe_graph_rebalance_bindings_.erase(domain_key);
                if (!ensureGraphRebalanceTransferMode())
                    return {};

                const bool transfer_slot_mode_enabled =
                    graphRebalanceUsesTransferSlots();
                const std::string rebalance_workspace =
                    "moe_device_rebalance_" + graphRebalanceCollectiveKey();
                std::string transfer_key = domain_key + ":resident_only";
                DeviceMoEExpertDirectoryEntry *local_transfer_slots = nullptr;
                uint32_t local_transfer_slot_count = 0;
                uint64_t collective_payload_slot_bytes = 0;
                uint32_t collective_payload_slot_capacity = 0;
                std::shared_ptr<DeviceMoERebalanceTransferState> transfer_state;

                if (transfer_slot_mode_enabled)
                {
                    if (deviceMoERebalanceModePlansMissingArrivals(
                            graph_rebalance_transfer_mode.value()))
                    {
                        rebalance_config.flags |=
                            static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals);
                    }
                    if (!graph_rebalance_transfer_profile.has_value())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-side rebalance requires an expert weight transfer profile for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string() +
                            (context ? std::string(" (") + context + ")" : std::string{}));
                    }

                    const auto transfer_capacity =
                        DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                            rebalance_config,
                            /*minimum_active_slots=*/0u,
                            static_cast<uint32_t>(
                                std::max(
                                    1,
                                    env.moe_rebalance.gpu_direct_transfer_wave_experts)),
                            static_cast<uint32_t>(
                                std::max(
                                    1,
                                    env.moe_rebalance.gpu_direct_transfer_buffers)));
                    rebalance_config.active_transfer_slot_capacity =
                        transfer_capacity.active_slots;
                    rebalance_config.transfer_slot_directory_capacity =
                        transfer_capacity.total_slots;
                    auto directory_result =
                        getOrCreateGraphRebalanceTransferDirectory(
                            rebalance_config,
                            transfer_capacity,
                            context);
                    transfer_key = std::move(directory_result.first);
                    auto transfer_directory =
                        std::move(directory_result.second);

                    local_transfer_slots = transfer_directory->deviceEntries();
                    local_transfer_slot_count = transfer_directory->slotCount();
                    collective_payload_slot_capacity =
                        std::min<uint32_t>(
                            local_transfer_slot_count,
                            static_cast<uint32_t>(
                                std::max(1, env.moe_rebalance.device_rebalance_compact_payload_slots)));
                    auto &state_ref =
                        moe_rebalance_transfer_states_[transfer_key + ":workspace=" + rebalance_workspace];
                    if (!state_ref)
                        state_ref = std::make_shared<DeviceMoERebalanceTransferState>();
                    transfer_state = state_ref;
                    collective_payload_slot_bytes =
                        DeviceMoERebalanceWorkspaceContract::
                            collectivePayloadSlotBytes(
                                transfer_directory->wirePayloadBytes());
                }

                moe_graph_rebalance_bindings_[domain_key] = GraphSideRebalanceBinding{
                    .transfer_key = transfer_key,
                    .workspace_name = rebalance_workspace,
                    .role = GraphSideRebalanceBindingRole::DecodeMaintenance,
                    .device_id = device,
                    .collective_tp_ctx = local_tp_ctx,
                    .moe_runtime_table = moe_runtime_table,
                    .tp_device_idx = config_.tp_device_idx,
                    .config = rebalance_config,
                    .local_transfer_slots = local_transfer_slots,
                    .local_transfer_slot_count = local_transfer_slot_count,
                    .transfer_mode = graph_rebalance_transfer_mode.value(),
                    .collective_payload_slot_bytes = collective_payload_slot_bytes,
                    .collective_payload_slot_capacity = collective_payload_slot_capacity,
                    .transfer_state = transfer_state,
                    .producer_layer_idx = layer_idx};
                graph_rebalance_plan_inserted = true;

                const bool producer_runs_in_maintenance_graph =
                    env.moe_rebalance.device_rebalance_maintenance_graph;
                const bool producer_can_use_collective_sideband =
                    !producer_runs_in_maintenance_graph &&
                    graph_rebalance_transfer_mode.value() ==
                        DeviceMoERebalanceTransferMode::CollectiveSidebandPayload &&
                    graph_rebalance_can_have_allreduce_anchor &&
                    local_tp_ctx &&
                    local_tp_ctx->supportsCollectiveSidebandOnStreamGraphCapture();

                if (producer_can_use_collective_sideband)
                {
                    graph_rebalance_collect_node =
                        prefix + "moe_device_rebalance_collect_state";
                    MoEDeviceRebalanceStage::Params collect_params;
                    collect_params.device_id = device;
                    collect_params.tp_ctx = local_tp_ctx;
                    collect_params.moe_runtime_table = moe_runtime_table;
                    collect_params.tp_device_idx = config_.tp_device_idx;
                    collect_params.config = rebalance_config;
                    collect_params.local_transfer_slots = local_transfer_slots;
                    collect_params.local_transfer_slot_count = local_transfer_slot_count;
                    collect_params.collective_payload_slot_bytes = collective_payload_slot_bytes;
                    collect_params.collective_payload_slot_capacity = collective_payload_slot_capacity;
                    collect_params.stage_name = graph_rebalance_collect_node;
                    collect_params.workspace_name = rebalance_workspace;
                    collect_params.phase = DeviceMoERebalanceStagePhase::CollectState;
                    collect_params.transfer_mode = graph_rebalance_transfer_mode.value();
                    collect_params.transfer_state = transfer_state;

                    graph.addNode(graph_rebalance_collect_node,
                                  ComputeStageFactory::createMoEDeviceRebalance(collect_params),
                                  device);
                    if (!graph_rebalance_producer_node_name.empty())
                    {
                        graph.addDependency(graph_rebalance_collect_node,
                                            graph_rebalance_producer_node_name);
                    }
                }

                if (!producer_runs_in_maintenance_graph)
                {
                    const std::string plan_node = prefix + "moe_device_rebalance_plan_copy";

                    MoEDeviceRebalanceStage::Params plan_params;
                    plan_params.device_id = device;
                    plan_params.tp_ctx = local_tp_ctx;
                    plan_params.moe_runtime_table = moe_runtime_table;
                    plan_params.tp_device_idx = config_.tp_device_idx;
                    plan_params.config = rebalance_config;
                    plan_params.local_transfer_slots = local_transfer_slots;
                    plan_params.local_transfer_slot_count = local_transfer_slot_count;
                    plan_params.collective_payload_slot_bytes = collective_payload_slot_bytes;
                    plan_params.collective_payload_slot_capacity = collective_payload_slot_capacity;
                    plan_params.stage_name = plan_node;
                    plan_params.workspace_name = rebalance_workspace;
                    plan_params.phase = DeviceMoERebalanceStagePhase::PlanAndCopy;
                    plan_params.transfer_mode = graph_rebalance_transfer_mode.value();
                    plan_params.join_transfer_stream_after_copy =
                        producer_runs_in_maintenance_graph;
                    plan_params.transfer_state = transfer_state;

                    if (producer_can_use_collective_sideband)
                    {
                        plan_params.phase =
                            DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband;
                        graph_rebalance_plan_after_sideband_node = plan_node;
                        graph_rebalance_plan_after_sideband_params = plan_params;
                    }
                    else
                    {
                        graph.addNode(plan_node,
                                      ComputeStageFactory::createMoEDeviceRebalance(plan_params),
                                      device);
                        graph.addDependency(plan_node, prefix + "ffn_norm");
                    }
                }

                LOG_DEBUG("[Qwen35MoEGraph] Added graph-side MoE device rebalance producer"
                          << " layer=" << layer_idx
                          << " device=" << device.to_string()
                          << " participant=" << config_.tp_device_idx
                          << " degree=" << local_tp_ctx->degree()
                          << " hot_replica_cap=" << hot_replica_cap
                          << " transfer_mode="
                          << graphRebalanceTransferModeName(graph_rebalance_transfer_mode.value())
                          << " transfer_slots=" << local_transfer_slot_count
                          << " collective_payload_slot_capacity=" << collective_payload_slot_capacity
                          << " window_tokens=" << rebalance_config.window_size_tokens
                          << " layer_window_start=" << rebalance_config.layer_window_start
                          << " layer_window_count=" << rebalance_config.layer_window_count
                          << " layer_wave_count=" << rebalance_config.layer_wave_count
                          << " producer_mode="
                          << (producer_runs_in_maintenance_graph ? "maintenance_graph" : "inline_decode_graph")
                          << " context=" << (context ? context : ""));

                const bool resident_boundary_apply_candidate =
                    producer_runs_in_maintenance_graph &&
                    !deviceMoERebalanceModeUsesTransferSlots(graph_rebalance_transfer_mode.value());
                const bool producer_covers_current_layer =
                    rebalance_config.layer_window_count == 1u &&
                    rebalance_config.layer_window_start ==
                        static_cast<uint32_t>(moe_runtime_table->storageIndexForModelLayer(layer_idx));
                if (!producer_covers_current_layer && !resident_boundary_apply_candidate)
                    return {};
            }

            const auto binding_it = moe_graph_rebalance_bindings_.find(domain_key);
            if (binding_it == moe_graph_rebalance_bindings_.end())
                return {};
            /*
             * Maintenance-graph mode applies ready waves from the route kernel
             * itself so steady-state decode does not pay an extra per-token
             * apply-stage launch.
             */
            if (env.moe_rebalance.device_rebalance_maintenance_graph)
            {
                return {};
            }
            const bool decode_apply_poll_enabled =
                env.moe_rebalance.device_rebalance_decode_apply_poll;
            if (!decode_apply_poll_enabled)
            {
                return {};
            }

            const bool needs_transfer_slot_apply =
                deviceMoERebalanceModeUsesTransferSlots(binding_it->second.transfer_mode) &&
                binding_it->second.local_transfer_slot_count > 0;
            const bool needs_deferred_resident_apply =
                hasDeviceMoERebalanceFlag(
                    binding_it->second.config.flags,
                    DeviceMoERebalanceFlags::DeferRuntimeApply);
            if (!needs_transfer_slot_apply && !needs_deferred_resident_apply)
            {
                return {};
            }
            const bool boundary_apply =
                needs_deferred_resident_apply || needs_transfer_slot_apply;
            if (boundary_apply && !last_device_rebalance_decode_layer)
            {
                return {};
            }

            DeviceMoEExpertDirectoryEntry *apply_transfer_slots = nullptr;
            uint32_t apply_transfer_slot_count = 0;
            std::shared_ptr<DeviceMoERebalanceTransferState> apply_transfer_state =
                binding_it->second.transfer_state;
            if (needs_transfer_slot_apply)
            {
                const auto transfer_directory_it =
                    moe_transfer_slot_directories_.find(binding_it->second.transfer_key);
                const auto transfer_state_it =
                    moe_rebalance_transfer_states_.find(
                        binding_it->second.transfer_key + ":workspace=" +
                        binding_it->second.workspace_name);
                if (transfer_directory_it == moe_transfer_slot_directories_.end() ||
                    !transfer_directory_it->second ||
                    transfer_state_it == moe_rebalance_transfer_states_.end() ||
                    !transfer_state_it->second)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph-side rebalance lost its device transfer binding for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }
                apply_transfer_slots = transfer_directory_it->second->deviceEntries();
                apply_transfer_slot_count = transfer_directory_it->second->slotCount();
                apply_transfer_state = transfer_state_it->second;
            }

            graph_rebalance_apply_node = prefix + "moe_device_rebalance_apply";
            MoEDeviceRebalanceStage::Params apply_params;
            apply_params.device_id = device;
            apply_params.tp_ctx = local_tp_ctx;
            apply_params.moe_runtime_table = moe_runtime_table;
            apply_params.tp_device_idx = config_.tp_device_idx;
            apply_params.config = rebalance_config;
            apply_params.local_transfer_slots = apply_transfer_slots;
            apply_params.local_transfer_slot_count = apply_transfer_slot_count;
            apply_params.collective_payload_slot_bytes =
                binding_it->second.collective_payload_slot_bytes;
            apply_params.collective_payload_slot_capacity =
                binding_it->second.collective_payload_slot_capacity;
            apply_params.stage_name = graph_rebalance_apply_node;
            apply_params.workspace_name = binding_it->second.workspace_name;
            apply_params.phase = DeviceMoERebalanceStagePhase::Apply;
            apply_params.transfer_mode = binding_it->second.transfer_mode;
            apply_params.apply_layer_idx = boundary_apply ? -1 : layer_idx;
            apply_params.transfer_state = apply_transfer_state;

            graph.addNode(graph_rebalance_apply_node,
                          ComputeStageFactory::createMoEDeviceRebalance(apply_params),
                          device);
            graph.addDependency(graph_rebalance_apply_node, prefix + "ffn_norm");
            graph.addDependency(prefix + "moe_routing", graph_rebalance_apply_node);

            LOG_DEBUG("[Qwen35MoEGraph] Added graph-side MoE device rebalance layer apply"
                      << " producer_layer=" << binding_it->second.producer_layer_idx
                      << " layer=" << layer_idx
                      << " apply_layer=" << apply_params.apply_layer_idx
                      << " device=" << device.to_string()
                      << " participant=" << config_.tp_device_idx
                      << " context=" << (context ? context : ""));

            return graph_rebalance_apply_node;
        };

        // =====================================================================
        // Stage 1: Pre-FFN RMSNorm (fused with attention residual add)
        // =====================================================================
        std::optional<Qwen4ExpHyperconnectionBuilder::BlockRead> residual_read;
        if (config_.flash_next)
        {
            residual_read.emplace(addBranchBlockRead(graph, prefix, buffers,
                layer_idx, total_tokens, device, ResidualBlock::FeedForward, predictor));
            ffn_terminal = residual_read->terminal();
        }
        else
        {
            FusedResidualNormStage::Params fused_params;
            fused_params.device_id = device;
            fused_params.input = buffers.attn_proj;
            fused_params.residual = buffers.current_hidden;
            fused_params.gamma = layer.ffn_norm;
            fused_params.norm_output = buffers.normalized;
            fused_params.eps = config_.rms_norm_eps;
            fused_params.seq_len = total_tokens;
            fused_params.hidden_dim = config_.d_model;
            fused_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            fused_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            fused_params.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);

            graph.addNode(prefix + "ffn_norm",
                          ComputeStageFactory::createFusedResidualNorm(fused_params),
                          device);
            ffn_terminal = prefix + "ffn_norm";
        }

        // =====================================================================
        // Stage 2: MoE Routing (softmax top-k expert selection)
        // =====================================================================
        /*
         * Routing and routed-expert execution form one graph-local producer /
         * consumer transaction. They share Q8 hidden rows and device route
         * metadata through this explicit owner, while every separately built
         * main graph, MTP sidecar, and layer gets a different owner.
         */
        auto routed_pipeline_kernel_owner =
            std::make_shared<MoERoutedPipelineKernelOwner>();
        TensorBase *routing_indices = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_INDICES));
        TensorBase *routing_weights = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS));
        int expert_intermediate = config_.moe.intermediate_size;
        if (expert_intermediate == 0 && layer.moe_gate_exps)
        {
            // gate_exps shape: [num_experts, intermediate, d_model] or rows=num_experts*intermediate
            size_t total_rows = layer.moe_gate_exps->rows();
            expert_intermediate = static_cast<int>(total_rows / config_.moe.num_experts);
        }
        if (expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE graph requires an explicit positive routed-expert "
                "intermediate size when raw expert parents are registry-owned");
        }

        {
            MoERoutingStage::Params route_params;
            route_params.device_id = device;
            route_params.input = buffers.normalized;
            route_params.seq_len = total_tokens;
            route_params.d_model = config_.d_model;
            route_params.gate_weights = layer.moe_gate;
            route_params.num_experts = config_.moe.num_experts;
            route_params.top_k = config_.moe.top_k;
            route_params.norm_topk_prob = config_.moe.norm_topk_prob;
            route_params.layer_idx = layer_idx;
            route_params.decode_histogram =
                !device.is_cpu() || device_side_graph_rebalance_candidate ||
                        config_.moe.rebalance_config.mode == MoERebalanceRuntimeMode::Off
                    ? nullptr
                    : config_.moe.decode_histogram;
            route_params.host_routing_source = routed_histogram_source;
            route_params.host_logical_row_count =
                device.is_cpu()
                    ? total_tokens
                    : 0;
            route_params.moe_runtime_table = moe_runtime_table;
            route_params.collect_device_runtime_histogram =
                collect_runtime_histogram && !mapped_intermediate_shards;
            route_params.decode_route_publication =
                use_expert_overlay && device.is_gpu() && !moe_runtime_table
                    ? MoEDecodeRoutePublicationPolicy::
                          FixedCapacityOverlayTicket
                    : MoEDecodeRoutePublicationPolicy::DeviceRuntimeTable;
            route_params.routed_row_execution_policy =
                routed_row_execution_policy;
            route_params.routed_resident_set_policy =
                routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    hot_replica_cap == 0
                    ? RoutedExpertResidentSetPolicy::UniqueOwner
                    : RoutedExpertResidentSetPolicy::ReplicaAware;
            route_params.force_grouped_verifier_prefill_for_decode =
                mapped_intermediate_shards ? mapped_shard_grouped && total_tokens == 1
                                           : forceGroupedMoEVerifierPrefill(device);
            route_params.absolute_position_ids_device =
                device.is_gpu()
                    ? absolute_position_ids_device
                    : nullptr;
            route_params.routed_pipeline_kernel_owner =
                routed_pipeline_kernel_owner;
            route_params.force_decode_equivalent_verifier_prefill =
                mapped_intermediate_shards ? mapped_shard_grouped && mapped_shard_phase != MoEOverlayServicePhaseHint::Prefill
                                           : forceDecodeEquivalentMoERouting(device);
            // Mapped route-slot lowering has no expert FFN stage. The router
            // marks each participant's verifier boundary, while the original
            // root dispatch ticket records executed-row demand exactly once.
            route_params.grouped_verifier_histogram_role = mapped_intermediate_shards
                ? selectMappedMoEVerifierHistogramRole(grouped_main_verifier_layer, collect_runtime_histogram)
                : selectMoEGroupedVerifierHistogramRole(
                    forceGpuSmallMMainVerifierPrefill(device) &&
                    use_expert_overlay &&
                    (projection_arena_.has_value() ||
                     (!masked_local_tp_overlay_decode_runtime_table &&
                      !captured_distributed_overlay_runtime_table &&
                      !full_local_tp_replicated_overlay_decode_runtime_table)),
                    collect_runtime_histogram);
            route_params.active_row_count_device =
                device.is_gpu() && batch_size == 1
                    ? routed_live_rows
                    : nullptr;
            route_params.output_indices = routing_indices;
            route_params.output_weights = routing_weights;
            route_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            route_params.output_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
            route_params.output_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);

            if (device_side_graph_rebalance_candidate &&
                env.moe_rebalance.device_rebalance_maintenance_graph &&
                device_rebalance_decode_layer &&
                (first_device_rebalance_decode_layer ||
                 last_device_rebalance_decode_layer))
            {
                if (first_device_rebalance_decode_layer &&
                    shouldCollectGraphRebalanceTransferSpecs() &&
                    !graph_rebalance_transfer_profile.has_value())
                {
                    graph_rebalance_transfer_profile =
                        collectGraphRebalanceTransferProfile();
                }

                if (first_device_rebalance_decode_layer)
                    (void)maybeInsertGraphSideRebalance(
                        "MoE routing ready-wave apply piggyback",
                        std::string{});

                const auto binding_it =
                    moe_graph_rebalance_bindings_.find(graphRebalanceDomainKey());
                if (binding_it != moe_graph_rebalance_bindings_.end())
                {
                    const auto &binding = binding_it->second;
                    const bool needs_transfer_slot_apply =
                        deviceMoERebalanceModeUsesTransferSlots(binding.transfer_mode) &&
                        binding.local_transfer_slot_count > 0;
                    const bool needs_deferred_resident_apply =
                        hasDeviceMoERebalanceFlag(
                            binding.config.flags,
                            DeviceMoERebalanceFlags::DeferRuntimeApply);
                    if (needs_transfer_slot_apply || needs_deferred_resident_apply)
                    {
                        const auto *durable_runtime =
                            dynamic_cast<const DeviceMoERuntimeTable *>(
                                moe_runtime_table);
                        const bool durable_epoch_bound =
                            durable_runtime &&
                            durable_runtime->usesOverlayEpochTicket();
                        if (last_device_rebalance_decode_layer &&
                            !durable_epoch_bound)
                        {
                            route_params.device_rebalance_route_apply = true;
                            route_params.device_rebalance_workspace_name =
                                binding.workspace_name;
                            route_params.device_rebalance_config = binding.config;
                            route_params.device_rebalance_local_transfer_slots =
                                binding.local_transfer_slots;
                            route_params.device_rebalance_local_transfer_slot_count =
                                binding.local_transfer_slot_count;
                            route_params.device_rebalance_plan_capacity =
                                static_cast<uint32_t>(
                                    transferPlanCapacityForBinding(binding));
                            route_params.device_rebalance_command_buffer_count =
                                static_cast<uint32_t>(
                                    commandBufferCountForBinding(binding));
                            route_params.device_rebalance_apply_layer_idx = -1;
                        }
                    }
                }
            }

            MoERoutingDistribution distribution;
            if (projection_arena_ && ordinary_prefill_graph && batch_size == 1 && local_tp_ctx)
            {
                distribution = MoERoutingDistribution::fromDomain(*local_tp_ctx, device, buffers);
            }
            appendMoERoutingPipeline(graph, std::move(route_params),
                prefix + "moe_routing", prefix + "ffn_norm", std::move(distribution));
        }

        // =====================================================================
        // Stage 3: MoE Expert Compute (routed expert SwiGLU FFN)
        // =====================================================================
        TensorBase *moe_output = buffers.get(buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
        TensorBase *canonical_route_contributions =
            buffers.get(
                buffers.idFor(
                    BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS));
        TensorBase *shared_output = buffers.get(buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        bool moe_combined_output_ready = false;
        std::string shared_ffn_last; // Track last shared expert stage (empty if no shared expert)
        // Only an ordinary-prefill CPU authority can publish this existing
        // routing ticket. Carry its node identity to the independent shared FFN
        // so topological scheduling cannot delay the CPU fork behind that work.
        using SharedPrefillCPUInputFork = FFNBuildContext::SharedPrefillCPUInputFork;
        std::optional<SharedPrefillCPUInputFork> shared_prefill_cpu_input_fork;
        std::optional<std::string> shared_prefill_mapped_dispatch;
        std::optional<std::string> shared_decode_cpu_packet;
        std::optional<std::string> shared_decode_ticket_cutpoint;
        /* Retain the root-only publisher identity so trace-only checkpoints can
         * localize the exact captured sibling immediately before it. */
        /*
         * A distributed sparse overlay and dense NodeTP share one rank-wide
         * collective order.  The routed branch is constructed before the
         * shared branch, but the two are otherwise graph siblings.  Retain a
         * typed construction fact so the later shared-expert collective can
         * depend on completion of the sparse protocol on every rank; relying
         * on topological insertion order deadlocks when root and peer graphs
         * contain different payload-authority nodes.
         */
        bool routed_overlay_has_distributed_sparse_protocol = false;

        auto plannedSharedExpertDevice = [&]() -> DeviceId
        {
            DeviceId shared_device = device;
            if (overlay_runtime_plan)
            {
                const auto &shared_domain = overlay_runtime_plan->sharedExpertDomain();
                const bool distributed_participant_graph =
                    config_.moe.overlay_mpi_ctx &&
                    config_.moe.overlay_mpi_ctx->world_size() > 1;
                /*
                 * Remote expert ranks execute a participant-local graph whose
                 * only authoritative outputs are the sparse routed rows.  The
                 * surrounding full graph is temporary construction scaffolding
                 * until the dedicated participant graph owns this branch; it
                 * must never reach across runtimes to execute the root's shared
                 * expert.  Keep that non-authoritative branch on the exact local
                 * device so every operation still has a valid device context.
                 */
                if (distributed_participant_graph &&
                    !shared_domain.primary_owned_by_current_rank)
                {
                    shared_device = device;
                }
                else
                {
                    shared_device =
                        overlay_runtime_plan->sharedExpertDeviceForMVP(layer_idx);
                }
                if (domainContainsDevice(shared_domain, device))
                    shared_device = device;
            }
            return shared_device;
        };
        const bool has_shared_expert_branch =
            layer.shared_expert_gate && layer.shared_expert_up &&
            layer.shared_expert_down && shared_output;
        const DeviceId planned_shared_device =
            has_shared_expert_branch ? plannedSharedExpertDevice() : device;

        const RoutedExpertTier *local_tp_apportioned_tier = nullptr;
        const bool local_tp_apportioned_fast_candidate =
            use_expert_overlay &&
            canUseLocalTPExpertIdApportionedFastPath(
                *overlay_plan,
                device,
                &local_tp_apportioned_tier);
        const RoutedExpertTier *local_tp_replicated_tier =
            resolved_local_tp_replicated_tier;
        const bool local_tp_replicated_fast_candidate =
            resolved_local_tp_replicated_fast_candidate;

        auto needsMoEParticipantAllreduce = [&]() -> bool
        {
            return !projection_arena_ && config_.tp_ctx && config_.tp_ctx->degree() > 1 &&
                   routed_row_execution_policy ==
                       RoutedExpertRowExecutionPolicy::ParticipantAssigned;
        };
        const bool shared_expert_requires_tp_allreduce =
            has_shared_expert_branch && needsTPAllreduce() &&
            (predictor ? predictor->layout().sharedNeedsReduction() : denseTPAllreduceEnabledForCurrentGraph());

        /*
         * The grouped ROCm verifier has enough independent route work to make
         * one serial top-k loop per output lane uneconomical. A single-device
         * graph therefore publishes route dots independently and declares the
         * exact ordered reducer as its next node. This is a graph policy, not a
         * runtime branch: capture records one fixed producer/reducer topology,
         * all storage is persistent, and the reducer preserves serial decode's
         * increasing-route FP32 addition order byte-for-byte.
         */
        const int graph_participant_count =
            config_.tp_ctx ? config_.tp_ctx->degree() : 1;
        const MoERouteAccumulationPolicy route_accumulation_policy =
            selectMoERouteAccumulationPolicy(
                MoERouteAccumulationSelection{
                    .backend = device.type,
                    .participant_count = graph_participant_count,
                    .workload = forceGroupedMoEVerifierPrefill(device)
                                    ? MoERouteAccumulationWorkload::
                                          GroupedVerifier
                                    : MoERouteAccumulationWorkload::Ordinary});

        /*
         * LocalTP expert ownership must never shape the FP32 route addition
         * tree. Apportioned paths therefore publish every original route into
         * an independent slot. CPU graphs allreduce raw expert rows and repeat
         * the serial weighted-FMA fold on every participant. GPU graphs use a
         * rooted reduction of preweighted slots followed by one ordered root
         * fold and compact broadcast. When the shared GPU branch is
         * input-parallel, rank-addressed shared banks join that same rooted
         * transaction. The complete transport/arithmetic choice is a typed
         * policy below rather than an implicit backend convention.
         */
        bool canonical_local_tp_route_publication = false;
        FFNBuildContext::CanonicalMoEPublicationLowering canonical_publication_lowering;

        /**
         * Rooted continuation publication deferred until the shared branch.
         *
         * Presence means the routed branch deliberately did not broadcast its
         * intermediate row. The shared branch must reduce its partial row to
         * the same fixed authority, fuse gate+combine there, and publish only
         * the final tensor. Keeping this as one complete contract prevents a
         * later graph edit from accidentally consuming root-only bytes on a
         * non-root participant.
         */
        using DeferredOverlayCombinedPublication = FFNBuildContext::DeferredOverlayCombinedPublication;
        std::optional<DeferredOverlayCombinedPublication>
            deferred_overlay_combined_publication;
        std::string captured_overlay_routed_unit_terminal;
        std::string captured_overlay_pre_return_terminal;
        std::string captured_overlay_first_return_node;

            auto makeFFNBuildContext = [&](
                std::function<MoEExpertComputeStage::Params(TensorBase *, BufferId, std::vector<bool>, DeviceId)> make_expert_fn,
                std::function<bool(MoEExpertComputeStage::Params &, DeviceId, const std::string &, const std::string &, const MoEExpertOwnerParticipant *)> prepare_expert_fn,
                std::function<void(const MoEExpertOwnerMap &, int, const MoEExpertComputeStage::Params &, const std::string &)> register_bank_fn)
            {
                return FFNBuildContext{
                    .layer = layer,
                    .buffers = buffers,
                    .identity = identity,
                    .seq_len = seq_len,
                    .batch_size = batch_size,
                    .device = device,
                    .device_state_publication_stream = device_state_publication_stream,
                    .sequence_lengths_device = sequence_lengths_device,
                    .absolute_position_ids_device = absolute_position_ids_device,
                    .predictor = predictor,
                    .layer_idx = layer_idx,
                    .graph = graph,
                    .mtp_sidecar_context = mtp_sidecar_context,
                    .mtp_depth_idx = mtp_depth_idx,
                    .prefix = prefix,
                    .ffn_terminal = ffn_terminal,
                    .total_tokens = total_tokens,
                    .routed_live_rows = routed_live_rows,
                    .routed_service_phase = routed_service_phase,
                    .routed_histogram_source = routed_histogram_source,
                    .overlay_runtime_plan = overlay_runtime_plan,
                    .overlay_plan = overlay_plan,
                    .overlay_placement = overlay_placement,
                    .mapped_intermediate_shards = mapped_intermediate_shards,
                    .mapped_shard_phase = mapped_shard_phase,
                    .mapped_shard_grouped = mapped_shard_grouped,
                    .mapped_cuda_retained_m1 = mapped_cuda_retained_m1,
                    .mapped_decode_shared_overlap = mapped_decode_shared_overlap,
                    .layer_bindings = layer_bindings,
                    .moe_runtime_table = moe_runtime_table,
                    .ordinary_prefill_graph = ordinary_prefill_graph,
                    .node_local_route_transport = node_local_route_transport,
                    .last_device_rebalance_decode_layer = last_device_rebalance_decode_layer,
                    .local_tp_ctx = local_tp_ctx,
                    .global_tp_ctx = global_tp_ctx,
                    .distributed_expert_overlay = distributed_expert_overlay,
                    .routed_continuation_topology = routed_continuation_topology,
                    .captured_overlay_continuation = captured_overlay_continuation,
                    .captured_local_tp_continuation = captured_local_tp_continuation,
                    .cpu_llep_overlay_domain = cpu_llep_overlay_domain,
                    .cpu_llep_prefill_transport_supported = cpu_llep_prefill_transport_supported,
                    .captured_distributed_overlay_runtime_table = captured_distributed_overlay_runtime_table,
                    .captured_overlay_route_ledger_uses_grouped_publication = captured_overlay_route_ledger_uses_grouped_publication,
                    .captured_overlay_route_weight_projection = captured_overlay_route_weight_projection,
                    .current_batch_llep_plan_node = current_batch_llep_plan_node,
                    .current_batch_llep_apply_node = current_batch_llep_apply_node,
                    .current_batch_llep_expert_node = current_batch_llep_expert_node,
                    .current_batch_llep_plan_params = current_batch_llep_plan_params,
                    .current_batch_llep_apply_params = current_batch_llep_apply_params,
                    .current_batch_llep_payload_sideband_taken = current_batch_llep_payload_sideband_taken,
                    .residual_read = residual_read,
                    .routed_pipeline_kernel_owner = routed_pipeline_kernel_owner,
                    .routing_indices = routing_indices,
                    .routing_weights = routing_weights,
                    .expert_intermediate = expert_intermediate,
                    .moe_output = moe_output,
                    .canonical_route_contributions = canonical_route_contributions,
                    .shared_output = shared_output,
                    .moe_combined_output_ready = moe_combined_output_ready,
                    .shared_ffn_last = shared_ffn_last,
                    .shared_prefill_cpu_input_fork = shared_prefill_cpu_input_fork,
                    .shared_prefill_mapped_dispatch = shared_prefill_mapped_dispatch,
                    .shared_decode_cpu_packet = shared_decode_cpu_packet,
                    .shared_decode_ticket_cutpoint = shared_decode_ticket_cutpoint,
                    .routed_overlay_has_distributed_sparse_protocol = routed_overlay_has_distributed_sparse_protocol,
                    .has_shared_expert_branch = has_shared_expert_branch,
                    .planned_shared_device = planned_shared_device,
                    .shared_expert_requires_tp_allreduce = shared_expert_requires_tp_allreduce,
                    .canonical_publication_lowering = canonical_publication_lowering,
                    .deferred_overlay_combined_publication = deferred_overlay_combined_publication,
                    .captured_overlay_routed_unit_terminal = captured_overlay_routed_unit_terminal,
                    .captured_overlay_pre_return_terminal = captured_overlay_pre_return_terminal,
                    .captured_overlay_first_return_node = captured_overlay_first_return_node,
                    .forceDecodeEquivalentMoEVerifier = forceDecodeEquivalentMoEVerifier,
                    .forceGroupedSharedMoEVerifierPrefill = forceGroupedSharedMoEVerifierPrefill,
                    .graphRebalanceDomainKey = graphRebalanceDomainKey,
                    .takeCurrentBatchLLEPPayloadSideband = takeCurrentBatchLLEPPayloadSideband,
                    .appendRebalanceSidebands = appendRebalanceSidebands,
                    .takeGraphRebalanceSidebandsForAllreduce = takeGraphRebalanceSidebandsForAllreduce,
                    .maybeAddGraphRebalancePayloadStageAfterSideband = maybeAddGraphRebalancePayloadStageAfterSideband,
                    .needsMoEParticipantAllreduce = needsMoEParticipantAllreduce,
                    .makeExpertParams = std::move(make_expert_fn),
                    .prepareExpertParams = std::move(prepare_expert_fn),
                    .registerInitialOverlayParticipantBank = std::move(register_bank_fn),
                };
            };
        {
            auto makeExpertParams = [&](TensorBase *output,
                                        BufferId output_buffer_id,
                                        std::vector<bool> expert_mask,
                                        DeviceId stage_device)
            {
                MoEExpertComputeStage::Params expert_params;
                expert_params.device_id = stage_device;
                expert_params.input = buffers.normalized;
                expert_params.seq_len = total_tokens;
                expert_params.d_model = config_.d_model;
                expert_params.num_experts = config_.moe.num_experts;
                expert_params.top_k = config_.moe.top_k;
                expert_params.routed_pipeline_kernel_owner =
                    routed_pipeline_kernel_owner;
                expert_params.gate_exps = layer.moe_gate_exps;
                expert_params.up_exps = layer.moe_up_exps;
                expert_params.down_exps = layer.moe_down_exps;
                expert_params.expert_intermediate = expert_intermediate;
                expert_params.layer_idx = layer_idx;
                expert_params.routing_indices = routing_indices;
                expert_params.routing_weights = routing_weights;
                expert_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
                expert_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                expert_params.output = output;
                expert_params.output_buffer_id = output_buffer_id;
                expert_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                expert_params.prepared_store = prepared_weight_store_;
                expert_params.persistent_weight_owner = weight_bindings_.physical_memory_owner;
                expert_params.expert_mask = std::move(expert_mask);
                expert_params.moe_runtime_table = moe_runtime_table;
                expert_params.service_phase = routed_service_phase;
                expert_params.weight_descriptor_source =
                    dynamic_distributed_overlay_uses_mutable_descriptors ||
                            graphRebalanceDecodeUsesMutableDescriptors() ||
                            activeRuntimeBankUsesTransientLocalPayload(layer_idx)
                        ? MoEDecodeDescriptorSource::RuntimePlacementTable
                        : MoEDecodeDescriptorSource::StaticDescriptorTable;
                expert_params.runtime_decode_has_explicit_owner_metadata =
                    single_participant_overlay_local ||
                    masked_local_tp_overlay_decode_runtime_table ||
                    captured_distributed_overlay_runtime_table ||
                    full_local_tp_replicated_overlay_decode_runtime_table ||
                    masked_local_tp_apportioned_decode_runtime_table;
                expert_params.force_grouped_verifier_prefill_for_decode =
                    forceGroupedMoEVerifierPrefill(stage_device);
                expert_params.grouped_verifier_histogram_role =
                    selectMoEGroupedVerifierHistogramRole(
                        forceGpuSmallMMainVerifierPrefill(stage_device),
                        collect_runtime_histogram);
                expert_params.force_decode_equivalent_verifier_prefill =
                    forceDecodeEquivalentMoEVerifier(stage_device);
                expert_params.absolute_position_ids_device =
                    stage_device.is_gpu()
                        ? absolute_position_ids_device
                        : nullptr;
                expert_params.active_row_count_device =
                    stage_device.is_gpu() && batch_size == 1
                        ? routed_live_rows
                        : nullptr;
                expert_params.my_socket_id =
                    std::max(0, config_.tp_device_idx);
                expert_params.participant_count =
                    config_.tp_ctx && config_.tp_ctx->degree() > 0
                        ? config_.tp_ctx->degree()
                        : std::max(1, expert_params.my_socket_id + 1);
                expert_params.routed_assignment_policy =
                    prefill_routed_expert_assignment_policy;
                expert_params.routed_row_execution_policy =
                    routed_row_execution_policy;
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }

                if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::Apportioned)
                {
                    expert_params.local_expert_start = config_.moe.local_expert_start;
                    expert_params.local_expert_count = config_.moe.local_expert_count;
                    if (config_.moe.owner_participant_count > 1 &&
                        expert_params.expert_mask.empty())
                    {
                        /*
                         * Static ownership is derived by one shared authority
                         * for ordinary TP and graph-native overlays. Ordinal
                         * ownership reproduces the former contiguous range;
                         * random ownership uses a deterministic per-layer
                         * permutation and therefore cannot be represented by
                         * start/count alone.
                         */
                        expert_params.expert_mask =
                            routed_expert_ownership::expertMaskForParticipant(
                                config_.moe.num_experts,
                                config_.moe.owner_participant_count,
                                config_.moe.owner_participant_index,
                                layer_idx,
                                config_.moe.owner_order);
                    }
                }

                const int gpu_cache_experts = debugEnv().moe_rebalance.gpu_cache_experts_per_layer;
                if (expert_params.expert_mask.empty() && gpu_cache_experts > 0)
                {
                    expert_params.expert_mask.assign(config_.moe.num_experts, false);
                    for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                        expert_params.expert_mask[expert] = !stage_device.is_gpu();
                    LOG_DEBUG("[Qwen35MoEGraph] Initial MoE GPU expert cache bootstrap mask: device="
                              << stage_device.to_string() << " layer=" << layer_idx
                              << " cpu_initial_owner=" << (!stage_device.is_gpu()));
                }

                // GPU scratch buffers
                expert_params.gate_scratch = buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
                expert_params.up_scratch = buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));

                return expert_params;
            };

            /**
             * @brief Bind admitted overlay engines or prepare a standalone expert bank.
             * @param expert_params Graph-local execution metadata and owned engine references.
             * @param stage_device Exact participant storage device.
             * @param placement_context Diagnostic logical placement spelling.
             * @param registry_domain_name Optional already-resolved routed domain.
             * @param registry_participant Exact owner already resolved by a scoped lowering.
             * @return True after every locally executable expert has a complete triplet.
             * @throws std::runtime_error For incomplete or conflicting CPU overlay ownership.
             */
            auto prepareExpertParams = [&](MoEExpertComputeStage::Params &expert_params,
                                           DeviceId stage_device,
                                           const std::string &placement_context = {},
                                           const std::string &registry_domain_name = {},
                                           const MoEExpertOwnerParticipant *registry_participant = nullptr)
            {
                auto withPlacementContext = [&](const std::string &reason)
                {
                    if (placement_context.empty())
                        return reason;
                    return placement_context + ": " + reason;
                };

                // Set expert_registry for dynamic rebalancing registry updates
                if (model_ctx_)
                {
                    auto weight_mgr = model_ctx_->concreteWeightManager();
                    if (weight_mgr)
                        expert_params.expert_registry = &weight_mgr->expertGemmRegistry();
                }

                /*
                 * A graph-native overlay owns participant-scoped prepared
                 * engines. Its generic model-weight accessor is intentionally
                 * not an expert-slice authority: on a rank that also owns a CPU
                 * endpoint, the same canonical tensor name may denote that
                 * endpoint's much smaller packed slice. Resolve the registry
                 * first and make raw fallback structurally impossible.
                 */
                const bool registry_only_cpu_overlay = stage_device.is_cpu() &&
                    overlay_plan && overlay_plan->usesExpertOverlayAuthority();
                const bool registry_only_overlay = registry_only_cpu_overlay ||
                    (stage_device.is_gpu() && !registry_domain_name.empty());

                // Standalone paths still derive their prepared engines from
                // the exact raw parent bound to this graph device.
                if (!registry_only_overlay &&
                    !MoEExpertComputeStage::extractExpertViews(expert_params))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Failed to extract expert views for layer " << layer_idx);
                    return false;
                }

                if (registry_only_cpu_overlay)
                {
                    if (!overlay_runtime_plan || !expert_params.expert_registry)
                        throw std::runtime_error(withPlacementContext(
                            "CPU ExpertOverlay requires its retained runtime and prepared registry"));
                    auto bank = adoptMoEOverlayPreparedGraphBank(*overlay_runtime_plan,
                        *expert_params.expert_registry, {
                            .device = stage_device,
                            .tp_context = config_.tp_ctx,
                            .tp_participant = config_.tp_device_idx,
                            .domain = registry_domain_name,
                            .participant = registry_participant,
                            .layer = layer_idx,
                            .experts = expert_params.num_experts,
                            .d_model = expert_params.d_model,
                            .expert_intermediate = expert_params.expert_intermediate,
                            .first_expert = expert_params.local_expert_start,
                            .local_experts = expert_params.local_expert_count,
                            .expert_mask = std::cref(expert_params.expert_mask),
                            .original_model = prepared_weight_store_ ? prepared_weight_store_->modelId() : ModelContextId{},
                            .original_residency = config_.moe.expert_overlay_participant_residency,
                            .original_owners = routedForwardRecipe(device).owners});
                    expert_params.prepared_gate_gemm = std::move(bank.gate);
                    expert_params.prepared_up_gemm = std::move(bank.up);
                    expert_params.prepared_down_gemm = std::move(bank.down);
                    expert_params.moe_owned_kernels = std::move(bank.owners);
                    expert_params.cpu_joint_layer = std::move(bank.cpu_joint_layer);
                    if (expert_params.cpu_joint_layer)
                        expert_params.expert_intermediate = bank.local_intermediate;
                    expert_params.gate_exps = nullptr;
                    expert_params.up_exps = nullptr;
                    expert_params.down_exps = nullptr;
                    expert_params.expert_gate_views.clear();
                    expert_params.expert_up_views.clear();
                    expert_params.expert_down_views.clear();
                    expert_params.expert_weight_resolution_policy = MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
                    return true;
                }

                // Standalone CPU owns its initial packing. GPU graph construction
                // consumes the unified pipeline registry below.
                if (stage_device.is_gpu())
                {
                    const bool has_active_masked_expert = hasActiveExpertMask(expert_params.expert_mask);

                    auto weight_mgr = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                    if (!weight_mgr)
                    {
                        if (expert_params.expert_mask.empty() || has_active_masked_expert)
                            failMissingGpuExpertGemmEngines(stage_device, layer_idx,
                                                            withPlacementContext(
                                                                "WeightManager unavailable (" +
                                                                describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                                expert_params.expert_mask,
                                                                                                expert_params.prepared_gate_gemm,
                                                                                                expert_params.prepared_up_gemm,
                                                                                                expert_params.prepared_down_gemm) +
                                                                ")"));

                        expert_params.prepared_gate_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_up_gemm.assign(config_.moe.num_experts, nullptr);
                        expert_params.prepared_down_gemm.assign(config_.moe.num_experts, nullptr);
                    }
                    else
                    {
                        const auto &registry = weight_mgr->expertGemmRegistry();
                        const bool domain_scoped = !registry_domain_name.empty();
                        const bool complete_layer = domain_scoped
                                                        ? registry.hasCompleteLayerInDomain(
                                                              registry_domain_name, stage_device, layer_idx, config_.moe.num_experts)
                                                        : registry.hasCompleteLayer(stage_device, layer_idx, config_.moe.num_experts);
                        const bool populated = domain_scoped
                                                   ? registry.populateExpertEnginesForDomain(
                                                         registry_domain_name, stage_device, layer_idx, config_.moe.num_experts,
                                                         expert_params.prepared_gate_gemm,
                                                         expert_params.prepared_up_gemm,
                                                         expert_params.prepared_down_gemm)
                                                   : registry.populateExpertEngines(stage_device, layer_idx, config_.moe.num_experts,
                                                                                    expert_params.prepared_gate_gemm,
                                                                                    expert_params.prepared_up_gemm,
                                                                                    expert_params.prepared_down_gemm);

                        if (expert_params.expert_mask.empty())
                        {
                            if (!complete_layer || !populated)
                                failMissingGpuExpertGemmEngines(
                                    stage_device, layer_idx,
                                    withPlacementContext("incomplete ExpertGemmRegistry entry (" +
                                                         describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                         expert_params.expert_mask,
                                                                                         expert_params.prepared_gate_gemm,
                                                                                         expert_params.prepared_up_gemm,
                                                                                         expert_params.prepared_down_gemm) +
                                                         ")"));
                        }
                        else if (has_active_masked_expert)
                        {
                            for (int expert = 0; expert < config_.moe.num_experts; ++expert)
                            {
                                if (!expert_params.expert_mask[expert])
                                    continue;
                                if (expert_params.prepared_gate_gemm[expert] == nullptr ||
                                    expert_params.prepared_up_gemm[expert] == nullptr ||
                                    expert_params.prepared_down_gemm[expert] == nullptr)
                                    failMissingGpuExpertGemmEngines(
                                        stage_device, layer_idx,
                                        withPlacementContext("missing active masked expert engine (" +
                                                             describeMissingExpertGemmEngine(config_.moe.num_experts,
                                                                                             expert_params.expert_mask,
                                                                                             expert_params.prepared_gate_gemm,
                                                                                             expert_params.prepared_up_gemm,
                                                                                             expert_params.prepared_down_gemm) +
                                                             ")"));
                            }
                        }

                        LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                            << ": populated expert GEMM engines from registry"
                                                            << (domain_scoped ? " domain=" + registry_domain_name : std::string())
                                                            << " complete_layer=" << complete_layer
                                                            << " active_masked_expert=" << has_active_masked_expert);
                    }

                    if (registry_only_overlay)
                    {
                        expert_params.gate_exps = nullptr;
                        expert_params.up_exps = nullptr;
                        expert_params.down_exps = nullptr;
                        expert_params.expert_gate_views.clear();
                        expert_params.expert_up_views.clear();
                        expert_params.expert_down_views.clear();
                        expert_params.expert_weight_resolution_policy =
                            MoEExpertWeightResolutionPolicy::
                                PreparedRegistryOnly;
                    }
                }
                else
                {
                    if (!MoEExpertComputeStage::prepareExpertGemmEngines(expert_params))
                        return false;
                }

                return true;
            };

            /**
             * Publish one graph-local participant's immutable initial RCU bank.
             *
             * Both the homogeneous fast path and a heterogeneous topology's
             * captured continuation-local branch consume prepared engines
             * directly through @ref MoEExpertComputeStage.  Neither constructs
             * a host-staged @ref MoELocalExpertStage, so graph construction must
             * install the exact registry-owned lifetimes here.  The resident
             * mask always comes from the canonical owner map; a broader
             * execution mask (for example a replicated decode policy) must
             * never become a second residency authority.
             */
            auto registerInitialOverlayParticipantBank =
                [&](const MoEExpertOwnerMap &owner_map,
                    int local_participant,
                    const MoEExpertComputeStage::Params &expert_params,
                    const std::string &path_name)
            {
                if (config_.moe.expert_overlay_residency_authority &&
                    !config_.moe.expert_overlay_participant_residency)
                {
                    throw std::runtime_error(
                        path_name +
                        " has a global residency authority but no process-local epoch-indexed prepared banks");
                }
                if (!config_.moe.expert_overlay_participant_residency)
                    return;
                if (!expert_params.expert_registry)
                {
                    throw std::runtime_error(
                        path_name +
                        " has no model-owned prepared engine registry");
                }

                const auto *participant =
                    owner_map.participantForId(local_participant);
                auto endpoint =
                    config_.moe.expert_overlay_participant_residency
                        ->endpoint(local_participant);
                if (!participant || !endpoint)
                {
                    throw std::runtime_error(
                        path_name + " owns participant p" +
                        std::to_string(local_participant) +
                        " but its process-local residency endpoint is missing");
                }

                const auto canonical_resident_mask =
                    owner_map.expertMaskForParticipant(
                        layer_idx,
                        local_participant,
                        config_.moe.num_experts);
                std::vector<MoEOverlayPreparedExpertPayload>
                    prepared_payloads;
                std::string residency_error;
                if (!resolveMoEOverlayPreparedExpertPayloads(
                        *expert_params.expert_registry,
                        *participant,
                        layer_idx,
                        config_.moe.num_experts,
                        canonical_resident_mask,
                        prepared_payloads,
                        &residency_error) ||
                    !config_.moe.expert_overlay_participant_residency
                         ->registerInitialLayer(
                             local_participant,
                             layer_idx,
                             canonical_resident_mask,
                             prepared_payloads,
                             &residency_error))
                {
                    throw std::runtime_error(
                        path_name +
                        " could not install initial prepared residency bank for layer " +
                        std::to_string(layer_idx) + " participant " +
                        std::to_string(local_participant) + ": " +
                        residency_error);
                }
            };

            /*
             * Phase 9.8 production guard: the only accepted GPU verifier route
             * is split branch-local math.  Routed experts use the grouped MoE
             * verifier pipeline, while the shared expert uses its standalone
             * grouped table-prefill verifier path plus the normal gate/combine
             * stage.  The combined routed+shared owner is intentionally kept out
             * of the graph until it proves byte-identical to row-by-row serial
             * decode across the full model, not merely close on relaxed metrics.
            */
            const bool can_combine_shared_verifier = false;

            const RoutedExpertTier *local_tp_fast_tier =
                single_participant_overlay_local
                    ? &overlay_plan->routed_tiers.front()
                    : local_tp_apportioned_fast_candidate
                          ? local_tp_apportioned_tier
                          : local_tp_replicated_tier;
            const bool use_local_tp_routed_fast_path =
                single_participant_overlay_local ||
                local_tp_apportioned_fast_candidate ||
                phase_split_local_tp_apportioned_gpu_prefill ||
                local_tp_replicated_fast_candidate;

            if (projection_arena_)
            {
                // This is a separate physical mode. The reusable lowering
                // owns its local compute/collective DAG; the model supplies
                // only frozen weights, the existing runtime and arena IDs.
                auto *runtime = dynamic_cast<DeviceMoERuntimeTable *>(moe_runtime_table);
                const auto residency = config_.moe.expert_overlay_participant_residency;
                if (!runtime || !local_tp_ctx || !residency || !model_ctx_)
                    throw std::logic_error("Projection graph has incomplete runtime/native-domain ownership");
                auto fixed = runtime->retainFixedDownProjectionBank(layer_idx);
                const auto &owners = *routedForwardRecipe(device).owners;
                const auto *participant = fixed ? owners.participantForId(fixed->participantId()) : nullptr;
                const auto manager = model_ctx_->concreteWeightManager();
                if (!participant || !manager)
                    throw std::logic_error("Projection graph cannot resolve its fixed bank or prepared weights");
                const auto mask = owners.expertMaskForParticipant(layer_idx, participant->participant_id, config_.moe.num_experts);
                std::vector<MoEOverlayPreparedExpertPayload> payloads;
                std::string error;
                if (!resolveMoEOverlayPreparedExpertPayloads(manager->expertGemmRegistry(), *participant, layer_idx,
                        config_.moe.num_experts, mask, payloads, &error, fixed->ownership()) ||
                    !residency->registerInitialLayer(participant->participant_id, layer_idx, mask, payloads, &error))
                    throw std::runtime_error("Projection graph initial prepared residency: " + error);
                publishInitialMoEProjectionRuntimeLayer(*runtime, owners, layer_idx, payloads, device_state_publication_stream);
                MoEProjectionPipelineParams params;
                params.fixed_down = std::move(fixed);
                params.initial_gate_up = std::move(payloads);
                params.kernel_owner = routed_pipeline_kernel_owner;
                params.runtime = runtime; params.tp = local_tp_ctx;
                params.counted_exchange = local_tp_ctx->deviceCountedAllGather();
                params.rows = total_tokens; params.top_k = config_.moe.top_k;
                params.live_rows = prefillCollectiveRows(device, total_tokens);
                params.service_phase = routed_service_phase;
                // The router owns serial-decode demand and the accepted-row
                // publication boundary. Projection grouping only fills that
                // same layer ledger after final gate/up ownership is known.
                params.demand = !collect_runtime_histogram ? MoEGroupedPlanDemand::None
                    : grouped_main_verifier_layer ? MoEGroupedPlanDemand::DeferredAcceptedRows
                    : ordinary_prefill_graph ? MoEGroupedPlanDemand::OrdinaryPrefill
                                             : MoEGroupedPlanDemand::None;
                params.prefix = prefix + "projection_"; params.router_node = prefix + "moe_routing";
                params.bindArena(buffers);
                ffn_terminal = appendMoEProjectionPipeline(graph, std::move(params));
            }
            else if (use_local_tp_routed_fast_path)
            {
                auto owner_map_lifetime = routedForwardRecipe(device).owners;
                const auto graph_scope = MoEExpertOverlayGraphScope::resolveForTPParticipant(
                    *overlay_runtime_plan, device, config_.tp_ctx, config_.tp_device_idx);
                const int local_participant = participantIdForTierDevice(
                    *owner_map_lifetime,
                    0,
                    device,
                    &graph_scope);
                if (local_participant < 0)
                {
                    throw std::runtime_error(
                        "Qwen35 MoE LocalTP routed-expert fast path could not find graph-local participant for " +
                        device.to_string() + " in layer " + std::to_string(layer_idx));
                }

                const bool least_loaded_phase_split_prefill =
                    phase_split_local_tp_apportioned_gpu_prefill &&
                    prefill_routed_expert_assignment_policy ==
                        RoutedExpertAssignmentPolicy::LeastLoadedResident;
                auto participant_mask =
                    local_tp_replicated_fast_candidate &&
                            (!phase_split_local_tp_apportioned_gpu_prefill ||
                             least_loaded_phase_split_prefill)
                        ? std::vector<bool>(
                              static_cast<size_t>(config_.moe.num_experts),
                              true)
                        : owner_map_lifetime->expertMaskForParticipant(
                              layer_idx,
                              local_participant,
                              config_.moe.num_experts);
                if (!hasActiveExpertMask(participant_mask))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE LocalTP routed-expert fast path produced an empty expert mask for participant " +
                        std::to_string(local_participant) + " in layer " + std::to_string(layer_idx));
                }

                auto expert_params = makeExpertParams(
                    moe_output,
                    buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                    std::move(participant_mask),
                    device);
                expert_params.my_socket_id = local_participant;
                expert_params.participant_count =
                    participantCountForGraphNativeOverlay(
                        *owner_map_lifetime,
                        continuationRootParticipant(*overlay_plan));
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::ParticipantAssigned &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }
                const std::string domain_name = local_tp_fast_tier ? local_tp_fast_tier->domain : std::string{};
                if (!prepareExpertParams(
                        expert_params,
                        device,
                        "LocalTP routed-expert fast path participant " +
                            std::to_string(local_participant),
                        domain_name,
                        owner_map_lifetime->participantForId(local_participant)))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare LocalTP routed-expert fast-path parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }

                registerInitialOverlayParticipantBank(
                    *owner_map_lifetime,
                    local_participant,
                    expert_params,
                    "Qwen35 MoE LocalTP fast path");

                /**
                 * @brief Select the one producer for the per-route participant ledger.
                 *
                 * The mapped-sparse rooted reducer consumes a participant id for
                 * every original `[token, top-k]` router slot.  A Static
                 * placement is just as much a runtime fact as a LeastLoaded
                 * placement: its ids are read from the request-pinned durable
                 * bank, whereas LeastLoaded may select a resident participant
                 * from that same bank.  Leaving Static on the fixed-mask-only
                 * grouped path computed the local expert rows correctly but
                 * left the reducer's route ledger at its zero-initialized
                 * scratch value.  The root then silently treated every slot as
                 * root-owned and omitted its LocalTP sibling's contributions.
                 *
                 * Keep this as one typed graph-construction decision.  It
                 * names an existing device-owned producer rather than adding a
                 * host mirror or a second placement authority.
                 */
                enum class LocalTPPrefillRouteLedgerProducer : std::uint8_t
                {
                    FixedTopologyMask,
                    DurableStaticOwnerBank,
                    LeastLoadedResidentBank,
                };
                const auto prefill_route_ledger_producer = [&]()
                {
                    if (!moe_runtime_table || total_tokens <= 1)
                    {
                        return LocalTPPrefillRouteLedgerProducer::
                            FixedTopologyMask;
                    }
                    if (prefill_routed_expert_assignment_policy ==
                        RoutedExpertAssignmentPolicy::LeastLoadedResident)
                    {
                        return LocalTPPrefillRouteLedgerProducer::
                            LeastLoadedResidentBank;
                    }

                    const bool static_mapped_sparse_rooted_publication =
                        device.is_gpu() &&
                        ordinary_prefill_graph &&
                        local_tp_ctx &&
                        routed_row_execution_policy ==
                            RoutedExpertRowExecutionPolicy::
                                ParticipantAssigned &&
                        prefill_routed_expert_assignment_policy ==
                            RoutedExpertAssignmentPolicy::StaticOwner &&
                        needsMoEParticipantAllreduce() &&
                        shared_expert_requires_tp_allreduce &&
                        has_shared_expert_branch &&
                        layer.shared_expert_gate_inp &&
                        planned_shared_device == device &&
                        node_local_route_transport ==
                            MoEOverlayNodeLocalRouteTransport::MappedSparse &&
                        config_.moe.node_local_route_exchange;
                    return static_mapped_sparse_rooted_publication
                               ? LocalTPPrefillRouteLedgerProducer::
                                     DurableStaticOwnerBank
                               : LocalTPPrefillRouteLedgerProducer::
                                     FixedTopologyMask;
                }();
                if (prefill_route_ledger_producer !=
                    LocalTPPrefillRouteLedgerProducer::FixedTopologyMask)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeMaskedLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            expert_params.expert_mask,
                            local_participant,
                            participant_count,
                            owner_participants,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            prefill_route_ledger_producer ==
                                LocalTPPrefillRouteLedgerProducer::
                                    LeastLoadedResidentBank,
                            prefill_route_ledger_producer ==
                                    LocalTPPrefillRouteLedgerProducer::
                                        DurableStaticOwnerBank
                                ? "LocalTP mapped-sparse Static grouped prefill runtime bank"
                                : "LocalTP expert-ID-apportioned LLEP grouped prefill runtime bank"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize LocalTP grouped prefill runtime table for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }
                    expert_params.use_runtime_row_grouping = true;
                }

                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_row_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "LocalTP expert-ID-apportioned LLEP grouped prefill");
                }

                if (current_batch_llep_transfer_candidate)
                {
                    if (!expert_params.use_runtime_row_grouping ||
                        expert_params.prefill_llep_assignment_mode !=
                            PrefillLLEPAssignmentMode::
                                GraphPhasedCurrentBatch ||
                        !expert_params.prefill_llep_tp_ctx ||
                        !expert_params.prefill_llep_transfer_slots ||
                        expert_params.prefill_llep_transfer_slot_count == 0 ||
                        expert_params.prefill_llep_payload_slot_bytes == 0 ||
                        expert_params.prefill_llep_payload_slot_capacity == 0)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE graph-phased current-batch LLEP has an incomplete transfer binding for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }

                    auto make_current_batch_phase =
                        [&](GPUCurrentBatchLLEPPhase phase,
                            const std::string &stage_name)
                    {
                        MoEGPUCurrentBatchLLEPStage::Params params;
                        params.device_id = device;
                        params.phase = phase;
                        params.tp_ctx = expert_params.prefill_llep_tp_ctx;
                        params.moe_runtime_table =
                            expert_params.moe_runtime_table;
                        params.routing_indices =
                            expert_params.routing_indices;
                        params.routing_weights =
                            expert_params.routing_weights;
                        params.routing_indices_buffer_id =
                            expert_params.routing_indices_buffer_id;
                        params.routing_weights_buffer_id =
                            expert_params.routing_weights_buffer_id;
                        params.tp_device_idx = config_.tp_device_idx;
                        params.layer_idx = layer_idx;
                        params.seq_len = total_tokens;
                        params.num_experts = config_.moe.num_experts;
                        params.top_k = config_.moe.top_k;
                        params.config =
                            expert_params.prefill_llep_rebalance_config;
                        params.local_transfer_slots =
                            expert_params.prefill_llep_transfer_slots;
                        params.local_transfer_slot_count =
                            expert_params.prefill_llep_transfer_slot_count;
                        params.payload_slot_bytes =
                            expert_params.prefill_llep_payload_slot_bytes;
                        params.payload_slot_capacity =
                            expert_params.prefill_llep_payload_slot_capacity;
                        params.transfer_mode =
                            expert_params.prefill_llep_transfer_mode;
                        params.workspace_name =
                            expert_params.prefill_llep_workspace_name;
                        params.stage_name = stage_name;
                        return params;
                    };

                    current_batch_llep_plan_node =
                        prefix + "moe_current_batch_llep_plan_pack";
                    current_batch_llep_apply_node =
                        prefix +
                        "moe_current_batch_llep_unpack_apply_assign";
                    current_batch_llep_expert_node =
                        prefix + "moe_expert_ffn_overlay_fast";
                    current_batch_llep_plan_params =
                        make_current_batch_phase(
                            GPUCurrentBatchLLEPPhase::PlanAndPack,
                            current_batch_llep_plan_node);
                    current_batch_llep_apply_params =
                        make_current_batch_phase(
                            GPUCurrentBatchLLEPPhase::
                                UnpackApplyAndAssign,
                            current_batch_llep_apply_node);
                }

                if (moe_runtime_table && single_participant_overlay_local)
                {
                    // Seed the canonical overlay bank even when the first cold
                    // shape is prefill. Every descriptor is registry-owned and
                    // the existing epoch ticket remains the replay authority.
                    if (!initializeFullLocalDecodeRuntimeTable(
                            moe_runtime_table, layer_idx, config_.moe.num_experts,
                            config_.moe.top_k, config_.d_model, expert_intermediate,
                            FullLocalDecodeRuntimePolicy::SingleParticipant,
                            local_participant, /*participant_count=*/1, {},
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            "singleton overlay native graph build"))
                        throw std::runtime_error("Qwen35 MoE failed to initialize singleton overlay runtime placement");
                    if (forceGroupedMoEVerifierPrefill(device))
                        expert_params.use_runtime_row_grouping = true;
                }
                else if (moe_runtime_table &&
                    masked_local_tp_overlay_decode_runtime_table &&
                    expert_params.replica_set.num_replicated == 0)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeMaskedLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            expert_params.expert_mask,
                            local_participant,
                            participant_count,
                            owner_participants,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            /*allow_existing_dynamic_bank=*/true,
                            "LocalTP expert-ID-apportioned masked GPU decode graph build"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize masked LocalTP decode runtime table for layer " +
                            std::to_string(layer_idx) + " on " + device.to_string());
                    }
                    if (total_tokens > 1 && forceGroupedMoEVerifierPrefill(device))
                    {
                        expert_params.use_runtime_row_grouping = true;
                    }
                }
                else if (moe_runtime_table &&
                         full_local_tp_replicated_overlay_decode_runtime_table)
                {
                    const int participant_count = expert_params.participant_count;
                    const auto owner_participants =
                        ownerParticipantsFromMap(
                            *owner_map_lifetime,
                            layer_idx,
                            config_.moe.num_experts);
                    if (!initializeFullLocalDecodeRuntimeTable(
                            moe_runtime_table,
                            layer_idx,
                            config_.moe.num_experts,
                            config_.moe.top_k,
                            config_.d_model,
                            expert_intermediate,
                            FullLocalDecodeRuntimePolicy::FullyReplicatedLocalTP,
                            local_participant,
                            participant_count,
                            owner_participants,
                            expert_params.prepared_gate_gemm,
                            expert_params.prepared_up_gemm,
                            expert_params.prepared_down_gemm,
                            device_state_publication_stream,
                            "LocalTP replicated GPU decode graph build"))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph failed to initialize replicated LocalTP decode runtime table for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    if (total_tokens > 1 &&
                        forceGroupedMoEVerifierPrefill(device))
                    {
                        expert_params.use_runtime_row_grouping = true;
                    }
                }

                /*
                 * Routed-expert ownership is an independent parallelism axis
                 * from dense tensor parallelism. A graph may deliberately
                 * replicate all dense branches while still apportioning MoE
                 * experts across CPU participants. Gate canonical route
                 * publication on the typed row-ownership contract itself;
                 * using needsTPAllreduce() here would silently restore the
                 * ownership-dependent compact-sum path whenever dense TP is
                 * disabled.
                 */
                const bool participant_routes_require_collective =
                    needsMoEParticipantAllreduce();
                if (!participant_routes_require_collective)
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            IndependentBranchCollectives;
                }
                else if (device.is_cpu())
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedPackedRouteRows;
                }
                else if (current_batch_llep_transfer_candidate)
                {
                    /* Current-batch foreign expert rows cannot execute until
                     * their payload arrives. Qwen's shared-expert allreduce is
                     * the only existing post-router collective that is early
                     * enough to carry that payload. Keep shared publication
                     * independent in this mode and retain the exact routed-slot
                     * rooted reduction after expert compute. */
                    if (!shared_expert_requires_tp_allreduce ||
                        !has_shared_expert_branch ||
                        !layer.shared_expert_gate_inp ||
                        planned_shared_device != device)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph-phased current-batch LLEP requires an input-parallel local shared-expert collective anchor for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedRouteSlots;
                }
                else if (ordinary_prefill_graph &&
                         shared_expert_requires_tp_allreduce &&
                         has_shared_expert_branch &&
                         layer.shared_expert_gate_inp &&
                         planned_shared_device == device &&
                         node_local_route_transport ==
                             MoEOverlayNodeLocalRouteTransport::MappedSparse &&
                         config_.moe.node_local_route_exchange)
                {
                    /*
                     * The no-P2P topology already selected a sparse mapped
                     * transport before graph construction. Preserve that
                     * choice in the optimized LocalTP graph instead of
                     * transmitting a zero-filled `[top_k + participants]`
                     * rank bank through RCCL. The routed reducer and compact
                     * shared reduction below retain one fixed root and one
                     * final publication, so arithmetic and collective order
                     * remain explicit and symmetric.
                     */
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalMappedRoutesRootedShared;
                }
                else if (shared_expert_requires_tp_allreduce &&
                         has_shared_expert_branch &&
                         layer.shared_expert_gate_inp &&
                         planned_shared_device == device)
                {
                    canonical_publication_lowering.policy =
                        MoEParticipantPublicationPolicy::
                            CanonicalRootedRankBanks;
                }
                else
                {
                    /*
                     * A replicated ROCm decode graph has no dense/shared
                     * partial to join into a rooted transaction. Every live
                     * canonical slot has one physical owner, so one native
                     * FP32 allreduce publishes the exact rounded slot bank on
                     * every participant and avoids a second compact broadcast.
                     * Prefill and non-ROCm graphs retain their independently
                     * selected transports and arithmetic policies.
                     */
                    canonical_publication_lowering.policy =
                        device.is_rocm() &&
                                !(predictor ? predictor->layout().sharedNeedsReduction() : denseTPAllreduceEnabledForCurrentGraph())
                            ? MoEParticipantPublicationPolicy::
                                  CanonicalAllreduceRouteSlots
                            : MoEParticipantPublicationPolicy::
                                  CanonicalRootedRouteSlots;
                }
                canonical_local_tp_route_publication =
                    canonical_publication_lowering.
                        usesCanonicalRouteSlots();
                if (routed_row_execution_policy ==
                        RoutedExpertRowExecutionPolicy::FullyReplicatedLocal &&
                    canonical_local_tp_route_publication)
                {
                    throw std::logic_error(
                        "Qwen35 MoE fully replicated local row execution cannot publish canonical route contributions");
                }
                if (canonical_local_tp_route_publication)
                {
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE LocalTP canonical route buffer is missing for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    expert_params.canonical_route_contributions =
                        canonical_route_contributions;
                    expert_params.canonical_route_arithmetic =
                        device.is_cpu()
                            ? MoECanonicalRouteArithmeticPolicy::
                                  UnweightedExpertRowThenOrderedFMA
                            : MoECanonicalRouteArithmeticPolicy::
                                  PreweightedContributionThenOrderedAdd;
                    expert_params.canonical_route_layout =
                        canonical_publication_lowering.
                                usesPackedRouteGather()
                            ? MoECanonicalRoutePublicationLayout::
                                  PackedIndexedRouteRows
                            : MoECanonicalRoutePublicationLayout::
                                  DenseOriginalRouteSlots;
                    expert_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                }

                if (current_batch_llep_plan_params.has_value())
                {
                    if (current_batch_llep_plan_node.empty())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE current-batch LLEP plan has no graph node identity");
                    }
                    graph.addNode(
                        current_batch_llep_plan_node,
                        ComputeStageFactory::createMoEGPUCurrentBatchLLEP(
                            *current_batch_llep_plan_params),
                        device);
                    graph.addDependency(
                        current_batch_llep_plan_node,
                        prefix + "moe_routing");
                }

                graph.addNode(prefix + "moe_expert_ffn_overlay_fast",
                              ComputeStageFactory::createMoEExpertCompute(expert_params),
                              device);
                const std::string rebalance_apply_dependency =
                    maybeInsertGraphSideRebalance(
                        "LocalTP expert-ID-apportioned fast path",
                        prefix + "moe_expert_ffn_overlay_fast");
                graph.addDependency(prefix + "moe_expert_ffn_overlay_fast",
                                    rebalance_apply_dependency.empty()
                                        ? prefix + "moe_routing"
                                        : rebalance_apply_dependency);
                ffn_terminal = prefix + "moe_expert_ffn_overlay_fast";

                if (needsMoEParticipantAllreduce() &&
                    canonical_publication_lowering.usesMappedSparseRoutes())
                {
                    if (!local_tp_ctx || !moe_runtime_table ||
                        !canonical_route_contributions ||
                        config_.tp_device_idx < 0 ||
                        config_.tp_device_idx >= local_tp_ctx->degree())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped sparse LocalTP publication has an incomplete graph/runtime contract for layer " +
                            std::to_string(layer_idx));
                    }

                    const auto *const local_descriptor =
                        owner_map_lifetime->participantForId(local_participant);
                    const auto *const root_descriptor =
                        owner_map_lifetime->participantForId(
                            continuationRootParticipant(*overlay_plan));
                    if (!local_descriptor || !root_descriptor ||
                        local_descriptor->tier_idx != root_descriptor->tier_idx ||
                        local_descriptor->domain_name !=
                            root_descriptor->domain_name ||
                        local_descriptor->domain_participant_index !=
                            config_.tp_device_idx ||
                        root_descriptor->domain_participant_index < 0 ||
                        root_descriptor->domain_participant_index >=
                            local_tp_ctx->degree())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped sparse LocalTP publication has an unaligned participant identity for layer " +
                            std::to_string(layer_idx));
                    }
                    const int root_tp_index =
                        root_descriptor->domain_participant_index;

                    std::vector<MoENodeLocalRouteEndpoint> route_endpoints;
                    route_endpoints.reserve(
                        static_cast<std::size_t>(local_tp_ctx->degree()));
                    for (const auto &participant :
                         owner_map_lifetime->participants())
                    {
                        if (participant.tier_idx ==
                                local_descriptor->tier_idx &&
                            participant.domain_name ==
                                local_descriptor->domain_name)
                        {
                            route_endpoints.push_back({
                                .participant_id =
                                    participant.domain_participant_index,
                                .device = participant.device,
                            });
                        }
                    }
                    if (route_endpoints.size() !=
                        static_cast<std::size_t>(local_tp_ctx->degree()))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped sparse LocalTP endpoint count differs from its continuation cell for layer " +
                            std::to_string(layer_idx));
                    }

                    const auto &route_exchange =
                        config_.moe.node_local_route_exchange;
                    route_exchange->materialize(
                        std::move(route_endpoints),
                        root_tp_index,
                        static_cast<std::uint32_t>(
                            graphStableActivationRowCapacity(config_, device)),
                        static_cast<std::uint32_t>(config_.moe.top_k),
                        static_cast<std::uint32_t>(config_.d_model));

                    const auto &runtime_layer =
                        moe_runtime_table->hostLayerState(layer_idx);
                    const std::uint64_t required_route_slots =
                        static_cast<std::uint64_t>(total_tokens) *
                        static_cast<std::uint64_t>(config_.moe.top_k);
                    if (!runtime_layer.route_participant_ids ||
                        required_route_slots == 0u ||
                        required_route_slots >
                            static_cast<std::uint64_t>(
                                runtime_layer.prefill_route_capacity))
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped sparse LocalTP publication has no complete device route-assignment ledger for layer " +
                            std::to_string(layer_idx));
                    }
                    const MoEDomainRouteAssignmentLedger route_assignment{
                        .participant_ids =
                            runtime_layer.route_participant_ids,
                        .capacity = runtime_layer.prefill_route_capacity,
                    };
                    const MoERuntimeRouteWeightBinding runtime_weights =
                        bindMoERuntimeRouteWeights(
                            moe_runtime_table->deviceLayerState(layer_idx),
                            runtime_layer,
                            captured_overlay_route_weight_projection);
                    const MoEOverlayRoutePlacementDeviceBinding
                        route_placement =
                            moe_runtime_table->overlayRoutePlacementBinding(
                                layer_idx);
                    if (!runtime_weights.validFor(
                            static_cast<std::uint32_t>(total_tokens),
                            static_cast<std::uint32_t>(config_.moe.top_k)) ||
                        !route_placement.valid())
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE mapped sparse LocalTP publication has incomplete request-pinned route evidence for layer " +
                            std::to_string(layer_idx));
                    }

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        config_.tp_device_idx == root_tp_index
                            ? MoECanonicalRouteReductionRole::RootOwner
                            : MoECanonicalRouteReductionRole::
                                  NonRootParticipant;
                    reduce_params.node_local_route_exchange = route_exchange;
                    reduce_params.domain_route_assignment = route_assignment;
                    reduce_params.external_route_source =
                        MoEExternalCanonicalRouteSource::DeferredDenseMerge;
                    reduce_params.runtime_route_weights = runtime_weights;
                    reduce_params.overlay_route_placement = route_placement;
                    reduce_params.route_participant_id =
                        config_.tp_device_idx;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string reduce_name =
                        prefix + "moe_mapped_routes_ordered_reduce";
                    graph.addNode(
                        reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        reduce_name,
                        prefix + "moe_expert_ffn_overlay_fast");
                    ffn_terminal = reduce_name;

                    deferred_overlay_combined_publication =
                        DeferredOverlayCombinedPublication{
                            .root_device_index = root_tp_index,
                            .participant_count = local_tp_ctx->degree(),
                            .routed_terminal = reduce_name,
                            .capture_rendezvous =
                                DeferredOverlayCombinedPublication::
                                    CaptureRendezvous::NativeFullGraph,
                            .capture_wave_identity = {},
                        };
                    if (!deferred_overlay_combined_publication->valid())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE mapped sparse LocalTP publication produced an incomplete rooted transaction");
                    }
                }
                else if (needsMoEParticipantAllreduce())
                {
                    TensorBase *allreduce_buffer =
                        canonical_local_tp_route_publication
                            ? canonical_route_contributions
                            : moe_output;
                    const BufferId allreduce_buffer_id =
                        canonical_local_tp_route_publication
                            ? buffers.idFor(
                                  BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS)
                            : buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const size_t allreduce_count =
                        static_cast<size_t>(total_tokens) *
                        static_cast<size_t>(config_.d_model) *
                        (canonical_local_tp_route_publication
                             ? static_cast<size_t>(config_.moe.top_k) +
                                   (canonical_publication_lowering.usesRankBanks()
                                        ? static_cast<size_t>(
                                              expert_params.participant_count)
                                        : size_t{0})
                             : size_t{1});
                    std::string ar_name;
                    if (!canonical_local_tp_route_publication)
                    {
                        ar_name = prefix + "moe_expert_overlay_fast_allreduce";
                    }
                    else if (canonical_publication_lowering.usesRankBanks())
                    {
                        ar_name =
                            prefix + "moe_canonical_publication_reduce_to_root";
                    }
                    else if (canonical_publication_lowering.
                                 usesPackedRouteGather())
                    {
                        ar_name =
                            prefix + "moe_canonical_routes_gather_to_root";
                    }
                    else if (canonical_publication_lowering.
                                 usesRootedCollective())
                    {
                        ar_name =
                            prefix + "moe_canonical_routes_reduce_to_root";
                    }
                    else if (canonical_publication_lowering.
                                 usesAllreduceRouteSlots())
                    {
                        ar_name = prefix +
                                  "moe_canonical_routes_allreduce";
                    }
                    else
                    {
                        throw std::logic_error(
                            "Qwen35 MoE canonical publication policy has no "
                            "collective lowering for layer " +
                            std::to_string(layer_idx));
                    }
                    auto rebalance_sidebands =
                        takeGraphRebalanceSidebandsForAllreduce();
                    std::unique_ptr<IComputeStage> collective_stage;
                    int canonical_route_root_participant = -1;
                    if (canonical_publication_lowering.usesPackedRouteGather())
                    {
                        if (!rebalance_sidebands.empty() ||
                            !graph_rebalance_collect_node.empty() ||
                            graph_rebalance_plan_after_sideband_params.has_value() ||
                            graph_rebalance_pack_payload_params.has_value() ||
                            graph_rebalance_unpack_payload_params.has_value())
                        {
                            throw std::logic_error(
                                "Qwen35 MoE CPU packed route publication cannot "
                                "consume GPU graph-side rebalance sidebands for layer " +
                                std::to_string(layer_idx));
                        }
                        canonical_route_root_participant = 0;
                        MoECanonicalRouteGatherStage::Params gather_params;
                        gather_params.device_id = device;
                        gather_params.tp_ctx = config_.tp_ctx;
                        gather_params.packed_route_records =
                            canonical_route_contributions;
                        gather_params.seq_len = total_tokens;
                        gather_params.top_k = config_.moe.top_k;
                        gather_params.d_model = config_.d_model;
                        gather_params.root_participant =
                            canonical_route_root_participant;
                        gather_params.stage_name = ar_name;
                        gather_params.packed_route_records_buffer_id =
                            allreduce_buffer_id;
                        collective_stage =
                            ComputeStageFactory::createMoECanonicalRouteGather(
                                gather_params);
                    }
                    else if (canonical_publication_lowering.usesRootedCollective())
                    {
                        canonical_route_root_participant =
                            continuationRootParticipant(*overlay_plan);
                        if (!local_tp_ctx ||
                            config_.tp_device_idx < 0 ||
                            config_.tp_device_idx >= local_tp_ctx->degree() ||
                            canonical_route_root_participant < 0 ||
                            canonical_route_root_participant >=
                                local_tp_ctx->degree())
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE canonical rooted route publication "
                                "requires a valid homogeneous LocalTP participant "
                                "and root for layer " +
                                std::to_string(layer_idx));
                        }

                        TPLocalRootedCollectiveStage::Params rooted_params;
                        rooted_params.device_id = device;
                        rooted_params.tp_ctx = local_tp_ctx;
                        rooted_params.tensor = allreduce_buffer;
                        rooted_params.count = allreduce_count;
                        rooted_params.dtype = CollectiveDataType::FLOAT32;
                        rooted_params.operation =
                            TPLocalRootedCollectiveOperation::ReduceSum;
                        rooted_params.root_device_index =
                            canonical_route_root_participant;
                        rooted_params.participant_device_index =
                            config_.tp_device_idx;
                        rooted_params.stage_name = ar_name;
                        rooted_params.tensor_buffer_id = allreduce_buffer_id;
                        rooted_params.sideband_workspace_bindings =
                            std::move(rebalance_sidebands);
                        collective_stage =
                            ComputeStageFactory::createTPLocalRootedCollective(
                                rooted_params);
                    }
                    else if (canonical_publication_lowering.
                                 usesAllreduceRouteSlots())
                    {
                        collective_stage =
                            createDisjointOwnerSlotAllreduceStage(
                                allreduce_buffer,
                                allreduce_count,
                                device,
                                ar_name,
                                allreduce_buffer_id,
                                std::move(rebalance_sidebands));
                    }
                    else
                    {
                        collective_stage = createTPAllreduceStage(
                            allreduce_buffer,
                            allreduce_count,
                            device,
                            layer_idx,
                            /*is_attention=*/false,
                            ar_name,
                            allreduce_buffer_id,
                            std::move(rebalance_sidebands));
                    }
                    if (!collective_stage)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE graph could not create the required "
                            "LocalTP routed-expert collective for layer " +
                            std::to_string(layer_idx));
                    }
                    graph.addNode(ar_name, std::move(collective_stage), device);
                    /*
                     * The rooted transaction always consumes routed slots.
                     * Keep that producer edge explicit even when the rank-bank
                     * policy later adds a shared publisher and when rebalance
                     * state is piggybacked on the same collective. Those nodes
                     * are additional producers, never substitutes for routed
                     * evidence. The collective therefore cannot race either
                     * branch as the graph evolves.
                     */
                    graph.addDependency(
                        ar_name,
                        prefix + "moe_expert_ffn_overlay_fast");
                    if (!graph_rebalance_collect_node.empty())
                    {
                        graph.addDependency(
                            ar_name,
                            graph_rebalance_collect_node);
                    }
                    if (graph_rebalance_plan_after_sideband_params.has_value() &&
                        !graph_rebalance_plan_after_sideband_node.empty())
                    {
                        graph.addNode(
                            graph_rebalance_plan_after_sideband_node,
                            ComputeStageFactory::createMoEDeviceRebalance(
                                *graph_rebalance_plan_after_sideband_params),
                            device);
                        graph.addDependency(
                            graph_rebalance_plan_after_sideband_node,
                            ar_name);
                        ffn_terminal =
                            graph_rebalance_plan_after_sideband_node;
                        graph_rebalance_plan_after_sideband_node.clear();
                        graph_rebalance_plan_after_sideband_params.reset();
                    }
                    else
                    {
                        ffn_terminal = ar_name;
                    }
                    maybeAddGraphRebalancePayloadStageAfterSideband(
                        ar_name,
                        ffn_terminal);

                    if (canonical_local_tp_route_publication &&
                        canonical_publication_lowering.usesRankBanks())
                    {
                        /*
                         * Shared publication is created after its FFN branch.
                         * Retain the complete rooted-transaction identity now
                         * so later lowering can add the missing producer edge,
                         * root finalizer, and final-row broadcast without
                         * rediscovering topology or collective policy.
                         */
                        canonical_publication_lowering = {
                            .policy = MoEParticipantPublicationPolicy::
                                CanonicalRootedRankBanks,
                            .root_participant =
                                canonical_route_root_participant,
                            .participant_count =
                                expert_params.participant_count,
                            .routed_producer =
                                prefix + "moe_expert_ffn_overlay_fast",
                            .rooted_reduce_node = ar_name,
                            .post_collective_terminal = ffn_terminal,
                        };
                    }
                    else if (canonical_publication_lowering.
                                 usesRootedCollective())
                    {
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                PreweightedContributionThenOrderedAdd;
                        reduce_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                DenseOriginalRouteSlots;
                        reduce_params.reduction_role =
                            config_.tp_device_idx ==
                                    canonical_route_root_participant
                                ? MoECanonicalRouteReductionRole::RootOwner
                                : MoECanonicalRouteReductionRole::
                                      NonRootParticipant;
                        reduce_params.canonical_route_contributions_buffer_id =
                            allreduce_buffer_id;
                        reduce_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                        const std::string reduce_name =
                            prefix + "moe_canonical_routes_reduce";
                        graph.addNode(
                            reduce_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                reduce_params),
                            device);
                        graph.addDependency(reduce_name, ffn_terminal);

                        TPLocalRootedCollectiveStage::Params broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = local_tp_ctx;
                        broadcast_params.tensor = moe_output;
                        broadcast_params.count =
                            static_cast<size_t>(total_tokens) *
                            static_cast<size_t>(config_.d_model);
                        broadcast_params.dtype =
                            CollectiveDataType::FLOAT32;
                        broadcast_params.operation =
                            TPLocalRootedCollectiveOperation::Broadcast;
                        broadcast_params.root_device_index =
                            canonical_route_root_participant;
                        broadcast_params.participant_device_index =
                            config_.tp_device_idx;
                        broadcast_params.stage_name =
                            prefix + "moe_canonical_routes_broadcast";
                        broadcast_params.tensor_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string broadcast_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            broadcast_name,
                            ComputeStageFactory::
                                createTPLocalRootedCollective(
                                    broadcast_params),
                            device);
                        graph.addDependency(broadcast_name, reduce_name);
                        ffn_terminal = broadcast_name;
                    }
                    else if (canonical_publication_lowering.
                                 usesAllreduceRouteSlots())
                    {
                        /*
                         * The native collective has already copied each
                         * independently rounded original-order route slot to
                         * every participant. All graphs now perform the same
                         * deterministic increasing-slot fold locally; no root
                         * authority or return broadcast remains.
                         */
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                PreweightedContributionThenOrderedAdd;
                        reduce_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                DenseOriginalRouteSlots;
                        reduce_params.reduction_role =
                            MoECanonicalRouteReductionRole::
                                AllreduceParticipant;
                        reduce_params.canonical_route_contributions_buffer_id =
                            allreduce_buffer_id;
                        reduce_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                        const std::string reduce_name =
                            prefix + "moe_canonical_routes_reduce";
                        graph.addNode(
                            reduce_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                reduce_params),
                            device);
                        graph.addDependency(reduce_name, ffn_terminal);
                        ffn_terminal = reduce_name;
                    }
                    else if (canonical_publication_lowering.
                                 usesPackedRouteGather())
                    {
                        /*
                         * The gather transported only locally computed indexed
                         * rows. The fixed root reconstructs original slot order
                         * and owns the one serial-equivalent FMA fold; peers
                         * remain declarative no-ops until compact broadcast.
                         */
                        MoECanonicalRouteReduceStage::Params reduce_params;
                        reduce_params.device_id = device;
                        reduce_params.canonical_route_contributions =
                            canonical_route_contributions;
                        reduce_params.routing_weights = routing_weights;
                        reduce_params.output = moe_output;
                        reduce_params.seq_len = total_tokens;
                        reduce_params.top_k = config_.moe.top_k;
                        reduce_params.d_model = config_.d_model;
                        reduce_params.canonical_route_arithmetic =
                            MoECanonicalRouteArithmeticPolicy::
                                UnweightedExpertRowThenOrderedFMA;
                        reduce_params.canonical_route_layout =
                            MoECanonicalRoutePublicationLayout::
                                PackedIndexedRouteRows;
                        reduce_params.reduction_role =
                            config_.tp_device_idx ==
                                    canonical_route_root_participant
                                ? MoECanonicalRouteReductionRole::RootOwner
                                : MoECanonicalRouteReductionRole::
                                      NonRootParticipant;
                        reduce_params.canonical_route_contributions_buffer_id =
                            allreduce_buffer_id;
                        reduce_params.routing_weights_buffer_id =
                            buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                        reduce_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                        const std::string reduce_name =
                            prefix + "moe_canonical_routes_ordered_fma";
                        graph.addNode(
                            reduce_name,
                            ComputeStageFactory::createMoECanonicalRouteReduce(
                                reduce_params),
                            device);
                        graph.addDependency(reduce_name, ffn_terminal);

                        MoECanonicalOutputBroadcastStage::Params
                            broadcast_params;
                        broadcast_params.device_id = device;
                        broadcast_params.tp_ctx = config_.tp_ctx;
                        broadcast_params.output = moe_output;
                        broadcast_params.seq_len = total_tokens;
                        broadcast_params.d_model = config_.d_model;
                        broadcast_params.root_participant =
                            canonical_route_root_participant;
                        broadcast_params.stage_name =
                            prefix + "moe_canonical_routes_broadcast";
                        broadcast_params.output_buffer_id =
                            buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                        const std::string broadcast_name =
                            broadcast_params.stage_name;
                        graph.addNode(
                            broadcast_name,
                            ComputeStageFactory::
                                createMoECanonicalOutputBroadcast(
                                    broadcast_params),
                            device);
                        graph.addDependency(broadcast_name, reduce_name);
                        ffn_terminal = broadcast_name;
                    }
                }

                LOG_TRACE("[Qwen35MoEGraph] Layer " << layer_idx
                                                    << " using LocalTP routed-expert fast path on "
                                                    << device.to_string()
                                                    << " participant=" << local_participant
                                                    << " domain=" << domain_name
                                                    << " compute="
                                                    << routedExpertComputePolicyToString(
                                                           config_.moe.routed_compute_policy)
                                                    << " row_execution="
                                                    << routedExpertRowExecutionPolicyToString(
                                                           routed_row_execution_policy)
                                                    << " publication="
                                                    << moeParticipantPublicationPolicyToString(
                                                           canonical_publication_lowering.policy)
                                                    << " phase="
                                                    << (local_tp_fast_tier
                                                            ? routedExpertPhasePolicyToString(
                                                                  expertDomainForTier(
                                                                      *overlay_plan,
                                                                      *local_tp_fast_tier)
                                                                      ->routed_phase_policy)
                                                            : "uniform"));
            }
            else if (use_expert_overlay)
            {
                auto ffn_ctx = makeFFNBuildContext(
                    makeExpertParams,
                    prepareExpertParams,
                    registerInitialOverlayParticipantBank);
                buildFFNExpertOverlayBranch(ffn_ctx);
            }
            else
            {
                auto expert_params = makeExpertParams(moe_output,
                                                      buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                                                      {},
                                                      device);
                if (local_tp_ctx &&
                    total_tokens > 1 &&
                    prefill_routed_expert_assignment_policy == RoutedExpertAssignmentPolicy::LeastLoadedResident)
                {
                    expert_params.prefill_llep_tp_ctx = local_tp_ctx;
                    expert_params.prefill_llep_assignment_mode =
                        current_batch_llep_transfer_candidate
                            ? PrefillLLEPAssignmentMode::
                                  GraphPhasedCurrentBatch
                            : PrefillLLEPAssignmentMode::
                                  LogicalPositionResidentOnly;
                }
                if (!prepareExpertParams(expert_params, device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE graph failed to prepare expert parameters for layer " +
                        std::to_string(layer_idx) + " on " + device.to_string());
                }
                const bool routed_expert_output_is_partial =
                    expert_params.local_expert_count >= 0 ||
                    !expert_params.expert_mask.empty();
                const bool standard_cpu_canonical_publication =
                    device.is_cpu() &&
                    routed_expert_output_is_partial &&
                    needsMoEParticipantAllreduce();
                if (prefill_llep_transfer_candidate &&
                    (expert_params.use_runtime_row_grouping ||
                     prefix_runtime_device_rehydration))
                {
                    attachPrefillLLEPTransferBinding(
                        expert_params,
                        "standard routed expert LLEP grouped prefill");
                }

                const bool serial_decode_runtime_table_requested =
                    total_tokens == 1 &&
                    debugEnv().rocm.moe_grouped_decode &&
                    debugEnv().rocm.moe_device_routed_decode;
                const bool standard_gpu_runtime_table_requested =
                    device.is_gpu() &&
                    moe_runtime_table &&
                    (serial_decode_runtime_table_requested ||
                     grouped_main_verifier_layer);
                if (standard_gpu_runtime_table_requested)
                {
                    if (masked_local_tp_apportioned_decode_runtime_table)
                    {
                        const int participant_count =
                            local_tp_ctx ? local_tp_ctx->degree() : expert_params.participant_count;
                        const auto owner_participants =
                            routed_expert_ownership::ownerParticipantByExpert(
                                config_.moe.num_experts,
                                participant_count,
                                layer_idx,
                                config_.moe.owner_order);
                        if (!initializeMaskedLocalDecodeRuntimeTable(
                                moe_runtime_table,
                                layer_idx,
                                config_.moe.num_experts,
                                config_.moe.top_k,
                                config_.d_model,
                                expert_intermediate,
                                expert_params.expert_mask,
                                config_.tp_device_idx,
                                participant_count,
                                owner_participants,
                                {},
                                expert_params.prepared_gate_gemm,
                                expert_params.prepared_up_gemm,
                                expert_params.prepared_down_gemm,
                                device_state_publication_stream,
                                /*allow_existing_dynamic_bank=*/true,
                                "LocalTP expert-ID-apportioned masked GPU decode graph build"))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to initialize masked LocalTP decode "
                                "runtime table for layer " +
                                std::to_string(layer_idx) + " on " + device.to_string());
                        }
                    }
                    else
                    {
                        const bool full_local_decode =
                            expert_params.local_expert_start == 0 &&
                            (expert_params.local_expert_count < 0 ||
                             expert_params.local_expert_count == config_.moe.num_experts) &&
                            expert_params.replica_set.num_replicated == 0 &&
                            allExpertsEnabled(expert_params.expert_mask, config_.moe.num_experts);
                        if (!full_local_decode)
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE GPU decode runtime table requires full local expert "
                                "ownership for layer " +
                                std::to_string(layer_idx) + " on " + device.to_string());
                        }

                        if (!initializeFullLocalDecodeRuntimeTable(
                                moe_runtime_table,
                                layer_idx,
                                config_.moe.num_experts,
                                config_.moe.top_k,
                                config_.d_model,
                                expert_intermediate,
                                FullLocalDecodeRuntimePolicy::SingleParticipant,
                                /*local_participant=*/0,
                                /*participant_count=*/1,
                                /*owner_participants=*/{},
                                expert_params.prepared_gate_gemm,
                                expert_params.prepared_up_gemm,
                                expert_params.prepared_down_gemm,
                                device_state_publication_stream,
                                "single-device GPU decode graph build"))
                        {
                            throw std::runtime_error(
                                "Qwen35 MoE graph failed to initialize device decode "
                                "runtime table for layer " +
                                std::to_string(layer_idx) + " on " + device.to_string());
                        }
                    }

                    /*
                     * A grouped main verifier must retain its final device route
                     * assignment until accepted-state publication commits the
                     * accepted prefix.  The same runtime table therefore owns
                     * both grouping scratch and the deferred route ledger.  This
                     * is not optional telemetry: omitting the binding lets the
                     * verifier compute rows but makes their state transaction
                     * impossible to commit.
                     */
                    if (grouped_main_verifier_layer)
                        expert_params.use_runtime_row_grouping = true;
                }

                if (standard_cpu_canonical_publication ||
                    route_accumulation_policy ==
                        MoERouteAccumulationPolicy::
                            IndependentRouteSlotsThenOrderedFold)
                {
                    if (!canonical_route_contributions)
                    {
                        throw std::runtime_error(
                            "Qwen35 MoE canonical route-slot publication "
                            "requires canonical device storage for layer " +
                            std::to_string(layer_idx) + " on " +
                            device.to_string());
                    }
                    expert_params.canonical_route_contributions =
                        canonical_route_contributions;
                    expert_params.canonical_route_arithmetic =
                        standard_cpu_canonical_publication
                            ? MoECanonicalRouteArithmeticPolicy::
                                  UnweightedExpertRowThenOrderedFMA
                            : MoECanonicalRouteArithmeticPolicy::
                                  PreweightedContributionThenOrderedAdd;
                    expert_params.canonical_route_layout =
                        standard_cpu_canonical_publication
                            ? MoECanonicalRoutePublicationLayout::
                                  PackedIndexedRouteRows
                            : MoECanonicalRoutePublicationLayout::
                                  DenseOriginalRouteSlots;
                    expert_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                }

                auto expert_stage_owner =
                    ComputeStageFactory::createMoEExpertCompute(expert_params);
                graph.addNode(prefix + "moe_expert_ffn",
                              std::move(expert_stage_owner), device);
                const std::string rebalance_apply_dependency =
                    maybeInsertGraphSideRebalance(
                        "standard routed expert path",
                        prefix + "moe_expert_ffn");
                graph.addDependency(
                    prefix + "moe_expert_ffn",
                    rebalance_apply_dependency.empty()
                        ? prefix + "moe_routing"
                        : rebalance_apply_dependency);
                ffn_terminal = prefix + "moe_expert_ffn";

                if (standard_cpu_canonical_publication)
                {
                    /*
                     * CPU Dynamic ownership changes which participant computes
                     * each route, but it must not change either arithmetic or
                     * communication volume by `top_k`. Each participant has
                     * already emitted only its live raw route rows as indexed
                     * packed records. Gather those records to rank zero, replay
                     * the original increasing-slot weighted FMA exactly once,
                     * then broadcast only the compact final row.
                     */
                    auto rebalance_sidebands =
                        takeGraphRebalanceSidebandsForAllreduce();
                    if (!rebalance_sidebands.empty() ||
                        !graph_rebalance_collect_node.empty() ||
                        graph_rebalance_plan_after_sideband_params.has_value() ||
                        graph_rebalance_pack_payload_params.has_value() ||
                        graph_rebalance_unpack_payload_params.has_value())
                    {
                        throw std::logic_error(
                            "Qwen35 MoE CPU packed route publication cannot "
                            "consume GPU graph-side rebalance sidebands for layer " +
                            std::to_string(layer_idx));
                    }

                    constexpr int canonical_root_participant = 0;
                    MoECanonicalRouteGatherStage::Params gather_params;
                    gather_params.device_id = device;
                    gather_params.tp_ctx = config_.tp_ctx;
                    gather_params.packed_route_records =
                        canonical_route_contributions;
                    gather_params.seq_len = total_tokens;
                    gather_params.top_k = config_.moe.top_k;
                    gather_params.d_model = config_.d_model;
                    gather_params.root_participant =
                        canonical_root_participant;
                    gather_params.stage_name =
                        prefix + "moe_canonical_routes_gather_to_root";
                    gather_params.packed_route_records_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    const std::string gather_name = gather_params.stage_name;
                    graph.addNode(
                        gather_name,
                        ComputeStageFactory::createMoECanonicalRouteGather(
                            gather_params),
                        device);
                    graph.addDependency(
                        gather_name,
                        prefix + "moe_expert_ffn");

                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.routing_weights = routing_weights;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            UnweightedExpertRowThenOrderedFMA;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            PackedIndexedRouteRows;
                    reduce_params.reduction_role =
                        config_.tp_device_idx ==
                                canonical_root_participant
                            ? MoECanonicalRouteReductionRole::RootOwner
                            : MoECanonicalRouteReductionRole::
                                  NonRootParticipant;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.routing_weights_buffer_id =
                        buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string reduce_name =
                        prefix + "moe_canonical_routes_ordered_fma";
                    graph.addNode(
                        reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(reduce_name, gather_name);

                    MoECanonicalOutputBroadcastStage::Params broadcast_params;
                    broadcast_params.device_id = device;
                    broadcast_params.tp_ctx = config_.tp_ctx;
                    broadcast_params.output = moe_output;
                    broadcast_params.seq_len = total_tokens;
                    broadcast_params.d_model = config_.d_model;
                    broadcast_params.root_participant =
                        canonical_root_participant;
                    broadcast_params.stage_name =
                        prefix + "moe_canonical_routes_broadcast";
                    broadcast_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    const std::string broadcast_name =
                        broadcast_params.stage_name;
                    graph.addNode(
                        broadcast_name,
                        ComputeStageFactory::
                            createMoECanonicalOutputBroadcast(
                                broadcast_params),
                        device);
                    graph.addDependency(broadcast_name, reduce_name);
                    ffn_terminal = broadcast_name;
                }
                else if (route_accumulation_policy ==
                    MoERouteAccumulationPolicy::
                        IndependentRouteSlotsThenOrderedFold)
                {
                    MoECanonicalRouteReduceStage::Params reduce_params;
                    reduce_params.device_id = device;
                    reduce_params.canonical_route_contributions =
                        canonical_route_contributions;
                    reduce_params.output = moe_output;
                    reduce_params.seq_len = total_tokens;
                    reduce_params.top_k = config_.moe.top_k;
                    reduce_params.d_model = config_.d_model;
                    reduce_params.canonical_route_arithmetic =
                        MoECanonicalRouteArithmeticPolicy::
                            PreweightedContributionThenOrderedAdd;
                    reduce_params.canonical_route_layout =
                        MoECanonicalRoutePublicationLayout::
                            DenseOriginalRouteSlots;
                    reduce_params.reduction_role =
                        MoECanonicalRouteReductionRole::RootOwner;
                    reduce_params.canonical_route_contributions_buffer_id =
                        buffers.idFor(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS);
                    reduce_params.output_buffer_id =
                        buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);

                    const std::string reduce_name =
                        prefix + "moe_local_canonical_routes_reduce";
                    graph.addNode(
                        reduce_name,
                        ComputeStageFactory::createMoECanonicalRouteReduce(
                            reduce_params),
                        device);
                    graph.addDependency(
                        reduce_name,
                        prefix + "moe_expert_ffn");
                    ffn_terminal = reduce_name;
                }

                LOG_TRACE(
                    "[Qwen35MoEGraph] Layer " << layer_idx
                                              << " route_accumulation="
                                              << moeRouteAccumulationPolicyToString(
                                                     route_accumulation_policy)
                                              << " device="
                                              << device.to_string());

                // Qwen35 MoE expert weights are normally replicated, so every rank
                // computes the full routed-expert contribution. GPU and legacy
                // non-participant partial paths retain their compact collective;
                // CPU participant ownership was lowered canonically above.
                if (!standard_cpu_canonical_publication &&
                    routed_expert_output_is_partial &&
                    needsTPAllreduce())
                {
                    size_t allreduce_count = static_cast<size_t>(total_tokens) * static_cast<size_t>(config_.d_model);
                    std::string ar_name = prefix + "moe_expert_allreduce";
                    auto rebalance_sidebands = takeGraphRebalanceSidebandsForAllreduce();
                    auto allreduce_stage = createTPAllreduceStage(
                        moe_output, allreduce_count, device, layer_idx,
                        /*is_attention=*/false, ar_name, buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                        std::move(rebalance_sidebands));
                    if (allreduce_stage)
                    {
                        graph.addNode(ar_name, std::move(allreduce_stage), device);
                        graph.addDependency(ar_name, prefix + "moe_expert_ffn");
                        ffn_terminal = ar_name;
                        maybeAddGraphRebalancePayloadStageAfterSideband(
                            ar_name,
                            ffn_terminal);
                    }
                }

            }
        }

        {
            auto ffn_ctx = makeFFNBuildContext({}, {}, {});
            buildFFNSharedAndFinalizeSection(ffn_ctx);
        }

        graph.setTerminalNode(ffn_terminal);
        return graph;
    }
} // namespace llaminar2
