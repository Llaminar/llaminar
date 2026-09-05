/**
 * @file NodeExpertOverlayParityRoutingEvidence.cpp
 * @brief RoutingEvidence implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Resolve and cross-check the request-pinned epoch on every MoE layer.
     *
     * @return One model-wide execution identity, or no value after recording a
     *         focused test failure for missing, malformed, or split evidence.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::pinnedDevicePlacementEpochEvidence() const -> std::optional<PinnedDevicePlacementEpochEvidence>
    {
        std::optional<PinnedDevicePlacementEpochEvidence> model_identity;
        for (int layer = 0; layer < parityLayerCount(); ++layer)
        {
            const std::string prefix =
                "layer" + std::to_string(layer) + '_';
            size_t selected_bank_elements = 0u;
            const float *const selected_bank_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_SELECTED_BANK",
                selected_bank_elements);
            size_t bank0_epoch_elements = 0u;
            const float *const bank0_epoch_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_BANK0_EPOCH",
                bank0_epoch_elements);
            size_t bank1_epoch_elements = 0u;
            const float *const bank1_epoch_value = activeSnapshot(
                prefix + "MOE_OVERLAY_ROUTE_BANK1_EPOCH",
                bank1_epoch_elements);
            if (!selected_bank_value || !bank0_epoch_value ||
                !bank1_epoch_value || selected_bank_elements != 1u ||
                bank0_epoch_elements != 1u || bank1_epoch_elements != 1u)
            {
                ADD_FAILURE()
                    << prefix
                    << "did not publish one complete request-pinned epoch identity";
                return std::nullopt;
            }

            const float selected = selected_bank_value[0];
            if (!std::isfinite(selected) ||
                (selected != 0.0f && selected != 1.0f))
            {
                ADD_FAILURE()
                    << prefix << "published invalid selected bank " << selected;
                return std::nullopt;
            }
            const int selected_bank = static_cast<int>(selected);
            const float selected_epoch = selected_bank == 0
                                             ? bank0_epoch_value[0]
                                             : bank1_epoch_value[0];
            if (!std::isfinite(selected_epoch) || selected_epoch <= 0.0f ||
                std::trunc(selected_epoch) != selected_epoch ||
                selected_epoch > static_cast<float>(
                                     std::numeric_limits<int32_t>::max()))
            {
                ADD_FAILURE()
                    << prefix << "selected bank " << selected_bank
                    << " published invalid epoch " << selected_epoch;
                return std::nullopt;
            }
            const PinnedDevicePlacementEpochEvidence layer_identity{
                .epoch = static_cast<uint64_t>(selected_epoch),
                .selected_bank = selected_bank,
            };
            if (!model_identity)
            {
                model_identity = layer_identity;
                continue;
            }
            if (model_identity->epoch != layer_identity.epoch ||
                model_identity->selected_bank !=
                    layer_identity.selected_bank)
            {
                ADD_FAILURE()
                    << prefix << "consumed placement epoch "
                    << layer_identity.epoch << " bank "
                    << layer_identity.selected_bank
                    << " but prior model layers consumed epoch "
                    << model_identity->epoch << " bank "
                    << model_identity->selected_bank;
                return std::nullopt;
            }
        }
        return model_identity;
    }

    /**
     * @brief Resolve both typed route projections from live device checkpoints.
     *
     * @param production_stage_prefix Exact live snapshot prefix ending in `_`.
     * @param expected_route_count Router slots in the current checkpoint.
     * @param expected_expert_count Logical routed experts in the model layer.
     * @return Complete evidence, or no value after recording a test failure.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::pinnedDeviceRouteEvidence(
        std::string_view production_stage_prefix,
        size_t expected_route_count,
        size_t expected_expert_count) const -> std::optional<PinnedDeviceRouteEvidence>
    {
        if (production_stage_prefix.empty() ||
            production_stage_prefix.back() != '_')
        {
            ADD_FAILURE()
                << "Pinned route evidence received an invalid snapshot prefix";
            return std::nullopt;
        }
        const std::string prefix(production_stage_prefix);
        size_t domain_elements = 0u;
        const float *const domain_participants = activeSnapshot(
            prefix + "MOE_DOMAIN_ROUTE_PARTICIPANT_IDS",
            domain_elements);
        size_t runtime_weight_elements = 0u;
        const float *const runtime_weights = activeSnapshot(
            prefix + "MOE_RUNTIME_ROUTE_WEIGHTS",
            runtime_weight_elements);
        size_t bank0_elements = 0u;
        const float *const bank0 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK0",
            bank0_elements);
        size_t bank1_elements = 0u;
        const float *const bank1 = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_PARTICIPANTS_BANK1",
            bank1_elements);
        size_t selected_bank_elements = 0u;
        const float *const selected_bank_value = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_SELECTED_BANK",
            selected_bank_elements);
        size_t bank0_epoch_elements = 0u;
        const float *const bank0_epoch_value = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_BANK0_EPOCH",
            bank0_epoch_elements);
        size_t bank1_epoch_elements = 0u;
        const float *const bank1_epoch_value = activeSnapshot(
            prefix + "MOE_OVERLAY_ROUTE_BANK1_EPOCH",
            bank1_epoch_elements);
        if (!domain_participants || !runtime_weights || !bank0 || !bank1 ||
            !selected_bank_value || !bank0_epoch_value ||
            !bank1_epoch_value)
        {
            ADD_FAILURE()
                << prefix
                << "mapped reducer did not publish both pinned route projections";
            return std::nullopt;
        }
        if (domain_elements != expected_route_count ||
            runtime_weight_elements != expected_route_count ||
            bank0_elements != expected_expert_count ||
            bank1_elements != expected_expert_count ||
            selected_bank_elements != 1u || bank0_epoch_elements != 1u ||
            bank1_epoch_elements != 1u)
        {
            ADD_FAILURE()
                << prefix << "route evidence geometry mismatch: domain="
                << domain_elements << " expected_routes="
                << expected_route_count << " bank0=" << bank0_elements
                << " runtime_weights=" << runtime_weight_elements
                << " bank1=" << bank1_elements << " expected_experts="
                << expected_expert_count << " selected_bank_elements="
                << selected_bank_elements << " bank0_epoch_elements="
                << bank0_epoch_elements << " bank1_epoch_elements="
                << bank1_epoch_elements;
            return std::nullopt;
        }
        const float selected = selected_bank_value[0];
        if (!std::isfinite(selected) ||
            (selected != 0.0f && selected != 1.0f))
        {
            ADD_FAILURE()
                << prefix << "acquired route epoch selected invalid bank "
                << selected;
            return std::nullopt;
        }
        const int selected_bank = static_cast<int>(selected);
        const float selected_epoch = selected_bank == 0
                                         ? bank0_epoch_value[0]
                                         : bank1_epoch_value[0];
        if (!std::isfinite(selected_epoch) || selected_epoch <= 0.0f ||
            std::trunc(selected_epoch) != selected_epoch ||
            selected_epoch > static_cast<float>(
                                 std::numeric_limits<int32_t>::max()))
        {
            ADD_FAILURE()
                << prefix << "selected bank " << selected_bank
                << " published invalid epoch " << selected_epoch;
            return std::nullopt;
        }
        return PinnedDeviceRouteEvidence{
            .overlay_participants = selected_bank == 0 ? bank0 : bank1,
            .domain_participants = domain_participants,
            .runtime_weights = runtime_weights,
            .expert_count = expected_expert_count,
            .route_count = expected_route_count,
            .epoch = static_cast<uint64_t>(selected_epoch),
            .selected_bank = selected_bank,
        };
    }

    /**
     * @brief Prove one live MTP sidecar consumed its request-pinned route bank.
     *
     * The ordinary parity forward cannot prove this value because speculative
     * sidecars do not execute in teacher-forced prefill/decode. This check runs
     * immediately after a real grouped transaction while its context-qualified
     * snapshots are live. It binds numerical router output to the exact RCU
     * bank/epoch and final domain assignment used by sparse dispatch.
     *
     * @param production_stage_prefix Context-qualified `MTP0_` prefix.
     * @param expected_epoch Main-model placement epoch for the transaction.
     * @param top_k Number of live router slots in the one-row sidecar.
     * @param num_experts Logical routed-expert count.
     * @return True only when every route-authority value is complete and valid.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::validatePinnedMTPSidecarRouteEvidence(
        std::string_view production_stage_prefix,
        uint64_t expected_epoch,
        int top_k,
        int num_experts) const -> bool
    {
        if (top_k <= 0 || num_experts <= 0)
        {
            ADD_FAILURE()
                << "MTP sidecar route proof received invalid MoE geometry";
            return false;
        }
        const auto evidence = pinnedDeviceRouteEvidence(
            production_stage_prefix,
            static_cast<size_t>(top_k),
            static_cast<size_t>(num_experts));
        if (!evidence)
            return false;
        if (evidence->epoch != expected_epoch)
        {
            ADD_FAILURE()
                << production_stage_prefix << "consumed placement epoch "
                << evidence->epoch << " while its grouped main-model "
                << "transaction consumed epoch " << expected_epoch;
            return false;
        }

        const std::string prefix(production_stage_prefix);
        size_t routing_index_elements = 0u;
        const float *const routing_indices = activeSnapshot(
            prefix + "MOE_ROUTING_INDICES",
            routing_index_elements);
        size_t routing_weight_elements = 0u;
        const float *const routing_weights = activeSnapshot(
            prefix + "MOE_ROUTING_WEIGHTS",
            routing_weight_elements);
        if (!routing_indices || !routing_weights ||
            routing_index_elements != evidence->route_count ||
            routing_weight_elements != evidence->route_count)
        {
            ADD_FAILURE()
                << prefix
                << "did not publish router values matching its pinned route ledger";
            return false;
        }

        const size_t participant_count = activeOverlayParticipantCount();
        if (participant_count == 0u)
        {
            ADD_FAILURE()
                << prefix << "has no typed ExpertOverlay participants";
            return false;
        }
        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto residency = concrete
                                   ? concrete
                                         ->expertOverlayResidencySnapshotForDiagnostics()
                                   : nullptr;
        if (!residency || !residency->valid() || !overlay_plan_)
        {
            ADD_FAILURE()
                << prefix
                << "cannot resolve the typed continuation-root route projection";
            return false;
        }
        const int continuation_root =
            overlay_plan_->continuation_domain_spec.logical_root_participant;
        const auto *const continuation_root_endpoint =
            residency->owner_map.participantForId(continuation_root);
        if (!continuation_root_endpoint ||
            continuation_root_endpoint->domain_name !=
                overlay_plan_->continuation_domain ||
            continuation_root_endpoint->domain_participant_index < 0)
        {
            ADD_FAILURE()
                << prefix << "has no valid continuation-root endpoint for "
                << continuation_root;
            return false;
        }
        const int reducer_domain_participant =
            continuation_root_endpoint->domain_participant_index;
        for (size_t expert = 0u; expert < evidence->expert_count; ++expert)
        {
            const float participant = evidence->overlay_participants[expert];
            if (!std::isfinite(participant) ||
                std::trunc(participant) != participant ||
                participant < 0.0f ||
                participant >= static_cast<float>(participant_count))
            {
                ADD_FAILURE()
                    << prefix << "placement bank contains invalid participant "
                    << participant << " for expert " << expert;
                return false;
            }
        }
        for (size_t route = 0u; route < evidence->route_count; ++route)
        {
            const float expert = routing_indices[route];
            const float domain_participant =
                evidence->domain_participants[route];
            const float router_weight = routing_weights[route];
            const float runtime_weight = evidence->runtime_weights[route];
            if (!std::isfinite(expert) || std::trunc(expert) != expert ||
                expert < 0.0f ||
                expert >= static_cast<float>(evidence->expert_count))
            {
                ADD_FAILURE()
                    << prefix << "router published invalid expert " << expert
                    << " at route " << route;
                return false;
            }
            if (!std::isfinite(domain_participant) ||
                std::trunc(domain_participant) != domain_participant ||
                domain_participant < -1.0f ||
                domain_participant >=
                    static_cast<float>(participant_count))
            {
                ADD_FAILURE()
                    << prefix << "domain ledger published invalid participant "
                    << domain_participant << " at route " << route;
                return false;
            }
            if (!std::isfinite(router_weight) || router_weight < 0.0f ||
                !std::isfinite(runtime_weight))
            {
                ADD_FAILURE()
                    << prefix << "route " << route
                    << " published invalid router/runtime weights: router="
                    << router_weight << " runtime=" << runtime_weight;
                return false;
            }

            const int expert_id = static_cast<int>(expert);
            const int global_participant = static_cast<int>(
                evidence->overlay_participants[expert_id]);
            const auto *const global_endpoint =
                residency->owner_map.participantForId(global_participant);
            if (!global_endpoint)
            {
                ADD_FAILURE()
                    << prefix << "route " << route << " expert " << expert_id
                    << " names unknown global participant "
                    << global_participant;
                return false;
            }

            const bool belongs_to_continuation =
                global_endpoint->domain_name ==
                overlay_plan_->continuation_domain;
            const int assigned_domain_participant =
                static_cast<int>(domain_participant);
            if (!belongs_to_continuation)
            {
                if (assigned_domain_participant != -1 ||
                    runtime_weight != 0.0f)
                {
                    ADD_FAILURE()
                        << prefix << "external-domain route " << route
                        << " must publish the -1 domain sentinel and zero "
                           "continuation-local weight: domain_participant="
                        << assigned_domain_participant
                        << " runtime_weight=" << runtime_weight;
                    return false;
                }
                continue;
            }

            const auto assigned_endpoint = std::find_if(
                residency->owner_map.participants().begin(),
                residency->owner_map.participants().end(),
                [&](const MoEExpertOwnerParticipant &candidate)
                {
                    return candidate.domain_name ==
                               overlay_plan_->continuation_domain &&
                           candidate.domain_participant_index ==
                               assigned_domain_participant;
                });
            if (assigned_endpoint ==
                residency->owner_map.participants().end())
            {
                ADD_FAILURE()
                    << prefix << "continuation route " << route
                    << " names unknown domain participant "
                    << assigned_domain_participant;
                return false;
            }
            if (!isLLEPProductionTest() &&
                assigned_endpoint->participant_id != global_participant)
            {
                ADD_FAILURE()
                    << prefix << "Static/Dynamic route " << route
                    << " disagrees between global placement participant "
                    << global_participant << " and continuation-domain "
                    << "participant " << assigned_domain_participant;
                return false;
            }

            /* Static/Dynamic grouping retains the raw FP32 router weight only
             * on the reducer participant that executes the route. Other
             * continuation shards publish exact zero while preserving the
             * final participant ledger for mapped sparse reduction. LLEP keeps
             * the raw weight beside every post-assignment ledger and filters
             * through the participant id during regrouping instead. */
            const float expected_runtime_weight =
                isLLEPProductionTest() ||
                        assigned_domain_participant ==
                            reducer_domain_participant
                    ? router_weight
                    : 0.0f;
            if (runtime_weight != expected_runtime_weight)
            {
                ADD_FAILURE()
                    << prefix << "route " << route
                    << " runtime weight does not match its typed execution "
                       "projection: router="
                    << router_weight << " runtime=" << runtime_weight
                    << " expected=" << expected_runtime_weight
                    << " expert=" << expert_id
                    << " global_participant=" << global_participant
                    << " assigned_domain_participant="
                    << assigned_domain_participant
                    << " reducer_domain_participant="
                    << reducer_domain_participant;
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Attribute live parity router checkpoints to the published epoch.
     *
     * Snapshot values are exact integer expert ids produced by the real router.
     * Pair them with the request-selected global placement bank and the final
     * domain-local schedule while all checkpoints remain live, before the
     * parity harness clears diagnostics. The setup-time residency snapshot
     * supplies only stable endpoint topology; it is deliberately not used to
     * reconstruct live placement after Dynamic movement.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::cacheDeviceRouteAssignmentEvidence() -> void
    {
        if (!isRootParityRank())
            return;

        auto *const concrete =
            dynamic_cast<OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        if (!snapshot || !snapshot->valid())
        {
            ADD_FAILURE()
                << "Production parity could not inspect its immutable live ExpertOverlay residency epoch";
            return;
        }

        const size_t participant_count = activeOverlayParticipantCount();
        if (snapshot->owner_map.participants().size() != participant_count)
        {
            ADD_FAILURE()
                << "Published ExpertOverlay participant cardinality changed before route attribution";
            return;
        }

        std::vector<uint64_t> route_counts(participant_count, 0u);
        std::vector<bool> requires_remote_completion(
            participant_count, false);
        std::vector<PublishedParticipantResidency> resident_participants(
            participant_count,
            PublishedParticipantResidency::Idle);
        for (const auto &participant : snapshot->owner_map.participants())
        {
            if (participant.participant_id < 0 ||
                static_cast<size_t>(participant.participant_id) >=
                    participant_count ||
                !participant.world_rank_known)
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay participant identity is incomplete";
                return;
            }
            requires_remote_completion[
                static_cast<size_t>(participant.participant_id)] =
                participant.world_rank != parityArtifactAuthorityRank();
        }

        size_t checkpoint_layers = 0u;
        for (const auto &placement :
             snapshot->placement_plan->placements)
        {
            if (placement.layer < 0 ||
                placement.routed_expert_tier.empty())
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay placement contains invalid layer geometry";
                return;
            }
            size_t route_elements = 0u;
            const std::string prefix =
                "layer" + std::to_string(placement.layer) + '_';
            const std::string key = prefix + "MOE_ROUTING_INDICES";
            const float *const routes = activeSnapshot(key, route_elements);
            if (!routes)
                continue;
            const auto route_evidence = pinnedDeviceRouteEvidence(
                prefix,
                route_elements,
                placement.routed_expert_tier.size());
            if (!route_evidence)
                return;
            try
            {
                includePublishedExpertOwners(
                    resident_participants,
                    std::span<const float>(
                        route_evidence->overlay_participants,
                        route_evidence->expert_count));
            }
            catch (const std::invalid_argument &error)
            {
                ADD_FAILURE()
                    << "Published ExpertOverlay placement bank is invalid at layer "
                    << placement.layer << ": " << error.what();
                return;
            }
            ++checkpoint_layers;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                const float routed = routes[index];
                if (!std::isfinite(routed) || routed < 0.0f ||
                    routed > static_cast<float>(std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert ID "
                        << routed << " at element " << index;
                    return;
                }

                // Validate the floating snapshot before converting it: a cast
                // of NaN or an out-of-range float to int is undefined behavior.
                const int expert = static_cast<int>(routed);
                if (routed != static_cast<float>(expert) ||
                    static_cast<size_t>(expert) >=
                        placement.routed_expert_tier.size())
                {
                    ADD_FAILURE()
                        << "Router checkpoint " << key
                        << " contains a non-integral or out-of-range expert id";
                    return;
                }
                const float global_assigned =
                    route_evidence->overlay_participants[expert];
                if (!std::isfinite(global_assigned) ||
                    global_assigned < 0.0f || global_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " contains an invalid global participant for expert "
                        << expert;
                    return;
                }
                const int participant = static_cast<int>(global_assigned);
                const auto *const global_endpoint =
                    snapshot->owner_map.participantForId(participant);
                if (global_assigned != static_cast<float>(participant) ||
                    participant < 0 ||
                    static_cast<size_t>(participant) >= participant_count ||
                    !global_endpoint)
                {
                    ADD_FAILURE()
                        << "Pinned overlay placement for " << key
                        << " names unknown global participant "
                        << global_assigned;
                    return;
                }

                const float domain_assigned =
                    route_evidence->domain_participants[index];
                if (!std::isfinite(domain_assigned) ||
                    domain_assigned < -1.0f || domain_assigned >
                        static_cast<float>(
                            std::numeric_limits<int>::max()))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains invalid participant " << domain_assigned
                        << " at element " << index;
                    return;
                }
                const int domain_participant =
                    static_cast<int>(domain_assigned);
                if (domain_assigned !=
                    static_cast<float>(domain_participant))
                {
                    ADD_FAILURE()
                        << "Domain route schedule for " << key
                        << " contains non-integral participant "
                        << domain_assigned;
                    return;
                }
                const bool belongs_to_continuation =
                    global_endpoint->domain_name ==
                    overlay_plan_->continuation_domain;
                if (!belongs_to_continuation && domain_participant != -1)
                {
                    ADD_FAILURE()
                        << "Remote-domain expert " << expert << " in " << key
                        << " must retain the -1 domain-route sentinel, observed "
                        << domain_participant;
                    return;
                }
                if (belongs_to_continuation)
                {
                    const auto local_endpoint = std::find_if(
                        snapshot->owner_map.participants().begin(),
                        snapshot->owner_map.participants().end(),
                        [&](const MoEExpertOwnerParticipant &candidate)
                        {
                            return candidate.domain_name ==
                                       overlay_plan_->continuation_domain &&
                                   candidate.domain_participant_index ==
                                       domain_participant;
                        });
                    if (domain_participant < 0 ||
                        local_endpoint ==
                            snapshot->owner_map.participants().end())
                    {
                        ADD_FAILURE()
                            << "Continuation-domain expert " << expert
                            << " in " << key
                            << " names unknown domain participant "
                            << domain_participant;
                        return;
                    }
                }
                ++route_counts[static_cast<size_t>(participant)];
            }
        }
        if (checkpoint_layers == 0u)
        {
            ADD_FAILURE()
                << "Production parity retained no MoE routing checkpoint for live-epoch attribution";
            return;
        }
        parity_route_counts_by_participant_ = std::move(route_counts);
        parity_route_requires_remote_completion_ =
            std::move(requires_remote_completion);
        parity_residency_by_participant_ =
            std::move(resident_participants);
    }

    /**
     * @brief Persist immutable setup ownership and participant topology.
     *
     * The host residency snapshot is the cold-start topology authority. In an
     * all-GPU Dynamic cell it intentionally does not shadow later device-owned
     * placement epochs. Keep the established `expert_owner_map.csv` artifact
     * as the setup baseline and endpoint dictionary; exact live assignments
     * are recorded per route in `prefill_routed_expert_routes.csv` from the
     * reducer's device ledger.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writeExpertOwnerTopologyBaselineCsv() const -> void
    {
        if (!isRootParityRank())
            return;

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto path = ensureResultsDir() / "expert_owner_map.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "epoch,layer,expert,tier_index,tier_name,domain_name,"
               "participant,domain_participant,world_rank,device,resident\n";
        for (const auto &owner : snapshot->owner_map.owners())
        {
            output
                << snapshot->epoch << ',' << owner.layer_idx << ','
                << owner.expert_id << ',' << owner.tier_idx << ','
                << owner.tier_name << ',' << owner.domain_name << ','
                << owner.owner_participant << ','
                << owner.domain_participant_index << ','
                << owner.owner_world_rank << ',' << owner.device.toString()
                << ',' << (owner.resident ? 1 : 0) << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
    }

    /**
     * @brief Persist value-level routed-expert evidence for baseline and moves.
     *
     * Aggregate cosine metrics cannot distinguish a missing participant from
     * a correct route computed with the wrong weight slice.  The baseline
     * layer and every layer containing a committed promotion are therefore
     * recorded element by element against Hugging Face.  Per-route norms also
     * expose a zero, duplicated, or explosive migrated contribution without
     * requiring another instrumented inference run.  This diagnostic executes
     * only after the captured production forward and before snapshot teardown.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writePrefillRoutedExpertDiagnosticCsv() -> void
    {
        if (!isRootParityRank())
            return;

        std::vector<int> diagnostic_layers{0};
        for (const auto &promotion : promoted_experts_)
        {
            if (promotion.layer >= 0 &&
                promotion.layer < parityLayerCount() &&
                std::find(
                    diagnostic_layers.begin(),
                    diagnostic_layers.end(),
                    promotion.layer) == diagnostic_layers.end())
            {
                diagnostic_layers.push_back(promotion.layer);
            }
        }
        /*
         * The synthetic MTP sidecar layer does not execute during prefill and
         * therefore has no `layerN_` checkpoint in this bank. Its exact routed
         * values are retained later by the typed MTP comparison callback and
         * exported through the moved-expert evidence artifacts.
         */
        std::sort(diagnostic_layers.begin(), diagnostic_layers.end());

        const auto values_path =
            ensureResultsDir() / "prefill_routed_expert_values.csv";
        std::ofstream values(values_path, std::ios::trunc);
        ASSERT_TRUE(values.is_open()) << values_path;
        values << std::setprecision(9);
        values
            << "layer,row,column,llaminar,pytorch,difference,"
               "continuation_local_sum\n";

        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto snapshot = concrete
                                  ? concrete->expertOverlayResidencySnapshotForDiagnostics()
                                  : nullptr;
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        const auto routes_path =
            ensureResultsDir() / "prefill_routed_expert_routes.csv";
        std::ofstream route_output(routes_path, std::ios::trunc);
        ASSERT_TRUE(route_output.is_open()) << routes_path;
        route_output
            << "layer,row,slot,expert,weight,runtime_weight,participant,domain_participant,"
               "selected_placement_bank,tier_index,"
               "domain_name,world_rank,device,canonical_l2,"
               "canonical_max_abs,canonical_nonzero,canonical_first\n";
        route_output << std::setprecision(9);

        const auto *const model = activeModelContextForDiagnostics();
        ASSERT_NE(model, nullptr);
        const size_t width = static_cast<size_t>(
            model->model().embedding_length);
        ASSERT_GT(width, 0u);

        for (const int diagnostic_layer : diagnostic_layers)
        {
            const std::string prefix =
                "layer" + std::to_string(diagnostic_layer) + '_';
            size_t actual_elements = 0u;
            const float *const actual = activeSnapshot(
                prefix + "MOE_EXPERT_OUTPUT", actual_elements);
            const auto reference = loadPyTorchSnapshot(
                prefix + "MOE_EXPERT_OUTPUT");
            ASSERT_NE(actual, nullptr) << prefix;
            ASSERT_EQ(actual_elements, reference.size()) << prefix;
            ASSERT_EQ(actual_elements % width, 0u) << prefix;
            const size_t rows = actual_elements / width;

            size_t route_elements = 0u;
            size_t weight_elements = 0u;
            const float *const routes = activeSnapshot(
                prefix + "MOE_ROUTING_INDICES", route_elements);
            const float *const weights = activeSnapshot(
                prefix + "MOE_ROUTING_WEIGHTS", weight_elements);
            ASSERT_NE(routes, nullptr) << prefix;
            ASSERT_NE(weights, nullptr) << prefix;
            ASSERT_EQ(route_elements, weight_elements) << prefix;
            ASSERT_GT(rows, 0u) << prefix;
            ASSERT_EQ(route_elements % rows, 0u) << prefix;
            const size_t top_k = route_elements / rows;
            const auto placement = std::find_if(
                snapshot->placement_plan->placements.begin(),
                snapshot->placement_plan->placements.end(),
                [diagnostic_layer](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == diagnostic_layer; });
            ASSERT_NE(
                placement,
                snapshot->placement_plan->placements.end())
                << prefix << " has no declared routed-expert placement";
            const auto route_evidence = pinnedDeviceRouteEvidence(
                prefix,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value()) << prefix;

            size_t canonical_elements = 0u;
            const float *const canonical = activeSnapshot(
                prefix + "MOE_ROUTE_CONTRIBUTIONS",
                canonical_elements);
            ASSERT_NE(canonical, nullptr)
                << prefix
                << " continuation did not retain canonical route-slot evidence";
            ASSERT_EQ(canonical_elements, route_elements * width) << prefix;

            std::vector<float> continuation_local_sum(
                actual_elements, 0.0f);
            for (size_t row = 0u; row < rows; ++row)
            {
                for (size_t slot = 0u; slot < top_k; ++slot)
                {
                    const size_t index = row * top_k + slot;
                    const size_t route_offset = index * width;
                    const size_t output_offset = row * width;
                    double squared_norm = 0.0;
                    float max_abs = 0.0f;
                    size_t nonzero = 0u;
                    for (size_t column = 0u; column < width; ++column)
                    {
                        const float contribution =
                            canonical[route_offset + column];
                        continuation_local_sum[output_offset + column] +=
                            contribution;
                        squared_norm += static_cast<double>(contribution) *
                                        static_cast<double>(contribution);
                        max_abs = std::max(max_abs, std::abs(contribution));
                        nonzero += contribution != 0.0f ? 1u : 0u;
                    }

                    ASSERT_TRUE(std::isfinite(routes[index]));
                    const int expert = static_cast<int>(routes[index]);
                    ASSERT_EQ(routes[index], static_cast<float>(expert));
                    ASSERT_GE(expert, 0);
                    ASSERT_LT(
                        static_cast<size_t>(expert),
                        route_evidence->expert_count);
                    const float global_assigned =
                        route_evidence->overlay_participants[expert];
                    ASSERT_TRUE(std::isfinite(global_assigned));
                    const int participant =
                        static_cast<int>(global_assigned);
                    ASSERT_EQ(
                        global_assigned,
                        static_cast<float>(participant))
                        << "Non-integral global route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float domain_assigned =
                        route_evidence->domain_participants[index];
                    ASSERT_TRUE(std::isfinite(domain_assigned));
                    const int domain_participant =
                        static_cast<int>(domain_assigned);
                    ASSERT_EQ(
                        domain_assigned,
                        static_cast<float>(domain_participant))
                        << "Non-integral domain route participant at layer "
                        << diagnostic_layer << " route slot " << index;
                    const float runtime_weight =
                        route_evidence->runtime_weights[index];
                    ASSERT_TRUE(std::isfinite(runtime_weight))
                        << "Non-finite runtime route weight at layer "
                        << diagnostic_layer << " route slot " << index;
                    const auto *const assigned_endpoint =
                        snapshot->owner_map.participantForId(participant);
                    ASSERT_NE(assigned_endpoint, nullptr)
                        << "Missing endpoint metadata for layer "
                        << diagnostic_layer << " expert " << expert
                        << " assigned participant " << participant;
                    route_output
                        << diagnostic_layer << ',' << row << ',' << slot
                        << ',' << expert << ',' << weights[index] << ','
                        << runtime_weight << ','
                        << participant << ',' << domain_participant << ','
                        << route_evidence->selected_bank << ','
                        << assigned_endpoint->tier_idx
                        << ',' << assigned_endpoint->domain_name << ','
                        << assigned_endpoint->world_rank << ','
                        << assigned_endpoint->device.toString() << ','
                        << std::sqrt(squared_norm) << ',' << max_abs << ','
                        << nonzero << ',' << canonical[route_offset] << '\n';
                }
            }

            for (size_t index = 0u; index < actual_elements; ++index)
            {
                values
                    << diagnostic_layer << ',' << index / width << ','
                    << index % width << ',' << actual[index] << ','
                    << reference[index] << ','
                    << (actual[index] - reference[index]) << ','
                    << continuation_local_sum[index] << '\n';
            }
        }
        values.flush();
        EXPECT_TRUE(values.good()) << values_path;
        route_output.flush();
        EXPECT_TRUE(route_output.good()) << routes_path;
    }

    /**
     * @brief Prove resident participants executed and capacity-idle ones did not.
     *
     * The device-owned placement bank determines whether each declared
     * endpoint owns any expert after automatic capacity resolution. Routed
     * checkpoints prove every selected endpoint was resident under that exact
     * epoch. Cross-rank residents additionally require endpoint-owned traffic
     * somewhere in the real production workload after both retained graphs
     * publish Complete. Residency alone cannot require one bounded prompt to
     * select an expert; an idle endpoint, however, can never appear in the
     * pinned route schedule. Local arithmetic remains covered by the same
     * layer/LM-head parity.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertActiveTierRouteEvidence() const -> void
    {
        struct ExpectedRoute
        {
            int participant;
            int tier;
            const char *device_kind;
            std::string label;
        };

        std::vector<ExpectedRoute> expected;
        if (!overlay_plan_)
        {
            ADD_FAILURE() << "Sparse-route proof has no active overlay plan";
            return;
        }
        int participant_id = 0;
        for (const auto &domain : overlay_plan_->domains)
        {
            const auto tier = std::find_if(
                overlay_plan_->routed_tiers.begin(),
                overlay_plan_->routed_tiers.end(),
                [&](const RoutedExpertTier &candidate)
                { return candidate.domain == domain.name; });
            if (tier == overlay_plan_->routed_tiers.end())
            {
                ADD_FAILURE()
                    << "Sparse-route domain has no integer-priority tier: "
                    << domain.name;
                return;
            }
            const int tier_index = static_cast<int>(std::distance(
                overlay_plan_->routed_tiers.begin(), tier));
            for (const auto &participant : domain.participants)
            {
                const char *kind = participant.isCUDA()
                                       ? "CUDA"
                                       : participant.isROCm() ? "ROCm" : "CPU";
                expected.push_back(ExpectedRoute{
                    .participant = participant_id++,
                    .tier = tier_index,
                    .device_kind = kind,
                    .label = domain.name + "/" + participant.toShortString(),
                });
            }
        }

        constexpr size_t kMalformedRecords = 0;
        constexpr size_t kSelectedStart = 1;
        const size_t completed_start = kSelectedStart + expected.size();
        std::vector<uint64_t> local(
            completed_start + expected.size(), 0u);
        if (isRootParityRank())
        {
            if (parity_route_counts_by_participant_.size() !=
                    expected.size() ||
                parity_route_requires_remote_completion_.size() !=
                    expected.size() ||
                parity_residency_by_participant_.size() !=
                    expected.size())
            {
                ++local[kMalformedRecords];
            }
            else
            {
                std::copy(
                    parity_route_counts_by_participant_.begin(),
                    parity_route_counts_by_participant_.end(),
                    local.begin() + kSelectedStart);
            }
        }
        for (const auto &record : PerfStatsCollector::snapshot({"forward_graph"}))
        {
            if (record.kind != PerfStatRecord::Kind::Counter ||
                record.domain != "forward_graph" ||
                record.name != "moe_overlay_local_expert_active_routes")
            {
                continue;
            }

            const auto tag = [&record](const char *name) -> const std::string *
            {
                const auto it = record.tags.find(name);
                return it == record.tags.end() ? nullptr : &it->second;
            };
            const auto completion = tag("completion");
            const auto device_kind = tag("device_kind");
            const auto identity_source = tag("identity_source");
            const auto participant = tag("participant");
            const auto tier = tag("tier");
            // PerfStats coalesces repeated calls with the same full tag set.
            // A completed-route record can therefore represent several live
            // sparse packets (for example one packet per MoE layer), rather
            // than one call.  `value` is their summed route count and `count`
            // is the number of completed packets, so requiring count==1 here
            // would reject the strongest possible production evidence.
            const bool legacy_completion =
                completion &&
                *completion == "local_expert_packet_complete" &&
                identity_source &&
                *identity_source == "sparse_collective_key";
            const bool device_epoch_completion =
                completion &&
                *completion == "device_owned_epoch_complete" &&
                identity_source &&
                *identity_source == "device_owned_activation_epoch";
            const bool record_contract_ok =
                record.phase == "moe_overlay" &&
                record.count > 0u &&
                record.value > 0.0 &&
                record.tags.count("generation") == 0u &&
                record.tags.count("logical_step") == 0u &&
                (legacy_completion || device_epoch_completion) &&
                participant && tier && device_kind;
            if (!record_contract_ok)
            {
                ++local[kMalformedRecords];
                continue;
            }

            auto route = std::find_if(
                expected.begin(),
                expected.end(),
                [&](const ExpectedRoute &candidate)
                {
                    return *participant == std::to_string(candidate.participant) &&
                           *tier == std::to_string(candidate.tier) &&
                           *device_kind == candidate.device_kind;
                });
            if (route == expected.end())
            {
                ++local[kMalformedRecords];
                continue;
            }

            const size_t destination = completed_start +
                static_cast<size_t>(std::distance(expected.begin(), route));
            local[destination] += static_cast<uint64_t>(record.value);
        }

        std::vector<uint64_t> global(local.size(), 0u);
        MPI_Allreduce(
            local.data(),
            global.data(),
            static_cast<int>(global.size()),
            MPI_UINT64_T,
            MPI_SUM,
            parityCoordinationCommunicator());

        if (!isRootParityRank())
            return;

        EXPECT_EQ(global[kMalformedRecords], 0u)
            << "Graph-native local-expert route evidence was malformed";
        for (size_t index = 0; index < expected.size(); ++index)
        {
            const bool owns_expert =
                parity_residency_by_participant_.size() == expected.size() &&
                parity_residency_by_participant_[index] ==
                    PublishedParticipantResidency::OwnsExpert;
            const bool requires_remote_completion =
                parity_route_requires_remote_completion_.size() ==
                    expected.size() &&
                parity_route_requires_remote_completion_[index];
            const auto verdict = validateParticipantRouteEvidence(
                owns_expert ? PublishedParticipantResidency::OwnsExpert
                            : PublishedParticipantResidency::Idle,
                global[kSelectedStart + index],
                requires_remote_completion,
                global[completed_start + index]);
            EXPECT_NE(
                verdict,
                PublishedParticipantRouteEvidence::IdleParticipantSelected)
                << expected[index].label << " participant p"
                << expected[index].participant
                << " was selected despite owning no expert in the pinned residency bank";
            EXPECT_NE(
                verdict,
                PublishedParticipantRouteEvidence::RemoteResidentNeverCompleted)
                << expected[index].label << " participant p"
                << expected[index].participant
                << " owns final experts but completed no real device-owned sparse traffic during the production workload";

            if (global[kSelectedStart + index] > 0u &&
                requires_remote_completion)
            {
                EXPECT_GT(global[completed_start + index], 0u)
                    << expected[index].label << " participant p"
                    << expected[index].participant
                    << " was selected by the pinned router but published no completed device-owned sparse traffic";
            }
        }
    }

}
