/**
 * @file NodeExpertOverlayParityPromotionEvidence.cpp
 * @brief PromotionEvidence implementation of the shared node-overlay parity fixture.
 *
 * Shared by the 35B and 122B generated node ExpertOverlay matrices. This
 * translation-unit boundary does not change request ownership, reference
 * mathematics, graph execution, or evidence publication ordering.
 */
#include "NodeExpertOverlayParityFixture.h"

namespace llaminar2::test::parity::qwen35moe::node_overlay
{

    /**
     * @brief Resolve route and input lineage for every compared routed layer.
     *
     * A current route-set difference preserves comparability for matched
     * per-route expert values because the layer input is still canonical. It
     * invalidates a dense aggregate as proof of an individual remote addend,
     * and it changes the residual consumed by every later layer. The shared
     * typed transition records all three cases explicitly. Ordered layer keys
     * make the rule independent of callback vector order.
     *
     * @param layers Complete per-layer comparison records for one checkpoint.
     * @return Input lineage keyed by model layer.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::routedExpertReferenceLineageByLayer(
        const std::vector<LayerStats> &layers) -> std::map<int, RoutedExpertReferenceLineage>
    {
        std::map<int, const LayerStats *> ordered_layers;
        for (const auto &layer : layers)
        {
            const bool inserted =
                ordered_layers.emplace(layer.layer_idx, &layer).second;
            if (!inserted)
            {
                throw std::logic_error(
                    "Duplicate layer in routed-expert parity lineage");
            }
        }

        std::map<int, RoutedExpertReferenceLineage> result;
        auto input_lineage = RoutedExpertReferenceLineage::Canonical;
        for (const auto &[layer_index, layer] : ordered_layers)
        {
            const auto routing = std::find_if(
                layer->stage_results.begin(),
                layer->stage_results.end(),
                [](const StageComparisonResult &stage)
                {
                    return extractStageType(stage.stage_name) ==
                           "MOE_ROUTING_INDICES";
                });
            const bool current_routes_equal =
                routing == layer->stage_results.end() ||
                routing->routing_overlap >= 1.0f - 1.0e-6f;
            const auto transition =
                advanceRoutedExpertReferenceLineage(
                    input_lineage,
                    current_routes_equal);
            result.emplace(layer_index, transition.current_layer);
            input_lineage = transition.next_layer;
        }
        return result;
    }

    /**
     * @brief Retain promotion identities before a parity collector reset.
     *
     * The parity harness resets live PerfStats between campaign phases so the
     * CSV for each numerical comparison has an unambiguous interval. Movement
     * is model-lifetime state and deliberately survives that reset. Preserve
     * only the immutable layer/expert identities from the production movement
     * ledger; routing values and numerical outputs are still read from the
     * later live graph checkpoints.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::cacheCommittedPromotionEvidence() -> void
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const auto ledger =
            orch_runner_->moeOptimizationMovementLedger();
        EXPECT_TRUE(ledger.complete())
            << "Production movement authority discarded "
            << ledger.discarded_edges
            << " typed edges before parity attribution";
        const auto &origin =
            dynamic_residency_proof_lifecycle_.convergenceOrigin();
        ASSERT_LE(origin.ledger_edges, ledger.edges.size());
        ASSERT_FALSE(authenticated_movement_routes_.empty());
        size_t malformed_edges = 0;
        for (std::size_t edge_index = origin.ledger_edges;
             edge_index < ledger.edges.size(); ++edge_index)
        {
            const auto &edge = ledger.edges[edge_index];
            if (!edge.valid())
            {
                ++malformed_edges;
                continue;
            }
            if (edge.direction !=
                MoEOptimizationMovementDirection::Promotion)
            {
                continue;
            }
            if (edge.authority == MoEOptimizationAuthority::Host &&
                edge.activation_count == 0u)
            {
                continue;
            }
            if (edge.layer < 0 ||
                static_cast<std::size_t>(edge.layer) >=
                    authenticated_movement_routes_.size())
            {
                /* The movement-proof corpus is main-model prefill. A
                 * sidecar-only promotion remains in expert_movement.csv but
                 * cannot be its numerical witness. */
                continue;
            }
            const auto &layer_routes = authenticated_movement_routes_[
                static_cast<std::size_t>(edge.layer)];
            if (edge.expert < 0 ||
                static_cast<std::size_t>(edge.expert) >=
                    layer_routes.size())
            {
                ++malformed_edges;
                continue;
            }
            if (layer_routes[static_cast<std::size_t>(edge.expert)] == 0u)
            {
                /* Service-economy movement is real and remains exported, but
                 * this fixed mathematical corpus cannot execute it. */
                continue;
            }

            const PromotedExpert promotion{
                .layer = edge.layer,
                .expert = edge.expert,
                .destination_participant = edge.destination_participant,
                .candidate_epoch = edge.candidate_epoch,
            };
            if (std::find(
                    promoted_experts_.begin(),
                    promoted_experts_.end(),
                    promotion) == promoted_experts_.end())
            {
                promoted_experts_.push_back(promotion);
            }
        }

        EXPECT_EQ(malformed_edges, 0u)
            << "Typed production movement ledger contained malformed edges";
    }

    /**
     * @brief Persist the authenticated physical movement ledger used by parity.
     *
     * Numerical CSVs name the layer and routed expert that diverged, but that is
     * not enough to diagnose a moved-weight defect: the production transaction
     * may have crossed a rank, backend, or numeric-priority boundary.  Export the
     * authority's typed edge ledger before the parity harness resets optional
     * telemetry. The CSV serializes authoritative state and never reconstructs
     * placement from PerfStats tags.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writeCommittedMovementEvidenceCsv() const -> void
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_NE(orch_runner_, nullptr);
        const auto ledger =
            orch_runner_->moeOptimizationMovementLedger();
        EXPECT_TRUE(ledger.complete())
            << "Cannot export a truncated authoritative movement ledger";
        const auto path = ensureResultsDir() / "expert_movement.csv";
        std::ofstream output(path, std::ios::trunc);
        ASSERT_TRUE(output.is_open()) << path;
        output
            << "domain,name,count,value,transaction,candidate_epoch,layer,expert,"
               "cycle_index,cycle_size,direction,movement_axis,source_participant,"
               "destination_participant,"
               "source_priority,destination_priority,source_device,"
               "destination_device,source_world_rank,destination_world_rank,"
               "estimated_weight_bytes,activation_count,blocking_inference,"
               "policy_owner\n";

        const auto direction_name = [](MoEOptimizationMovementDirection direction)
            -> const char *
        {
            switch (direction)
            {
            case MoEOptimizationMovementDirection::Promotion:
                return "promotion";
            case MoEOptimizationMovementDirection::Demotion:
                return "demotion";
            case MoEOptimizationMovementDirection::SamePriority:
                return "same_priority";
            }
            return "invalid";
        };
        const auto axis_name = [](MoEOptimizationMovementAxis axis)
            -> const char *
        {
            switch (axis)
            {
            case MoEOptimizationMovementAxis::TierResidency:
                return "tier_residency";
            case MoEOptimizationMovementAxis::ParticipantPlacement:
                return "participant_placement";
            case MoEOptimizationMovementAxis::Combined:
                return "combined";
            }
            return "invalid";
        };
        size_t edge_count = 0;
        for (const auto &edge : ledger.edges)
        {
            EXPECT_TRUE(edge.valid());
            if (!edge.valid())
                continue;
            ++edge_count;
            const bool host =
                edge.authority == MoEOptimizationAuthority::Host;
            output
                << (host ? "moe_overlay_residency"
                         : "moe_overlay_controller")
                << ','
                << (host ? "expert_migration_edges"
                         : "dynamic_migration_edges")
                << ",1,1," << edge.transaction << ','
                << edge.candidate_epoch << ',' << edge.layer << ','
                << edge.expert << ',' << edge.cycle_index << ','
                << edge.cycle_size << ',' << direction_name(edge.direction)
                << ',' << axis_name(edge.axis) << ','
                << edge.source_participant << ','
                << edge.destination_participant << ','
                << edge.source_priority << ',' << edge.destination_priority
                << ',' << edge.source_device.toString() << ','
                << edge.destination_device.toString() << ',';
            if (edge.source_world_rank_known)
                output << edge.source_world_rank;
            else
                output << "unknown";
            output << ',';
            if (edge.destination_world_rank_known)
                output << edge.destination_world_rank;
            else
                output << "unknown";
            output << ',' << edge.estimated_weight_bytes << ','
                   << edge.activation_count << ','
                   << (edge.blocking_inference ? "true" : "false") << ','
                   << (host ? "host" : "device") << '\n';
        }
        output.flush();
        EXPECT_TRUE(output.good()) << path;
        EXPECT_GT(edge_count, 0u)
            << "Dynamic parity produced no authenticated movement-ledger row";
    }

    /**
     * @brief Preserve the complete process-local residency decision trail.
     *
     * The compact `expert_movement.csv` contains committed edges only. This
     * companion artifact retains proposal, capacity, economy, and physical
     * publication records, including failures. PerfStats is process-local, so
     * followers use rank-qualified names while the artifact authority retains
     * the canonical filename. Export occurs after the production worker loop
     * has closed and cannot affect placement or inference ordering.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::writeResidencyDiagnosticsCsv() const noexcept -> void
    {
        if (!isDynamicResidencyProductionTest())
            return;

        try
        {
            const int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            const auto path = ensureResultsDir() /
                (isRootParityRank()
                     ? "expert_residency_diagnostics.csv"
                     : "expert_residency_diagnostics_rank_" +
                           std::to_string(rank) + ".csv");
            if (!PerfStatsCollector::writeCsv(
                    path.string(),
                    {"moe_overlay_residency", "moe_overlay_controller"}))
            {
                LOG_ERROR(
                    "[Qwen3.5 MoE GraphNative] Could not write residency diagnostics to "
                    << path);
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Residency diagnostics raised: "
                << error.what());
        }
        catch (...)
        {
            LOG_ERROR(
                "[Qwen3.5 MoE GraphNative] Residency diagnostics raised a non-standard exception");
        }
    }

    /**
     * @brief Resolve one available routed layer without guessing its namespace.
     *
     * Main-model snapshots are named by their real transformer layer. MTP
     * snapshots instead inherit an exact context-qualified graph namespace
     * from the comparison that just consumed them. A synthetic CSV layer is
     * deliberately never converted into a production key here.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::routedExpertSnapshotNamespace(
        const RoutedExpertCheckpointContext &context,
        int layer) const -> std::optional<RoutedExpertSnapshotNamespace>
    {
        if (!context.valid())
        {
            ADD_FAILURE()
                << "Routed-expert observer received an invalid checkpoint context";
            return std::nullopt;
        }
        if (context.mtp.has_value())
        {
            if (layer != context.mtp->model_layer)
                return std::nullopt;
            return RoutedExpertSnapshotNamespace{
                .production_stage_prefix =
                    context.mtp->production_stage_prefix,
                .reference_stage_prefix =
                    context.mtp->reference_stage_prefix,
            };
        }
        if (layer < 0 || layer >= parityLayerCount())
            return std::nullopt;

        const std::string layer_prefix =
            "layer" + std::to_string(layer) + '_';
        return RoutedExpertSnapshotNamespace{
            .production_stage_prefix = layer_prefix,
            .reference_stage_prefix =
                context.phase == ParityForwardPhase::Prefill
                    ? layer_prefix
                    : "decode_step" + std::to_string(context.step) + '_' +
                          layer_prefix,
        };
    }

    /**
     * @brief Retain exact promoted-expert use from one compared checkpoint.
     *
     * Histogram movement is trained by the complete authenticated request, so
     * a profitable promotion may be hot only during decode or the recursive
     * predictor. The typed context admits only layers actually compared at
     * this boundary and supplies the exact graph/reference namespace.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::observeComparedRoutedExpertCheckpoint(
        const RoutedExpertCheckpointContext &context,
        const std::vector<LayerStats> &layers) -> void
    {
        ASSERT_TRUE(context.valid());
        const ParityForwardPhase phase = context.phase;
        const int step = context.step;
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;
        const auto *const concrete =
            dynamic_cast<const OrchestrationRunner *>(orch_runner_.get());
        const auto residency = concrete
                                   ? concrete
                                         ->expertOverlayResidencySnapshotForDiagnostics()
                                   : nullptr;
        ASSERT_NE(residency, nullptr);
        ASSERT_TRUE(residency->valid());
        const auto reference_lineage_by_layer =
            routedExpertReferenceLineageByLayer(layers);

        /*
         * A promoted-expert witness proves the newly published destination,
         * but it cannot reveal a missing contribution from another tier. Keep
         * one route-conditioned record for every expert selected at each moved
         * layer. This is especially important for an asynchronous sparse
         * return: a correct local promotion can otherwise mask a late remote
         * contribution in the summed MOE_EXPERT_OUTPUT checkpoint.
         */
        std::set<int> moved_layers;
        for (const auto &promotion : promoted_experts_)
            moved_layers.insert(promotion.layer);
        for (const int layer : moved_layers)
        {
            const bool already_observed = std::any_of(
                routed_expert_contribution_witnesses_.begin(),
                routed_expert_contribution_witnesses_.end(),
                [&](const RoutedExpertContributionWitness &witness)
                {
                    return witness.phase == phase &&
                           witness.step == step &&
                           witness.layer == layer;
                });
            if (already_observed)
                continue;

            const auto layer_stats = std::find_if(
                layers.begin(),
                layers.end(),
                [&](const LayerStats &stats)
                { return stats.layer_idx == layer; });
            if (layer_stats == layers.end())
            {
                // This compared graph bank did not execute the moved layer.
                continue;
            }
            const auto checkpoint_namespace =
                routedExpertSnapshotNamespace(context, layer);
            ASSERT_TRUE(checkpoint_namespace.has_value());

            size_t route_elements = 0u;
            const std::string route_key = checkpoint_namespace->productionKey(
                "MOE_ROUTING_INDICES");
            const float *const routes =
                activeSnapshot(route_key, route_elements);
            ASSERT_NE(routes, nullptr) << route_key;

            const auto placement = std::find_if(
                residency->placement_plan->placements.begin(),
                residency->placement_plan->placements.end(),
                [&](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == layer; });
            ASSERT_NE(
                placement,
                residency->placement_plan->placements.end());
            const auto route_evidence = pinnedDeviceRouteEvidence(
                checkpoint_namespace->production_stage_prefix,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value());

            const auto expert_output = std::find_if(
                layer_stats->stage_results.begin(),
                layer_stats->stage_results.end(),
                [](const StageComparisonResult &result)
                {
                    return extractStageType(result.stage_name) ==
                           "MOE_EXPERT_OUTPUT";
                });
            ASSERT_NE(expert_output, layer_stats->stage_results.end())
                << "Moved layer " << layer
                << " omitted its post-return MOE_EXPERT_OUTPUT checkpoint";
            const auto lineage_it = reference_lineage_by_layer.find(layer);
            ASSERT_NE(lineage_it, reference_lineage_by_layer.end())
                << "Moved layer " << layer
                << " has no routed-expert reference lineage";
            const auto reference_lineage = lineage_it->second;

            const auto moe = getMoEConfig();
            ASSERT_GT(moe.top_k, 0);
            ASSERT_EQ(
                route_elements % static_cast<size_t>(moe.top_k),
                0u);
            const std::vector<float> reference_routes =
                loadPyTorchSnapshot(checkpoint_namespace->referenceKey(
                    "MOE_ROUTING_INDICES"));
            ASSERT_FALSE(reference_routes.empty())
                << checkpoint_namespace->referenceKey(
                       "MOE_ROUTING_INDICES");

            size_t contribution_elements = 0u;
            const std::string contribution_key =
                checkpoint_namespace->productionKey(
                    "MOE_ROUTE_CONTRIBUTIONS");
            const float *const contributions =
                activeSnapshot(contribution_key, contribution_elements);
            ASSERT_NE(contributions, nullptr) << contribution_key;
            const std::vector<float> reference_contributions =
                loadPyTorchSnapshot(checkpoint_namespace->referenceKey(
                    "MOE_ROUTE_CONTRIBUTIONS"));
            ASSERT_FALSE(reference_contributions.empty())
                << checkpoint_namespace->referenceKey(
                       "MOE_ROUTE_CONTRIBUTIONS");

            std::map<int, int> routed_expert_participants;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                const float raw_expert = routes[index];
                const float raw_participant =
                    route_evidence->domain_participants[index];
                ASSERT_TRUE(std::isfinite(raw_expert));
                ASSERT_TRUE(std::isfinite(raw_participant));
                const int expert = static_cast<int>(raw_expert);
                const int participant = static_cast<int>(raw_participant);
                ASSERT_EQ(raw_expert, static_cast<float>(expert));
                ASSERT_EQ(raw_participant, static_cast<float>(participant));
                const auto [it, inserted] =
                    routed_expert_participants.emplace(
                        expert, participant);
                ASSERT_TRUE(inserted || it->second == participant)
                    << "Expert " << expert
                    << " changed destination within one immutable route bank";
            }

            for (const auto &[expert, participant] :
                 routed_expert_participants)
            {
                const auto comparison = compareRoutedExpertContribution(
                    std::span<const float>(routes, route_elements),
                    reference_routes,
                    std::span<const float>(
                        route_evidence->domain_participants,
                        route_evidence->route_count),
                    expert,
                    participant,
                    static_cast<size_t>(moe.top_k),
                    std::span<const float>(
                        contributions,
                        contribution_elements),
                    reference_contributions);
                ASSERT_TRUE(comparison.validGeometry())
                    << "Route contribution geometry failed for layer "
                    << layer << " expert " << expert;
                /*
                 * The continuation endpoint always publishes its own route
                 * slots directly. External endpoints may either materialize
                 * canonical slots before the ordered fold or return a dense
                 * aggregate which is merged into MOE_EXPERT_OUTPUT. A zero
                 * raw external slot is therefore not missing execution by
                 * itself; it is valid only when the post-return checkpoint
                 * proves that the completed sparse transaction is present.
                 */
                const bool deferred_remote_aggregate =
                    participant < 0 &&
                    comparison.production_executed_rows == 0u;
                const auto publication =
                    participant >= 0
                        ? RoutedExpertContributionPublication::
                              ContinuationCanonical
                        : (deferred_remote_aggregate
                               ? RoutedExpertContributionPublication::
                                     DeferredRemoteAggregate
                               : RoutedExpertContributionPublication::
                                     ReturnedCanonical);
                const auto numerical_proof =
                    classifyPublishedRoutedExpertContribution(
                        publication,
                        comparison,
                        config_.cosine_threshold,
                        expert_output->cosine_similarity,
                        reference_lineage);
                const bool numerically_comparable =
                    comparison.comparable_rows > 0u;
                const bool evidence_passed =
                    routedExpertContributionProofPasses(numerical_proof);
                const auto disposition =
                    routedExpertContributionProofDisposition(
                        numerical_proof);
                routed_expert_contribution_witnesses_.push_back({
                    .phase = phase,
                    .step = step,
                    .layer = layer,
                    .expert = expert,
                    .domain_participant = participant,
                    .selected_placement_bank =
                        route_evidence->selected_bank,
                    .comparison = comparison,
                    .publication = publication,
                    .numerical_proof = numerical_proof,
                    .reference_lineage = reference_lineage,
                    .disposition = disposition,
                    .post_return_expert_output_cosine =
                        expert_output->cosine_similarity,
                    .numerically_comparable =
                        numerically_comparable,
                    .evidence_passed = evidence_passed,
                });
            }
        }

        for (const auto &promotion : promoted_experts_)
        {
            const auto layer_it = std::find_if(
                layers.begin(),
                layers.end(),
                [&](const LayerStats &stats)
                { return stats.layer_idx == promotion.layer; });
            if (layer_it == layers.end())
            {
                // This graph bank cannot witness a promotion in another layer.
                continue;
            }
            const auto checkpoint_namespace =
                routedExpertSnapshotNamespace(context, promotion.layer);
            ASSERT_TRUE(checkpoint_namespace.has_value());

            size_t route_elements = 0;
            const std::string snapshot_key =
                checkpoint_namespace->productionKey(
                    "MOE_ROUTING_INDICES");
            const float *const routes =
                activeSnapshot(snapshot_key, route_elements);
            if (!routes)
                continue;
            const auto placement = std::find_if(
                residency->placement_plan->placements.begin(),
                residency->placement_plan->placements.end(),
                [&](const RoutedExpertLayerPlacement &entry)
                { return entry.layer == promotion.layer; });
            ASSERT_NE(
                placement,
                residency->placement_plan->placements.end());
            const auto route_evidence = pinnedDeviceRouteEvidence(
                checkpoint_namespace->production_stage_prefix,
                route_elements,
                placement->routed_expert_tier.size());
            ASSERT_TRUE(route_evidence.has_value());
            ASSERT_GE(promotion.expert, 0);
            ASSERT_LT(
                static_cast<size_t>(promotion.expert),
                route_evidence->expert_count);
            const float placed_participant =
                route_evidence->overlay_participants[promotion.expert];
            if (placed_participant != static_cast<float>(
                                          promotion
                                              .destination_participant))
            {
                // A later committed epoch may have moved this expert again.
                continue;
            }
            const auto *const destination =
                residency->owner_map.participantForId(
                    promotion.destination_participant);
            ASSERT_NE(destination, nullptr);
            const int expected_domain_participant =
                destination->domain_name == overlay_plan_->continuation_domain
                    ? destination->domain_participant_index
                    : -1;

            // Both IDs are stored exactly as FP32 integers in parity dumps.
            bool routed_to_destination = false;
            for (size_t index = 0u; index < route_elements; ++index)
            {
                if (routes[index] !=
                    static_cast<float>(promotion.expert))
                {
                    continue;
                }
                if (route_evidence->domain_participants[index] ==
                    static_cast<float>(expected_domain_participant))
                {
                    routed_to_destination = true;
                    break;
                }
            }
            if (!routed_to_destination)
                continue;

            /*
             * MOE_EXPERT_OUTPUT proves the complete routed sum. The canonical
             * per-route tensors below isolate the moved expert itself, so an
             * unrelated low-weight top-k difference cannot make this physical
             * movement witness spuriously incomparable.
             */
            const auto expert_output_it = std::find_if(
                layer_it->stage_results.begin(),
                layer_it->stage_results.end(),
                [](const StageComparisonResult &result)
                {
                    return extractStageType(result.stage_name) ==
                           "MOE_EXPERT_OUTPUT";
                });
            ASSERT_NE(expert_output_it, layer_it->stage_results.end())
                << "Promoted-expert layer " << promotion.layer
                << " omitted the MOE_EXPERT_OUTPUT checkpoint";
            const auto lineage_it =
                reference_lineage_by_layer.find(promotion.layer);
            ASSERT_NE(lineage_it, reference_lineage_by_layer.end())
                << "Promoted-expert layer " << promotion.layer
                << " has no routed-expert reference lineage";
            const auto reference_lineage = lineage_it->second;

            const auto moe = getMoEConfig();
            ASSERT_GT(moe.top_k, 0);
            ASSERT_EQ(
                route_elements % static_cast<size_t>(moe.top_k),
                0u);
            const std::vector<float> reference_routes =
                loadPyTorchSnapshot(checkpoint_namespace->referenceKey(
                    "MOE_ROUTING_INDICES"));
            ASSERT_FALSE(reference_routes.empty())
                << "Promoted-expert witness has no Hugging Face routes for "
                << checkpoint_namespace->referenceKey(
                       "MOE_ROUTING_INDICES");

            size_t contribution_elements = 0u;
            const std::string contribution_key =
                checkpoint_namespace->productionKey(
                    "MOE_ROUTE_CONTRIBUTIONS");
            const float *const contributions =
                activeSnapshot(contribution_key, contribution_elements);
            ASSERT_NE(contributions, nullptr)
                << "The production sparse collective did not retain canonical "
                   "per-route execution evidence for "
                << contribution_key;
            const std::vector<float> reference_contributions =
                loadPyTorchSnapshot(checkpoint_namespace->referenceKey(
                    "MOE_ROUTE_CONTRIBUTIONS"));
            ASSERT_FALSE(reference_contributions.empty())
                << "Promoted-expert witness has no Hugging Face per-route "
                   "contribution for "
                << checkpoint_namespace->referenceKey(
                       "MOE_ROUTE_CONTRIBUTIONS");

            const auto comparison =
                compareRoutedExpertContribution(
                    std::span<const float>(routes, route_elements),
                    reference_routes,
                    std::span<const float>(
                        route_evidence->domain_participants,
                        route_evidence->route_count),
                    promotion.expert,
                    expected_domain_participant,
                    static_cast<size_t>(moe.top_k),
                    std::span<const float>(
                        contributions,
                        contribution_elements),
                    reference_contributions);
            ASSERT_TRUE(comparison.validGeometry())
                << "Promoted-expert row witness has incompatible route/contribution "
                   "geometry at layer "
                << promotion.layer << " during "
                << parityForwardPhaseName(phase) << " step " << step;
            ASSERT_EQ(comparison.routed_rows > 0u, routed_to_destination)
                << "Typed row comparison disagrees with the pinned destination route";

            const bool deferred_remote_aggregate =
                expected_domain_participant < 0 &&
                comparison.production_executed_rows == 0u;
            const auto publication =
                expected_domain_participant >= 0
                    ? RoutedExpertContributionPublication::
                          ContinuationCanonical
                    : (deferred_remote_aggregate
                           ? RoutedExpertContributionPublication::
                                 DeferredRemoteAggregate
                           : RoutedExpertContributionPublication::
                                 ReturnedCanonical);
            const auto numerical_proof =
                classifyPublishedRoutedExpertContribution(
                    publication,
                    comparison,
                    config_.cosine_threshold,
                    expert_output_it->cosine_similarity,
                    reference_lineage);
            const bool numerically_comparable =
                comparison.comparable_rows > 0u;
            const bool numerically_passed =
                routedExpertContributionProofPasses(numerical_proof);
            const auto disposition =
                routedExpertContributionProofDisposition(numerical_proof);

            const auto duplicate = std::find_if(
                promoted_expert_execution_witnesses_.begin(),
                promoted_expert_execution_witnesses_.end(),
                [&](const PromotedExpertExecutionWitness &witness)
                {
                    return witness.phase == phase && witness.step == step &&
                           witness.promotion == promotion;
                });
            if (duplicate == promoted_expert_execution_witnesses_.end())
            {
                promoted_expert_execution_witnesses_.push_back(
                    PromotedExpertExecutionWitness{
                        .phase = phase,
                        .step = step,
                        .promotion = promotion,
                        .selected_placement_bank =
                            route_evidence->selected_bank,
                        .routed_rows = comparison.routed_rows,
                        .comparable_route_rows =
                            comparison.comparable_rows,
                        .production_executed_route_rows =
                            comparison.production_executed_rows,
                        .reference_executed_route_rows =
                            comparison.reference_executed_rows,
                        .exact_zero_route_rows =
                            comparison.exact_zero_rows,
                        .one_sided_zero_route_rows =
                            comparison.one_sided_zero_rows,
                        .compared_elements =
                            comparison.compared_elements,
                        .expert_contribution_cosine =
                            comparison.cosine_similarity,
                        .production_l2_norm =
                            comparison.production_l2_norm,
                        .reference_l2_norm =
                            comparison.reference_l2_norm,
                        .absolute_l2_error =
                            comparison.absolute_l2_error,
                        .root_mean_square_error =
                            comparison.root_mean_square_error,
                        .contribution_state = comparison.state,
                        .numerical_proof = numerical_proof,
                        .reference_lineage = reference_lineage,
                        .disposition = disposition,
                        .valid_geometry = comparison.validGeometry(),
                        .finite = comparison.finite(),
                        .numerically_comparable =
                            numerically_comparable,
                        .numerically_passed = numerically_passed,
                    });
            }
            if (numerically_comparable)
            {
                if (publication == RoutedExpertContributionPublication::
                                       DeferredRemoteAggregate)
                {
                    EXPECT_EQ(comparison.production_executed_rows, 0u)
                        << "A deferred remote aggregate unexpectedly wrote a "
                           "canonical route slot";
                }
                else
                {
                    EXPECT_EQ(
                        comparison.production_executed_rows +
                            comparison.exact_zero_rows,
                        comparison.comparable_rows)
                        << "Promoted expert " << promotion.expert
                        << " at layer " << promotion.layer
                        << " omitted a canonical production route "
                           "contribution";
                    EXPECT_EQ(comparison.one_sided_zero_rows, 0u)
                        << "Promoted expert " << promotion.expert
                        << " at layer " << promotion.layer
                        << " has a one-sided canonical route contribution";
                }
                EXPECT_EQ(
                    comparison.reference_executed_rows +
                        comparison.exact_zero_rows,
                    comparison.comparable_rows)
                    << "Production produced a route contribution where "
                       "Hugging Face was exactly zero for promoted expert "
                    << promotion.expert << " at layer " << promotion.layer;
            }
        }
    }

    /**
     * @brief Assert and export a post-publication promoted-expert witness.
     *
     * Movement, route selection, and numerical comparison remain three
     * independently produced authorities.  This epilogue only joins their
     * immutable evidence; it neither chooses a route nor causes maintenance.
     */
    auto Qwen35MoENodeExpertOverlayParityTest::assertParityExecutionExercisesPromotedExpert() -> void
    {
        if (!isDynamicResidencyProductionTest() || !isRootParityRank())
            return;

        ASSERT_TRUE(
            dynamic_residency_proof_lifecycle_
                .numericalEvidenceComplete())
            << "Promoted-expert evidence was consumed before every enabled "
               "numerical checkpoint producer completed";
        ASSERT_FALSE(promoted_experts_.empty())
            << "Dynamic residency committed no promotion edge";

        const auto csv_path =
            ensureResultsDir() / "promoted_expert_execution.csv";
        std::ofstream csv(csv_path, std::ios::trunc);
        ASSERT_TRUE(csv.is_open()) << csv_path;
        csv << "phase,step,layer,expert,destination_participant,"
               "candidate_epoch,selected_placement_bank,"
               "routed_rows,comparable_route_rows,"
               "production_executed_route_rows,"
               "reference_executed_route_rows,exact_zero_route_rows,"
               "one_sided_zero_route_rows,compared_elements,"
               "moe_expert_contribution_cosine,valid_geometry,"
               "finite,numerically_comparable,numerically_passed,"
               "comparison_state,numerical_proof,production_l2_norm,"
               "reference_l2_norm,absolute_l2_error,rmse,"
               "reference_lineage,proof_disposition\n";
        for (const auto &witness : promoted_expert_execution_witnesses_)
        {
            csv << parityForwardPhaseName(witness.phase) << ','
                << witness.step << ','
                << witness.promotion.layer << ','
                << witness.promotion.expert << ','
                << witness.promotion.destination_participant << ','
                << witness.promotion.candidate_epoch << ','
                << witness.selected_placement_bank << ','
                << witness.routed_rows << ','
                << witness.comparable_route_rows << ','
                << witness.production_executed_route_rows << ','
                << witness.reference_executed_route_rows << ','
                << witness.exact_zero_route_rows << ','
                << witness.one_sided_zero_route_rows << ','
                << witness.compared_elements << ','
                << witness.expert_contribution_cosine << ','
                << (witness.valid_geometry ? "true" : "false") << ','
                << (witness.finite ? "true" : "false") << ','
                << (witness.numerically_comparable ? "true" : "false")
                << ','
                << (witness.numerically_passed ? "true" : "false") << ','
                << routedExpertContributionStateName(
                       witness.contribution_state)
                << ','
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << ',' << witness.production_l2_norm << ','
                << witness.reference_l2_norm << ','
                << witness.absolute_l2_error << ','
                << witness.root_mean_square_error << ','
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ','
                << routedExpertContributionDispositionName(
                       witness.disposition)
                << '\n';
        }
        csv.flush();
        ASSERT_TRUE(csv.good()) << csv_path;

        const auto routed_csv_path =
            ensureResultsDir() / "routed_expert_contributions.csv";
        std::ofstream routed_csv(routed_csv_path, std::ios::trunc);
        ASSERT_TRUE(routed_csv.is_open()) << routed_csv_path;
        routed_csv
            << "phase,step,layer,expert,domain_participant,"
               "selected_placement_bank,publication,routed_rows,"
               "comparable_route_rows,"
               "production_executed_route_rows,"
               "reference_executed_route_rows,exact_zero_route_rows,"
               "one_sided_zero_route_rows,compared_elements,cosine,"
               "post_return_expert_output_cosine,valid_geometry,finite,"
               "numerically_comparable,evidence_passed,comparison_state,"
               "numerical_proof,production_l2_norm,reference_l2_norm,"
               "absolute_l2_error,rmse,reference_lineage,"
               "proof_disposition\n";
        size_t comparable_routed_experts = 0u;
        for (const auto &witness :
             routed_expert_contribution_witnesses_)
        {
            const auto &comparison = witness.comparison;
            routed_csv
                << parityForwardPhaseName(witness.phase) << ','
                << witness.step << ',' << witness.layer << ','
                << witness.expert << ',' << witness.domain_participant << ','
                << witness.selected_placement_bank << ','
                << routedContributionPublicationName(witness.publication)
                << ','
                << comparison.routed_rows << ','
                << comparison.comparable_rows << ','
                << comparison.production_executed_rows << ','
                << comparison.reference_executed_rows << ','
                << comparison.exact_zero_rows << ','
                << comparison.one_sided_zero_rows << ','
                << comparison.compared_elements << ','
                << comparison.cosine_similarity << ','
                << witness.post_return_expert_output_cosine << ','
                << (comparison.validGeometry() ? "true" : "false") << ','
                << (comparison.finite() ? "true" : "false") << ','
                << (witness.numerically_comparable ? "true" : "false")
                << ','
                << (witness.evidence_passed ? "true" : "false") << ','
                << routedExpertContributionStateName(comparison.state)
                << ','
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << ',' << comparison.production_l2_norm << ','
                << comparison.reference_l2_norm << ','
                << comparison.absolute_l2_error << ','
                << comparison.root_mean_square_error << ','
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ','
                << routedExpertContributionDispositionName(
                       witness.disposition)
                << '\n';
            if (comparison.comparable_rows == 0u)
                continue;
            ++comparable_routed_experts;
            EXPECT_NE(
                witness.disposition,
                RoutedExpertContributionDisposition::Failed)
                << "Moved-layer routed contribution proof failed for layer "
                << witness.layer << " expert " << witness.expert
                << " domain participant " << witness.domain_participant
                << " publication="
                << routedContributionPublicationName(witness.publication)
                << " state="
                << routedExpertContributionStateName(comparison.state)
                << " proof="
                << routedExpertContributionProofName(
                       witness.numerical_proof)
                << " lineage="
                << routedExpertReferenceLineageName(
                       witness.reference_lineage)
                << ": cosine=" << comparison.cosine_similarity
                << " post_return_expert_output_cosine="
                << witness.post_return_expert_output_cosine
                << " one_sided_zero_rows="
                << comparison.one_sided_zero_rows
                << " production_rows="
                << comparison.production_executed_rows
                << " reference_rows="
                << comparison.reference_executed_rows;
        }
        routed_csv.flush();
        ASSERT_TRUE(routed_csv.good()) << routed_csv_path;
        ASSERT_GT(comparable_routed_experts, 0u)
            << "Moved layers exposed no same-expert route contributions";

        std::ostringstream candidates;
        for (const auto &promotion : promoted_experts_)
        {
            if (candidates.tellp() > 0)
                candidates << ", ";
            candidates << "layer" << promotion.layer << ":expert"
                       << promotion.expert << "->participant"
                       << promotion.destination_participant << "@epoch"
                       << promotion.candidate_epoch;
        }
        ASSERT_FALSE(promoted_expert_execution_witnesses_.empty())
            << "No numerically compared post-publication prefill, decode, or "
               "primary MTP checkpoint executed a promoted expert on its "
               "acquired destination; candidates: "
            << candidates.str();

        /*
         * A production router and the numerically close Hugging Face router
         * need not choose an identical eighth expert on every row. Such a row
         * proves destination execution, but no same-expert reference value
         * exists and it cannot vote on numerical correctness. Require every
         * observed destination participant to have at least one independent
         * same-expert, passing per-route witness. A comparable canonical-input
         * per-route mismatch fails. A deferred dense aggregate is inconclusive
         * when the current route set differs because it cannot isolate the
         * named addend. A later mismatch after an earlier discrete routing
         * divergence is likewise inconclusive: it cannot certify a path, but
         * comparing different inputs also cannot convict that path. This
         * certifies each physical publication path without turning the
         * stronger routed top-k parity gate into an accidental exact-routing
         * requirement.
         */
        std::vector<int> observed_destinations;
        for (const auto &witness : promoted_expert_execution_witnesses_)
        {
            if (std::find(
                    observed_destinations.begin(),
                    observed_destinations.end(),
                    witness.promotion.destination_participant) ==
                observed_destinations.end())
            {
                observed_destinations.push_back(
                    witness.promotion.destination_participant);
            }
            if (witness.numerically_comparable)
            {
                EXPECT_NE(
                    witness.disposition,
                    RoutedExpertContributionDisposition::Failed)
                    << "A canonical-lineage promoted-expert witness failed for layer "
                    << witness.promotion.layer << " expert "
                    << witness.promotion.expert << " proof="
                    << routedExpertContributionProofName(
                           witness.numerical_proof)
                    << " lineage="
                    << routedExpertReferenceLineageName(
                           witness.reference_lineage);
            }
        }
        for (const int destination_participant : observed_destinations)
        {
            const bool has_passing_comparable_witness = std::any_of(
                promoted_expert_execution_witnesses_.begin(),
                promoted_expert_execution_witnesses_.end(),
                [&](const PromotedExpertExecutionWitness &witness)
                {
                    return witness.promotion.destination_participant ==
                               destination_participant &&
                           witness.numerically_comparable &&
                           witness.numerically_passed;
                });
            EXPECT_TRUE(has_passing_comparable_witness)
                << "Promoted experts executed on destination participant "
                << destination_participant
                << " without any same-expert Hugging Face per-route "
                   "numerical witness for that physical publication path";
        }
    }

}
