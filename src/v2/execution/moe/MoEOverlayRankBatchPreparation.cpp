/**
 * @file MoEOverlayRankBatchPreparation.cpp
 * @brief Shared cold first-touch and status order for original compact MoE channels.
 *
 * This is the existing rank-batch geometry and transport setup factored from
 * OrchestrationRunner. No payload semantics, native routing, communicator or
 * publication protocol is substituted for either model-wide or stage owners.
 */
#include "MoEOverlayRankBatchPreparation.h"
#include "MoEOverlayActivationChannelPlan.h"
#include "MoEOverlayNodeLocalRankBatchTransport.h"
#include "utils/PerfStatsCollector.h"
#include <limits>
#include <stdexcept>
namespace llaminar2 {
std::shared_ptr<MoEOverlayRankBatchTransportRegistry> prepareOriginalMoEOverlayRankBatchTransports(
    const MoEOverlayRankBatchPreparationInput& input)
{
    const auto& placement=input.placement;const auto& owner_map=input.owner_map;
    const auto& overlay_context=input.original_context;const auto& metadata=input;const int top_k=input.top_k;
    if(!overlay_context || overlay_context->rank()<0 || overlay_context->rank()>=input.inventory.world_size ||
        overlay_context->world_size()!=input.inventory.world_size ||
        (overlay_context->world_size()>1 && overlay_context->communicator()==MPI_COMM_NULL))
        throw std::invalid_argument("Original rank-batch setup requires its exact parent MPI membership");
    if(!MoEOverlayActivationChannelPlanner::requiresActivationChannels(placement)) return {};
    if(!input.physical_memory_authority)throw std::invalid_argument("Original activation transport requires its actual physical authority");
            const int source_participant_id =
                placement.continuation_domain_spec.logical_root_participant;
            const auto *const source =
                owner_map.participantForId(source_participant_id);
            if (source_participant_id < 0 || !source ||
                !source->world_rank_known || !source->device.is_valid() ||
                source->world_rank < 0 || source->world_rank >= overlay_context->world_size() ||
                source->tier_idx < 0 ||
                static_cast<std::size_t>(source->tier_idx) >=
                    placement.routed_tiers.size())
            {
                throw std::runtime_error(
                    "Distributed MoE ExpertOverlay activation-channel preflight "
                    "could not resolve its continuation endpoint");
            }

            const auto domain_ordinal =
                [&](const std::string &domain_name)
            {
                for (std::size_t index = 0;
                     index < placement.domains.size();
                     ++index)
                {
                    if (placement.domains[index].name == domain_name)
                        return static_cast<int>(index);
                }
                throw std::logic_error(
                    "MoE activation-channel domain is absent from the frozen placement: " +
                    domain_name);
            };
            const int source_domain_ordinal =
                domain_ordinal(source->domain_name);

            const auto& graph_family=input.graph_family;
            const int max_graph_rows=graph_family.max_graph_rows;
            if(metadata.d_model<=0 || top_k<=0 || max_graph_rows<=0 || !graph_family.valid())
                throw std::invalid_argument("Original rank-batch preparation has invalid admitted graph geometry");
            const auto activation_graph_families =
                makeMoEOverlayActivationGraphFamilyManifests(graph_family);
            const MoEOverlayActivationLaneEndpoint source_endpoint{
                .world_rank = source->world_rank,
                .participant_id = source_participant_id,
                .tier_priority =
                    placement.routed_tiers[static_cast<std::size_t>(
                        source->tier_idx)]
                        .priority,
                .domain_ordinal = source_domain_ordinal,
            };

            auto registry =
                std::make_shared<MoEOverlayRankBatchTransportRegistry>();
            if(input.activation_batch_inventory)registry->installActivationBatchInventory(input.activation_batch_inventory);
            const std::size_t row_capacity =
                static_cast<std::size_t>(max_graph_rows);
            const std::size_t top_k_size = static_cast<std::size_t>(top_k);
            if (row_capacity >
                std::numeric_limits<std::size_t>::max() / top_k_size)
            {
                throw std::overflow_error(
                    "MoE activation-channel entry capacity overflows size_t");
            }
            const std::size_t max_total_entries = row_capacity * top_k_size;
            const auto activation_channel_plan =
                MoEOverlayActivationChannelPlanner::plan({
                    .placement_plan = &placement,
                    .cluster_inventory = &input.inventory,
                    .row_capacity = row_capacity,
                    .d_model = metadata.d_model,
                    .top_k = top_k,
                    .graph_family_count =
                        activation_graph_families.size(),
                    .original_native_slots = MoEOverlayActivationChannelPlanner::originalNativeSlotGeometry(
                        placement, input.inventory, graph_family),
                });

            for (const auto &channel : activation_channel_plan.channels)
            {
                const int tier_index = channel.tier_index;
                const int target_domain_ordinal = channel.domain_ordinal;
                const int target_world_rank = channel.target_world_rank;
                const auto participant_ids = channel.targetParticipantIds();
                if (channel.source_world_rank != source->world_rank ||
                    channel.source_device != source->device ||
                    (channel.source_world_rank != target_world_rank &&
                     resolveMoEOverlayRankBatchTransportKind(
                        *overlay_context,
                        channel.source_world_rank,
                        target_world_rank) !=
                        MoEOverlayRankBatchTransportKind::
                            NodeLocalSharedRows))
                {
                    throw std::logic_error(
                        "MoE activation capacity and MPI topology authorities disagree about a node-local channel");
                }
                for (const auto &lane : channel.target_lanes)
                {
                    const auto *const participant =
                        owner_map.participantForId(lane.participant_id);
                    if (!participant || !participant->world_rank_known ||
                        participant->world_rank != target_world_rank ||
                        participant->tier_idx != tier_index ||
                        participant->device != lane.device ||
                        domain_ordinal(participant->domain_name) !=
                            target_domain_ordinal)
                    {
                        throw std::logic_error(
                            "MoE activation capacity topology diverged from the frozen owner map");
                    }
                }
                const bool same_rank = source->world_rank == target_world_rank;
                const std::string channel_identity = same_rank ?
                    makeMoEOverlayRankLocalGPUChannelIdentity(tier_index, target_domain_ordinal,
                        source->world_rank, participant_ids) :
                    makeMoEOverlayRankBatchChannelIdentity(
                        tier_index,
                        target_domain_ordinal,
                        source->world_rank,
                        target_world_rank,
                        participant_ids);

                bool local_ok = true;
                std::string local_error;
                const int local_world_rank = overlay_context->rank();
                if (local_world_rank == source->world_rank ||
                    local_world_rank == target_world_rank)
                {
                    try
                    {
                        const std::size_t row_multiplier = std::min(
                            top_k_size, participant_ids.size());
                        if (row_capacity >
                            std::numeric_limits<std::size_t>::max() /
                                row_multiplier)
                        {
                            throw std::overflow_error(
                                "MoE activation-channel aggregate row capacity overflows size_t");
                        }
                        auto workspace = std::make_shared<
                            MoEOverlayRankBatchWireWorkspace>(
                            MoEOverlayRankBatchWireWorkspace::Config{
                                .participant_ids = participant_ids,
                                .max_total_rows =
                                    row_capacity * row_multiplier,
                                .max_total_entries = max_total_entries,
                                .d_model = metadata.d_model,
                                .top_k = top_k,
                                .return_layout = channel.mapping_layout.geometry.return_layout,
                            });

                        const auto &local_lanes =
                            channel.localLanes(local_world_rank);

                        auto transport_config = MoEOverlayRankBatchTransportConfig{
                                .mpi_ctx = overlay_context,
                                .physical_memory_authority = input.physical_memory_authority,
                                .endpoint_scope = same_rank ? MoEOverlayActivationEndpointScope::RankLocalGPU :
                                    MoEOverlayActivationEndpointScope::DistinctRanks,
                                .rank_local_source_device = same_rank ? source->device : DeviceId::invalid(),
                                .source_world_rank = source->world_rank,
                                .target_world_rank = target_world_rank,
                                .workspace = std::move(workspace),
                                .max_rows_per_participant = row_capacity,
                                .max_entries_per_participant =
                                    max_total_entries,
                                .d_model = metadata.d_model,
                                .top_k = top_k,
                                .tier_index = tier_index,
                                .domain_ordinal = target_domain_ordinal,
                                .channel_identity = channel_identity,
                                .transaction_slot_count = 4096,
                                .transaction_topology =
                                    same_rank ? makeMoEOverlayRankLocalInferenceTopologyIdentity(
                                        owner_map, graph_family, source_participant_id, participant_ids) :
                                    makeMoEOverlayInferenceTopologyIdentity(
                                        owner_map,
                                        graph_family,
                                        source->world_rank,
                                        target_world_rank),
                                .source_endpoint = source_endpoint,
                                .target_tier_priority =
                                    placement.routed_tiers[
                                        static_cast<std::size_t>(tier_index)]
                                        .priority,
                                .activation_graph_families =
                                    activation_graph_families,
                                .activation_layout =
                                    channel.mapping_layout,
                                .local_lanes = local_lanes,
                                .original_cpu_joint_owners=input.original_cpu_joint_owners,
                            };
                        auto transport = same_rank ?
                            createMoEOverlayRankLocalGPUActivationTransport(std::move(transport_config)) :
                            createMoEOverlayRankBatchTransport(std::move(transport_config));
                        if (!transport ||
                            transport->kind() !=
                                MoEOverlayRankBatchTransportKind::
                                    NodeLocalSharedRows)
                        {
                            throw std::logic_error(
                                "MoE activation-channel preflight selected a non-node-local transport");
                        }
                        registry->install(
                            channel_identity, std::move(transport));
                    }
                    catch (const std::exception &error)
                    {
                        local_ok = false;
                        local_error = error.what();
                    }
                    catch (...)
                    {
                        local_ok = false;
                        local_error =
                            "non-standard exception during node-local channel setup";
                    }
                }

                /*
                 * Every rank visits every group. Non-endpoint ranks wait in
                 * this setup-only reduction while the named pair performs
                 * concurrent first-touch, preventing a later target from
                 * starting its timeout before the source reaches that group.
                 */
                int group_ok = local_ok ? 1 : 0;
                int global_group_ok = group_ok;
                if (overlay_context->world_size() > 1 && MPI_Allreduce(
                        &group_ok,
                        &global_group_ok,
                        1,
                        MPI_INT,
                        MPI_MIN,
                        overlay_context->communicator()) != MPI_SUCCESS)
                {
                    throw std::runtime_error(
                        "Distributed MoE ExpertOverlay failed to publish "
                        "node-local activation-channel setup status");
                }
                if (global_group_ok == 0)
                {
                    throw std::runtime_error(
                        "Distributed MoE ExpertOverlay could not pre-create " +
                        channel_identity +
                        (local_error.empty()
                             ? " on another rank"
                             : ": " + local_error));
                }
            }


    PerfStatsCollector::addCounter("moe_overlay_transport","node_local_activation_channels_precreated",
        static_cast<double>(registry->size()),"model_setup",{},
        {{"first_touch","consumer_rank"},{"phase","pre_graph"},{"scope","node_local"},
         {"source_rank",std::to_string(source->world_rank)}});
    return registry;
}
}
