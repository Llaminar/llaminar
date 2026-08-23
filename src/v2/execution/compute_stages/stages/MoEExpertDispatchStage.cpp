/**
 * @file MoEExpertDispatchStage.cpp
 * @brief Host-side descriptor and routing-evidence boundary for ExpertOverlay.
 *
 * A captured continuation graph publishes one fixed-capacity ticket before
 * entering this explicitly heterogeneous boundary.  Dispatch validates the
 * real prefix, binds it to one immutable residency epoch, and constructs the
 * sparse per-tier work descriptors.  When dynamic residency is enabled, the
 * same already-host-visible route IDs are merged into the phase-specific
 * histogram; no additional D2H copy, stream wait, or hot-path allocation is
 * introduced for that evidence.
 */

#include "MoEExpertDispatchStage.h"

#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../execution/moe/MoEOverlaySparseCollective.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
namespace
{

    bool isFlatOrMatrixRoutingShape(const ITensor *tensor, int seq_len, int top_k, const char *tensor_name)
    {
        const size_t expected = static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        if (tensor->numel() < expected)
        {
            LOG_ERROR("[MoEExpertDispatchStage] " << tensor_name << " has " << tensor->numel()
                                                   << " elements, expected at least " << expected
                                                   << " for seq_len=" << seq_len
                                                   << " top_k=" << top_k);
            return false;
        }

        const auto &shape = tensor->shape();
        if (shape.size() == 1)
            return true;
        if (shape.size() == 2)
        {
            if (shape[0] >= static_cast<size_t>(seq_len) && shape[1] == static_cast<size_t>(top_k))
                return true;

            LOG_ERROR("[MoEExpertDispatchStage] " << tensor_name << " must have shape ["
                                                    << seq_len << " or larger, " << top_k
                                                    << "] or be a sufficiently large flat tensor, got ["
                                                    << shape[0] << ", " << shape[1] << "]");
            return false;
        }

        LOG_ERROR("[MoEExpertDispatchStage] " << tensor_name << " must be 1D flat or 2D, got rank " << shape.size());
        return false;
    }

    bool validateRoutingTensor(const ITensor *tensor, int seq_len, int top_k, const char *tensor_name)
    {
        if (!tensor)
        {
            LOG_ERROR("[MoEExpertDispatchStage] Null " << tensor_name << " tensor");
            return false;
        }
        if (tensor->native_type() != TensorType::FP32)
        {
            LOG_ERROR("[MoEExpertDispatchStage] " << tensor_name << " must be FP32");
            return false;
        }
        return isFlatOrMatrixRoutingShape(tensor, seq_len, top_k, tensor_name);
    }

    bool routeValueToExpertId(float value, int token_row, int route_slot, int &expert_id)
    {
        if (!std::isfinite(value))
        {
            LOG_ERROR("[MoEExpertDispatchStage] Non-finite expert id at token_row=" << token_row
                                                                                     << " route_slot=" << route_slot);
            return false;
        }

        const float rounded = std::round(value);
        if (std::fabs(value - rounded) > 1e-4f)
        {
            LOG_ERROR("[MoEExpertDispatchStage] Non-integral expert id " << value
                                                                         << " at token_row=" << token_row
                                                                         << " route_slot=" << route_slot);
            return false;
        }

        expert_id = static_cast<int>(rounded);
        return true;
    }

    MoEExpertTransferMode resolveTierTransferMode(
        MoEExpertTransferMode requested,
        int seq_len,
        size_t selected_rows)
    {
        if (requested == MoEExpertTransferMode::Auto)
        {
            if (seq_len == 1 && selected_rows == 1)
                return MoEExpertTransferMode::DecodeOneToken;
            return MoEExpertTransferMode::SparseTokenRows;
        }

        return requested;
    }

    std::string summarizeTokenRows(const std::vector<int> &token_rows)
    {
        constexpr size_t kMaxRowsToPrint = 16;
        std::ostringstream out;
        out << "[";
        const size_t printed = std::min(token_rows.size(), kMaxRowsToPrint);
        for (size_t i = 0; i < printed; ++i)
        {
            if (i > 0)
                out << ",";
            out << token_rows[i];
        }
        if (token_rows.size() > printed)
            out << ",...";
        out << "]";
        return out.str();
    }

    void traceDispatchOutput(const MoEExpertDispatchOutput &output, int layer)
    {
        const auto &env = debugEnv();
        if (!env.moe_expert_overlay.transfer_trace && !env.moe_expert_overlay.trace && !env.profile.enabled)
            return;

        for (const auto &tier : output.tiers)
        {
            LOG_DEBUG("[MoEExpertDispatchStage] layer=" << layer
                     << " tier=" << tier.tier_index
                     << " name=" << tier.tier_name
                     << " domain=" << tier.domain
                     << " selected_rows=" << tier.token_rows.size()
                     << " token_rows=" << summarizeTokenRows(tier.token_rows)
                     << " routed_entries=" << tier.entries.size()
                     << " transfer_required=" << tier.transfer_required
                     << " mode=" << toString(tier.transfer_mode)
                     << " outbound_bytes=" << tier.transfer_volume.outbound_bytes
                     << " return_bytes=" << tier.transfer_volume.return_bytes
                     << " dense_total_bytes=" << tier.transfer_volume.denseTotalBytes());
        }
    }

    void dumpPlacementIfRequested(
        const RoutedExpertLayerPlacement &placement,
        const std::vector<RoutedExpertTier> &tiers)
    {
        if (!debugEnv().moe_expert_overlay.dump_placement)
            return;

        for (size_t tier_index = 0; tier_index < tiers.size(); ++tier_index)
        {
            const auto &tier = tiers[tier_index];
            const int assigned_experts = static_cast<int>(std::count(
                placement.routed_expert_tier.begin(),
                placement.routed_expert_tier.end(),
                static_cast<int>(tier_index)));
            const int resident_experts = tier.max_experts_per_layer > 0
                                             ? tier.max_experts_per_layer
                                             : assigned_experts;
            LOG_DEBUG("[MoEExpertDispatchStage] placement layer=" << placement.layer
                     << " tier=" << tier_index
                     << " name=" << tier.name
                     << " domain=" << tier.domain
                     << " assigned_experts=" << assigned_experts
                     << " resident_experts=" << resident_experts
                     << " fallback=" << (tier.fallback ? "true" : "false"));
        }
    }

    bool sameTierTopology(
        const std::vector<RoutedExpertTier> &captured,
        const std::vector<RoutedExpertTier> &published)
    {
        if (captured.size() != published.size())
            return false;

        for (size_t index = 0; index < captured.size(); ++index)
        {
            const auto &lhs = captured[index];
            const auto &rhs = published[index];
            if (lhs.name != rhs.name ||
                lhs.domain != rhs.domain ||
                lhs.priority != rhs.priority ||
                lhs.max_experts_per_layer != rhs.max_experts_per_layer ||
                lhs.memory_budget_bytes != rhs.memory_budget_bytes ||
                lhs.resolved_live_experts_per_layer !=
                    rhs.resolved_live_experts_per_layer ||
                lhs.fallback != rhs.fallback)
            {
                return false;
            }
        }
        return true;
    }

    const RoutedExpertLayerPlacement *findLayerPlacement(
        const MoERoutedExpertPlacementPlan &plan,
        int layer_idx)
    {
        const auto found = std::find_if(
            plan.placements.begin(),
            plan.placements.end(),
            [layer_idx](const auto &placement)
            {
                return placement.layer == layer_idx;
            });
        return found == plan.placements.end() ? nullptr : &*found;
    }

} // namespace

    MoEExpertDispatchStage::MoEExpertDispatchStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.output && params_.output_lifetime)
            params_.output = params_.output_lifetime.get();

        if (params_.routing_evidence_publication)
        {
            const auto &publication =
                *params_.routing_evidence_publication;
            if (!publication.histogram)
            {
                throw std::invalid_argument(
                    "MoE ExpertOverlay routing evidence requires a histogram authority");
            }
            if (publication.source == ExpertHistogramSource::SyntheticTest ||
                publication.source == ExpertHistogramSource::GroupedVerifier)
            {
                throw std::invalid_argument(
                    "MoE ExpertOverlay dispatch may publish only committed decode or real-prefill evidence");
            }
            if (!params_.ticket_storage)
            {
                throw std::invalid_argument(
                    "MoE ExpertOverlay routing evidence requires the captured fixed-capacity ticket boundary");
            }
            if (params_.seq_len <= 0 || params_.top_k <= 0)
            {
                throw std::invalid_argument(
                    "MoE ExpertOverlay routing evidence requires positive ticket geometry");
            }
            const auto row_capacity = static_cast<std::size_t>(params_.seq_len);
            const auto top_k = static_cast<std::size_t>(params_.top_k);
            if (row_capacity >
                std::numeric_limits<std::size_t>::max() / top_k)
            {
                throw std::overflow_error(
                    "MoE ExpertOverlay routing evidence capacity overflows size_t");
            }
            routing_evidence_expert_ids_.resize(row_capacity * top_k);
        }
    }

    bool MoEExpertDispatchStage::publishRoutingEvidence(
        int logical_seq_len) const
    {
        if (!params_.routing_evidence_publication)
            return true;

        const auto &publication = *params_.routing_evidence_publication;
        const ExpertHistogramMergeResult merged =
            publication.histogram->mergeRoutedExpertRows(
                routing_evidence_expert_ids_.data(),
                RoutedExpertHistogramMerge{
                    .source = publication.source,
                    .layer_idx = params_.placement->layer,
                    .real_token_count = logical_seq_len,
                    .bucket_token_count = params_.seq_len,
                    .top_k = params_.top_k,
                    .route_stride = params_.top_k,
                    .count_window_tokens = true,
                });
        if (!merged)
        {
            LOG_ERROR(
                "[MoEExpertDispatchStage] Fixed-capacity ticket routing "
                "evidence publication failed"
                << " layer=" << params_.placement->layer
                << " logical_rows=" << logical_seq_len
                << " bucket_rows=" << params_.seq_len
                << " reason=" << merged.error);
            return false;
        }

        if (PerfStatsCollector::isDomainEnabled("moe"))
        {
            const bool decode =
                publication.source == ExpertHistogramSource::DecodeToken;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "ticket_routing_evidence_rows",
                static_cast<double>(logical_seq_len),
                decode ? "decode" : "prefill",
                params_.ticket_storage->sourceDevice().toString(),
                {{"layer", std::to_string(params_.placement->layer)},
                 {"bucket_rows", std::to_string(params_.seq_len)},
                 {"source", decode ? "decode" : "prefill"}});
        }
        return true;
    }

    bool MoEExpertDispatchStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (params_.ticket_storage)
        {
            PerfStatsCollector::ScopedTimer publication_wait(
                "moe_overlay_endpoint",
                "captured_ticket_publication_wait",
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.ticket_storage->sourceDevice().toString(),
                {{"consumer", "expert_dispatch"},
                 {"layer",
                  params_.placement.has_value()
                      ? std::to_string(params_.placement->layer)
                      : std::string("invalid")},
                 {"ordering", "mapped_system_release_acquire"},
                 {"stream_synchronize", "false"}});
            std::string publication_error;
            if (!params_.ticket_storage->awaitCapturedPublication(
                    &publication_error))
            {
                LOG_ERROR(
                    "[MoEExpertDispatchStage] Captured ticket publication "
                    "was not observable: "
                    << publication_error);
                return false;
            }
        }

        if (params_.seq_len <= 0 || params_.top_k <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[MoEExpertDispatchStage] Invalid dimensions seq_len=" << params_.seq_len
                                                                              << " top_k=" << params_.top_k
                                                                              << " d_model=" << params_.d_model);
            return false;
        }
        if (!params_.placement.has_value())
        {
            LOG_ERROR("[MoEExpertDispatchStage] Missing expert layer placement");
            return false;
        }
        if (params_.routed_tiers.empty())
        {
            LOG_ERROR("[MoEExpertDispatchStage] No routed tiers provided");
            return false;
        }
        if (!params_.output)
        {
            LOG_ERROR("[MoEExpertDispatchStage] Null dispatch output");
            return false;
        }

        const RoutedExpertLayerPlacement *effective_placement =
            &params_.placement.value();
        const std::vector<RoutedExpertTier> *effective_tiers =
            &params_.routed_tiers;
        const MoEExpertOwnerMap *effective_owner_map =
            params_.owner_map.get();
        std::shared_ptr<MoEOverlayResidencyAuthority::TicketLease>
            residency_lease;
        /*
         * Immutable static placement is the first published residency
         * snapshot even when it does not need a lease-holding authority.
         * Dynamic overlays replace this value with the authority snapshot.
         */
        uint64_t residency_epoch = 1;
        const auto *cpu_llep_state =
            params_.cpu_current_batch_llep_state.get();
        if (cpu_llep_state)
        {
            if (!cpu_llep_state->active ||
                !cpu_llep_state->durable_epoch_lease.has_value() ||
                cpu_llep_state->durable_epoch_lease->purpose() !=
                    MoEOverlayResidencyAuthority::TicketLeasePurpose::
                        CurrentBatchLLEP ||
                cpu_llep_state->durable_parent_epoch == 0 ||
                cpu_llep_state->durable_epoch_lease->epoch() !=
                    cpu_llep_state->durable_parent_epoch ||
                cpu_llep_state->layer_idx != params_.placement->layer ||
                cpu_llep_state->seq_len != params_.seq_len ||
                cpu_llep_state->top_k != params_.top_k ||
                cpu_llep_state->destination_by_flat_route.size() !=
                    static_cast<size_t>(params_.seq_len) *
                        static_cast<size_t>(params_.top_k))
            {
                LOG_ERROR(
                    "[MoEExpertDispatchStage] CPU LLEP dispatch has no complete active child publication");
                return false;
            }
            const auto &snapshot =
                **cpu_llep_state->durable_epoch_lease;
            if (!snapshot.valid() || !snapshot.placement_plan ||
                !sameTierTopology(
                    params_.routed_tiers,
                    snapshot.placement_plan->routed_tiers) ||
                (!params_.continuation_domain.empty() &&
                 snapshot.placement_plan->continuation_domain !=
                     params_.continuation_domain))
            {
                LOG_ERROR(
                    "[MoEExpertDispatchStage] CPU LLEP parent epoch changed graph-frozen topology");
                return false;
            }
            effective_placement = findLayerPlacement(
                *snapshot.placement_plan,
                params_.placement->layer);
            if (!effective_placement)
            {
                LOG_ERROR(
                    "[MoEExpertDispatchStage] CPU LLEP parent epoch has no addressed layer placement");
                return false;
            }
            effective_tiers = &snapshot.placement_plan->routed_tiers;
            effective_owner_map = &snapshot.owner_map;
            residency_epoch = snapshot.epoch;
        }
        else if (params_.residency_authority)
        {
            auto acquired =
                params_.residency_authority->tryAcquireTicketSnapshot();
            if (!acquired.has_value())
            {
                LOG_ERROR("[MoEExpertDispatchStage] Residency maintenance is active; dispatch admission is closed");
                return false;
            }
            residency_lease =
                std::make_shared<MoEOverlayResidencyAuthority::TicketLease>(
                    std::move(*acquired));
            const auto &snapshot = **residency_lease;
            if (!snapshot.valid() || !snapshot.placement_plan)
            {
                LOG_ERROR("[MoEExpertDispatchStage] Acquired invalid residency snapshot");
                return false;
            }
            if (!sameTierTopology(
                    params_.routed_tiers,
                    snapshot.placement_plan->routed_tiers))
            {
                LOG_ERROR("[MoEExpertDispatchStage] Published residency changed graph-frozen routed-tier topology");
                return false;
            }
            if (!params_.continuation_domain.empty() &&
                snapshot.placement_plan->continuation_domain !=
                    params_.continuation_domain)
            {
                LOG_ERROR("[MoEExpertDispatchStage] Published residency changed the graph-frozen continuation domain");
                return false;
            }

            effective_placement = findLayerPlacement(
                *snapshot.placement_plan,
                params_.placement->layer);
            if (!effective_placement)
            {
                LOG_ERROR("[MoEExpertDispatchStage] Published residency has no placement for layer "
                          << params_.placement->layer);
                return false;
            }
            effective_tiers = &snapshot.placement_plan->routed_tiers;
            effective_owner_map = &snapshot.owner_map;
            residency_epoch = snapshot.epoch;
        }

        const float *indices = nullptr;
        const float *weights = nullptr;
        int logical_seq_len = params_.seq_len;
        if (params_.ticket_storage)
        {
            const auto &ticket = params_.ticket_storage->ticket();
            if (!params_.ticket_storage->hasValidBoundIdentity() ||
                !ticket.isValid() ||
                ticket.header->layer_idx != effective_placement->layer ||
                ticket.header->bucket_row_capacity != params_.seq_len ||
                ticket.header->top_k != params_.top_k ||
                ticket.header->d_model != params_.d_model)
            {
                LOG_ERROR("[MoEExpertDispatchStage] Captured dispatch ticket identity or logical row count is invalid");
                return false;
            }
            logical_seq_len = ticket.header->logical_row_count;
            ticket.header->return_logical_row_count = 0;
            indices = ticket.routing_indices_fp32;
            weights = ticket.routing_weights_fp32;
        }
        else if (!validateRoutingTensor(params_.routing_indices, params_.seq_len, params_.top_k, "routing_indices") ||
                 !validateRoutingTensor(params_.routing_weights, params_.seq_len, params_.top_k, "routing_weights"))
        {
            return false;
        }
        else
        {
            indices = params_.routing_indices->data();
            weights = params_.routing_weights->data();
        }

        const auto &placement = *effective_placement;
        const auto &routed_tiers = *effective_tiers;
        if (placement.routed_expert_tier.empty())
        {
            LOG_ERROR("[MoEExpertDispatchStage] Placement for layer " << placement.layer
                                                                       << " has no routed expert assignments");
            return false;
        }

        MoEExpertDispatchOutput result;
        result.seq_len = params_.seq_len;
        result.logical_seq_len = logical_seq_len;
        result.top_k = params_.top_k;
        result.d_model = params_.d_model;
        result.continuation_domain = params_.continuation_domain;
        result.ticket_lifetime = params_.ticket_storage;
        result.residency_epoch = residency_epoch;
        result.residency_lease = std::move(residency_lease);
        result.tiers.reserve(routed_tiers.size());
        for (size_t tier_index = 0; tier_index < routed_tiers.size(); ++tier_index)
        {
            const auto &tier = routed_tiers[tier_index];
            MoEExpertTierDispatch tier_dispatch;
            tier_dispatch.tier_index = static_cast<int>(tier_index);
            tier_dispatch.tier_name = tier.name;
            tier_dispatch.domain = tier.domain;
            tier_dispatch.fallback = tier.fallback;
            result.tiers.push_back(std::move(tier_dispatch));
        }

        std::vector<std::vector<unsigned char>> seen_token_rows(
            routed_tiers.size(),
            std::vector<unsigned char>(static_cast<size_t>(logical_seq_len), 0));

        for (int token_row = 0; token_row < logical_seq_len; ++token_row)
        {
            for (int route_slot = 0; route_slot < params_.top_k; ++route_slot)
            {
                const size_t offset = static_cast<size_t>(token_row) * static_cast<size_t>(params_.top_k) +
                                      static_cast<size_t>(route_slot);

                int expert_id = -1;
                if (!routeValueToExpertId(indices[offset], token_row, route_slot, expert_id))
                    return false;

                if (expert_id < 0 || expert_id >= static_cast<int>(placement.routed_expert_tier.size()))
                {
                    LOG_ERROR("[MoEExpertDispatchStage] Expert id " << expert_id
                                                                     << " at token_row=" << token_row
                                                                     << " route_slot=" << route_slot
                                                                     << " is outside placement coverage of "
                                                                     << placement.routed_expert_tier.size()
                                                                     << " experts");
                    return false;
                }

                const int tier_index = placement.routed_expert_tier[static_cast<size_t>(expert_id)];
                if (tier_index < 0 || tier_index >= static_cast<int>(routed_tiers.size()))
                {
                    LOG_ERROR("[MoEExpertDispatchStage] Expert id " << expert_id
                                                                     << " maps to invalid tier index " << tier_index
                                                                     << " for layer " << placement.layer);
                    return false;
                }

                const float route_weight = weights[offset];
                if (!std::isfinite(route_weight))
                {
                    LOG_ERROR("[MoEExpertDispatchStage] Non-finite route weight at token_row=" << token_row
                                                                                               << " route_slot=" << route_slot);
                    return false;
                }

                int destination_participant = -1;
                if (cpu_llep_state)
                {
                    destination_participant =
                        route_weight == 0.0f
                            ? static_cast<int>(
                                  cpu_llep_state->owner_participants[
                                      static_cast<size_t>(expert_id)])
                            : static_cast<int>(
                                  cpu_llep_state->destinationForFlatRoute(
                                      offset));
                    if (destination_participant < 0 ||
                        destination_participant >=
                            cpu_llep_state->participant_count)
                    {
                        LOG_ERROR(
                            "[MoEExpertDispatchStage] CPU LLEP route has an invalid destination participant");
                        return false;
                    }
                }
                else if (effective_owner_map)
                {
                    const auto *owner = effective_owner_map->ownerFor(
                        placement.layer, expert_id);
                    if (!owner || !owner->resident ||
                        owner->tier_idx != tier_index ||
                        owner->owner_participant < 0)
                    {
                        LOG_ERROR(
                            "[MoEExpertDispatchStage] Active residency epoch has no valid physical owner for layer="
                            << placement.layer << " expert=" << expert_id
                            << " tier=" << tier_index);
                        return false;
                    }
                    destination_participant =
                        owner->owner_participant;
                }

                auto &tier_dispatch = result.tiers[static_cast<size_t>(tier_index)];
                tier_dispatch.entries.push_back(MoEExpertDispatchEntry{
                    .token_row = token_row,
                    .route_slot = route_slot,
                    .expert_id = expert_id,
                    .route_weight = route_weight,
                    .destination_participant = destination_participant,
                });

                if (params_.routing_evidence_publication)
                {
                    routing_evidence_expert_ids_[offset] = expert_id;
                }

                auto &seen = seen_token_rows[static_cast<size_t>(tier_index)][static_cast<size_t>(token_row)];
                if (!seen)
                {
                    tier_dispatch.token_rows.push_back(token_row);
                    seen = 1;
                }
            }
        }

        const bool has_continuation_domain = !params_.continuation_domain.empty();
        for (size_t tier_index = 0; tier_index < result.tiers.size(); ++tier_index)
        {
            auto &tier_dispatch = result.tiers[tier_index];
            const auto &tier = routed_tiers[tier_index];
            tier_dispatch.transfer_required =
                has_continuation_domain &&
                (tier_dispatch.domain != params_.continuation_domain || tier.fallback);

            if (!tier_dispatch.transfer_required)
            {
                tier_dispatch.transfer_mode = MoEExpertTransferMode::None;
                tier_dispatch.transfer_volume = MoEExpertTokenRowTransfer::estimateVolume(
                    logical_seq_len,
                    params_.top_k,
                    params_.d_model,
                    tier_dispatch.token_rows.size(),
                    MoEExpertTransferMode::None);
                continue;
            }

            const auto resolved_mode = resolveTierTransferMode(
                params_.transfer_mode,
                logical_seq_len,
                tier_dispatch.token_rows.size());
            if (resolved_mode == MoEExpertTransferMode::None ||
                resolved_mode == MoEExpertTransferMode::Auto)
            {
                LOG_ERROR("[MoEExpertDispatchStage] Invalid transfer mode "
                          << toString(resolved_mode) << " for routed tier "
                          << tier_dispatch.tier_name);
                return false;
            }
            if (resolved_mode == MoEExpertTransferMode::DecodeOneToken &&
                !(logical_seq_len == 1 && tier_dispatch.token_rows.size() == 1))
            {
                LOG_ERROR("[MoEExpertDispatchStage] DecodeOneToken transfer requires exactly one selected row");
                return false;
            }

            tier_dispatch.transfer_mode = resolved_mode;
            tier_dispatch.transfer_volume = MoEExpertTokenRowTransfer::estimateVolume(
                logical_seq_len,
                params_.top_k,
                params_.d_model,
                tier_dispatch.token_rows.size(),
                resolved_mode);
        }

        traceDispatchOutput(result, placement.layer);
        dumpPlacementIfRequested(placement, routed_tiers);
        MoEExpertOverlayProfiler::recordDispatch(
            placement.layer, result, placement, routed_tiers);

        /*
         * Publish only after every route in the real prefix has validated.
         * A malformed late row must not leave a partial history update behind.
         */
        if (!publishRoutingEvidence(logical_seq_len))
            return false;

        if (params_.ticket_storage)
        {
            /*
             * The present host-dispatch producer acquires the epoch here.  The
             * captured continuation-local branch will instead populate this
             * same ABI field from its device epoch slot and use the exact-epoch
             * authority overload; keeping the field explicit prevents a later
             * stage from silently consulting a newer publication.
             */
            params_.ticket_storage->ticket().header->residency_epoch =
                result.residency_epoch;
        }

        if (result.residency_lease && PerfStatsCollector::isEnabled())
        {
            const auto current_snapshot =
                params_.residency_authority->snapshot();
            const auto *first_owner = current_snapshot
                                          ? current_snapshot->owner_map.ownerFor(
                                                placement.layer,
                                                0)
                                          : nullptr;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dispatch_epoch_leases",
                1.0,
                logical_seq_len == 1 ? "decode" : "prefill",
                first_owner ? first_owner->device.toString()
                            : params_.device_id.toString(),
                {
                    {"epoch", std::to_string(result.residency_epoch)},
                    {"layer", std::to_string(placement.layer)},
                });
        }

        if (params_.ticket_storage && PerfStatsCollector::isEnabled())
        {
            const auto &ticket = params_.ticket_storage->ticket();
            const bool decode_phase =
                params_.routing_evidence_publication &&
                params_.routing_evidence_publication->source ==
                    ExpertHistogramSource::DecodeToken;
            const PerfStatsCollector::Tags tags{
                {"layer", std::to_string(placement.layer)},
                {"source_device",
                 params_.ticket_storage->sourceDevice().toString()},
                {"workspace_generation",
                 std::to_string(ticket.header->workspace_generation)},
            };
            PerfStatsCollector::addCounter(
                "moe_overlay",
                "ticket_dispatch_transactions",
                1.0,
                decode_phase ? "decode" : "prefill",
                params_.ticket_storage->sourceDevice().toString(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay",
                "ticket_dispatch_logical_rows",
                static_cast<double>(logical_seq_len),
                decode_phase ? "decode" : "prefill",
                params_.ticket_storage->sourceDevice().toString(),
                tags);
            PerfStatsCollector::addCounter(
                "moe_overlay",
                "ticket_dispatch_padding_rows",
                static_cast<double>(params_.seq_len - logical_seq_len),
                decode_phase ? "decode" : "prefill",
                params_.ticket_storage->sourceDevice().toString(),
                tags);
        }

        *params_.output = std::move(result);
        return true;
    }

    bool MoEExpertDispatchStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements MoEExpertDispatchStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (!params_.ticket_storage && params_.routing_indices)
            reqs.addInput("routing_indices", params_.routing_indices->shape(), toBufferTensorType(params_.routing_indices->native_type()));
        if (!params_.ticket_storage && params_.routing_weights)
            reqs.addInput("routing_weights", params_.routing_weights->shape(), toBufferTensorType(params_.routing_weights->native_type()));
        if (!params_.ticket_storage && params_.hidden)
            reqs.addInput("hidden", params_.hidden->shape(), toBufferTensorType(params_.hidden->native_type()));
        return reqs;
    }

    StageBufferContract MoEExpertDispatchStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (!params_.ticket_storage && params_.routing_indices && params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (!params_.ticket_storage && params_.routing_weights && params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        if (!params_.ticket_storage && params_.hidden && params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id, "FP32");
        return contract;
    }

    StageDumpInfo MoEExpertDispatchStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.routing_indices)
            info.addInput("routing_indices", params_.routing_indices,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.routing_weights)
            info.addInput("routing_weights", params_.routing_weights,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.hidden)
            info.addInput("hidden", params_.hidden,
                          static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model));

        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool("has_continuation_domain", !params_.continuation_domain.empty());
        info.addScalarInt("transfer_mode", static_cast<int>(params_.transfer_mode));
        info.addScalarInt("routed_tier_count", static_cast<int>(params_.routed_tiers.size()));
        info.addScalarBool("captured_ticket", params_.ticket_storage != nullptr);
        info.addScalarBool("live_residency_authority", params_.residency_authority != nullptr);
        if (params_.placement)
        {
            info.addScalarInt("layer", params_.placement->layer);
            info.addScalarInt("placement_expert_count", static_cast<int>(params_.placement->routed_expert_tier.size()));
        }
        return info;
    }

} // namespace llaminar2
