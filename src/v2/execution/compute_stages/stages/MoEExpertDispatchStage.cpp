/**
 * @file MoEExpertDispatchStage.cpp
 * @brief Implementation of host-side MoE expert-parallel dispatch descriptor stage.
 */

#include "MoEExpertDispatchStage.h"

#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <sstream>
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
        const ExpertLayerPlacement &placement,
        const std::vector<ExpertRoutedTier> &tiers)
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

} // namespace

    MoEExpertDispatchStage::MoEExpertDispatchStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.output && params_.output_lifetime)
            params_.output = params_.output_lifetime.get();
    }

    bool MoEExpertDispatchStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

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

        if (!validateRoutingTensor(params_.routing_indices, params_.seq_len, params_.top_k, "routing_indices") ||
            !validateRoutingTensor(params_.routing_weights, params_.seq_len, params_.top_k, "routing_weights"))
        {
            return false;
        }

        const auto &placement = params_.placement.value();
        if (placement.routed_expert_tier.empty())
        {
            LOG_ERROR("[MoEExpertDispatchStage] Placement for layer " << placement.layer
                                                                       << " has no routed expert assignments");
            return false;
        }

        MoEExpertDispatchOutput result;
        result.seq_len = params_.seq_len;
        result.top_k = params_.top_k;
        result.d_model = params_.d_model;
        result.continuation_domain = params_.continuation_domain;
        result.tiers.reserve(params_.routed_tiers.size());
        for (size_t tier_index = 0; tier_index < params_.routed_tiers.size(); ++tier_index)
        {
            const auto &tier = params_.routed_tiers[tier_index];
            MoEExpertTierDispatch tier_dispatch;
            tier_dispatch.tier_index = static_cast<int>(tier_index);
            tier_dispatch.tier_name = tier.name;
            tier_dispatch.domain = tier.domain;
            tier_dispatch.fallback = tier.fallback;
            result.tiers.push_back(std::move(tier_dispatch));
        }

        std::vector<std::vector<unsigned char>> seen_token_rows(
            params_.routed_tiers.size(),
            std::vector<unsigned char>(static_cast<size_t>(params_.seq_len), 0));

        const float *indices = params_.routing_indices->data();
        const float *weights = params_.routing_weights->data();

        for (int token_row = 0; token_row < params_.seq_len; ++token_row)
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
                if (tier_index < 0 || tier_index >= static_cast<int>(params_.routed_tiers.size()))
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

                auto &tier_dispatch = result.tiers[static_cast<size_t>(tier_index)];
                tier_dispatch.entries.push_back(MoEExpertDispatchEntry{
                    .token_row = token_row,
                    .route_slot = route_slot,
                    .expert_id = expert_id,
                    .route_weight = route_weight,
                });

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
            const auto &tier = params_.routed_tiers[tier_index];
            tier_dispatch.transfer_required =
                has_continuation_domain &&
                (tier_dispatch.domain != params_.continuation_domain || tier.fallback);

            if (!tier_dispatch.transfer_required)
            {
                tier_dispatch.transfer_mode = MoEExpertTransferMode::None;
                tier_dispatch.transfer_volume = MoEExpertTokenRowTransfer::estimateVolume(
                    params_.seq_len,
                    params_.top_k,
                    params_.d_model,
                    tier_dispatch.token_rows.size(),
                    MoEExpertTransferMode::None);
                continue;
            }

            const auto resolved_mode = resolveTierTransferMode(
                params_.transfer_mode,
                params_.seq_len,
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
                !(params_.seq_len == 1 && tier_dispatch.token_rows.size() == 1))
            {
                LOG_ERROR("[MoEExpertDispatchStage] DecodeOneToken transfer requires exactly one selected row");
                return false;
            }

            tier_dispatch.transfer_mode = resolved_mode;
            tier_dispatch.transfer_volume = MoEExpertTokenRowTransfer::estimateVolume(
                params_.seq_len,
                params_.top_k,
                params_.d_model,
                tier_dispatch.token_rows.size(),
                resolved_mode);
        }

        traceDispatchOutput(result, placement.layer);
    dumpPlacementIfRequested(placement, params_.routed_tiers);
    MoEExpertOverlayProfiler::recordDispatch(placement.layer, result, placement, params_.routed_tiers);

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
        if (params_.routing_indices)
            reqs.addInput("routing_indices", params_.routing_indices->shape(), toBufferTensorType(params_.routing_indices->native_type()));
        if (params_.routing_weights)
            reqs.addInput("routing_weights", params_.routing_weights->shape(), toBufferTensorType(params_.routing_weights->native_type()));
        if (params_.hidden)
            reqs.addInput("hidden", params_.hidden->shape(), toBufferTensorType(params_.hidden->native_type()));
        return reqs;
    }

    StageBufferContract MoEExpertDispatchStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.routing_indices && params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (params_.routing_weights && params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        if (params_.hidden && params_.hidden_buffer_id)
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
        if (params_.placement)
        {
            info.addScalarInt("layer", params_.placement->layer);
            info.addScalarInt("placement_expert_count", static_cast<int>(params_.placement->routed_expert_tier.size()));
        }
        return info;
    }

} // namespace llaminar2
