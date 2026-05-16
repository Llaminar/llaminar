/**
 * @file MoESparseDispatchStage.cpp
 * @brief Implementation of graph-native sparse MoE dispatch payload stage.
 */

#include "MoESparseDispatchStage.h"

#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool validateHostStagedStage(IDeviceContext *ctx, DeviceId device, const char *stage_name)
        {
            if (!ctx)
            {
                LOG_ERROR("[" << stage_name << "] Null device context");
                return false;
            }
            if (device != DeviceId::cpu())
            {
                LOG_ERROR("[" << stage_name << "] Host-staged sparse dispatch requires CPU stage device, got "
                              << device.to_string());
                return false;
            }
            return true;
        }

        bool validateFP32Matrix(const TensorBase *tensor, int rows, int cols, const char *name)
        {
            if (!tensor)
            {
                LOG_ERROR("[MoESparseDispatchStage] Null " << name << " tensor");
                return false;
            }
            if (tensor->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[MoESparseDispatchStage] " << name << " must be FP32, got " << tensor->dtype_name());
                return false;
            }
            const auto &shape = tensor->shape();
            if (shape.size() != 2 || shape[0] < static_cast<size_t>(rows) || shape[1] != static_cast<size_t>(cols))
            {
                LOG_ERROR("[MoESparseDispatchStage] " << name << " shape mismatch: expected at least ["
                                                      << rows << ", " << cols << "], got rank " << shape.size());
                return false;
            }
            return true;
        }

        const MoEExpertTierDispatch *resolveTierDispatch(const MoESparseDispatchStage::Params &params)
        {
            if (params.tier_dispatch)
                return params.tier_dispatch;

            const MoEExpertDispatchOutput *dispatch_output = params.dispatch_output;
            if (!dispatch_output)
                return nullptr;
            if (params.tier_index < 0 || params.tier_index >= static_cast<int>(dispatch_output->tiers.size()))
            {
                LOG_ERROR("[MoESparseDispatchStage] tier_index=" << params.tier_index
                                                                 << " outside dispatch tier count "
                                                                 << dispatch_output->tiers.size());
                return nullptr;
            }
            if (dispatch_output->seq_len != params.seq_len ||
                dispatch_output->top_k != params.top_k ||
                dispatch_output->d_model != params.d_model)
            {
                LOG_ERROR("[MoESparseDispatchStage] Dispatch descriptor dimension mismatch: descriptor seq_len="
                          << dispatch_output->seq_len << " top_k=" << dispatch_output->top_k
                          << " d_model=" << dispatch_output->d_model << ", stage seq_len="
                          << params.seq_len << " top_k=" << params.top_k
                          << " d_model=" << params.d_model);
                return nullptr;
            }
            return &dispatch_output->tiers[static_cast<size_t>(params.tier_index)];
        }

        bool validateTierEntries(const MoEExpertTierDispatch &tier,
                                 int seq_len,
                                 int top_k,
                                 size_t row_capacity,
                                 size_t entry_capacity)
        {
            if (tier.token_rows.size() > row_capacity || tier.entries.size() > entry_capacity)
            {
                LOG_ERROR("[MoESparseDispatchStage] Sparse dispatch capacity exceeded: rows="
                          << tier.token_rows.size() << "/" << row_capacity
                          << " entries=" << tier.entries.size() << "/" << entry_capacity);
                return false;
            }

            for (int token_row : tier.token_rows)
            {
                if (token_row < 0 || token_row >= seq_len)
                {
                    LOG_ERROR("[MoESparseDispatchStage] token_row=" << token_row
                                                                    << " outside seq_len=" << seq_len);
                    return false;
                }
            }

            for (const auto &entry : tier.entries)
            {
                if (entry.token_row < 0 || entry.token_row >= seq_len ||
                    entry.route_slot < 0 || entry.route_slot >= top_k ||
                    entry.expert_id < 0 || !std::isfinite(entry.route_weight))
                {
                    LOG_ERROR("[MoESparseDispatchStage] Invalid routed entry token=" << entry.token_row
                                                                                     << " slot=" << entry.route_slot
                                                                                     << " expert=" << entry.expert_id
                                                                                     << " weight=" << entry.route_weight);
                    return false;
                }
            }
            return true;
        }

        bool tierHasPayload(const MoEExpertTierDispatch *tier)
        {
            return tier && (!tier->token_rows.empty() || !tier->entries.empty());
        }

        bool dispatchOutputHasPayloadForNonRoot(const MoESparseDispatchStage::Params &params)
        {
            if (!params.dispatch_output)
                return false;

            if (params.tier_index >= 0 && params.tier_index < static_cast<int>(params.dispatch_output->tiers.size()))
                return tierHasPayload(&params.dispatch_output->tiers[static_cast<size_t>(params.tier_index)]);

            return std::any_of(params.dispatch_output->tiers.begin(),
                               params.dispatch_output->tiers.end(),
                               [](const auto &tier)
                               { return tierHasPayload(&tier); });
        }
    } // namespace

    MoESparseDispatchStage::MoESparseDispatchStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.collective_context && params_.collective_context_lifetime)
            params_.collective_context = params_.collective_context_lifetime.get();
        if (!params_.workspace && params_.workspace_lifetime)
            params_.workspace = params_.workspace_lifetime.get();
        if (!params_.dispatch_output && params_.dispatch_output_lifetime)
            params_.dispatch_output = params_.dispatch_output_lifetime.get();
        if (!params_.inbound_rows && params_.inbound_rows_lifetime)
            params_.inbound_rows = params_.inbound_rows_lifetime.get();
    }

    bool MoESparseDispatchStage::execute(IDeviceContext *ctx)
    {
        if (!validateHostStagedStage(ctx, params_.device_id, "MoESparseDispatchStage"))
            return false;
        if (!params_.collective_context || !params_.workspace || !params_.inbound_rows)
        {
            LOG_ERROR("[MoESparseDispatchStage] Missing collective context, workspace, or inbound rows view");
            return false;
        }
        if (params_.seq_len <= 0 || params_.top_k <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[MoESparseDispatchStage] Invalid dimensions seq_len=" << params_.seq_len
                                                                             << " top_k=" << params_.top_k
                                                                             << " d_model=" << params_.d_model);
            return false;
        }
        MoEOverlayCollectiveKey runtime_key = params_.key;
        runtime_key.step_id = execution_count_++;

        if (runtime_key.direction != MoEOverlayCollectiveDirection::Dispatch || !runtime_key.isValid())
        {
            LOG_ERROR("[MoESparseDispatchStage] Invalid dispatch key " << runtime_key.toString());
            return false;
        }
        if (params_.source_participant < 0 || params_.target_participant < 0)
        {
            LOG_ERROR("[MoESparseDispatchStage] Invalid participant ids source=" << params_.source_participant
                                                                                 << " target=" << params_.target_participant);
            return false;
        }
        if (params_.replicated_hidden_export && params_.logical_continuation_root_participant < 0)
        {
            LOG_ERROR("[MoESparseDispatchStage] replicated_hidden_export requires logical_continuation_root_participant >= 0");
            return false;
        }

        const bool is_replicated_hidden_non_root =
            params_.replicated_hidden_export &&
            params_.source_participant != params_.logical_continuation_root_participant;
        if (is_replicated_hidden_non_root &&
            (tierHasPayload(params_.tier_dispatch) || dispatchOutputHasPayloadForNonRoot(params_)))
        {
            LOG_ERROR("[MoESparseDispatchStage] Non-root replicated-hidden continuation participant "
                      << params_.source_participant
                      << " was given non-empty sparse dispatch payload for logical root "
                      << params_.logical_continuation_root_participant);
            return false;
        }

        const MoEExpertTierDispatch *tier = is_replicated_hidden_non_root ? nullptr : resolveTierDispatch(params_);
        const bool has_routed_rows = tier && !tier->entries.empty();
        if (has_routed_rows)
        {
            if (!validateFP32Matrix(params_.hidden, params_.seq_len, params_.d_model, "hidden") ||
                !validateFP32Matrix(params_.routing_indices, params_.seq_len, params_.top_k, "routing_indices") ||
                !validateFP32Matrix(params_.routing_weights, params_.seq_len, params_.top_k, "routing_weights"))
            {
                return false;
            }
        }

        auto outbound = params_.workspace->localExpertInput(runtime_key.layer_idx, runtime_key.tier_idx);
        outbound.key = runtime_key;
        outbound.source_participant = params_.source_participant;
        outbound.target_participant = params_.target_participant;
        outbound.d_model = params_.d_model;
        outbound.top_k = params_.top_k;
        outbound.live_row_count = 0;
        outbound.live_entry_count = 0;
        if (outbound.entry_offsets_host && outbound.row_capacity > 0)
            outbound.entry_offsets_host[0] = 0;

        if (has_routed_rows && !is_replicated_hidden_non_root)
        {
            if (!validateTierEntries(*tier,
                                     params_.seq_len,
                                     params_.top_k,
                                     outbound.row_capacity,
                                     outbound.entry_capacity))
            {
                return false;
            }

            const float *hidden = params_.hidden->data();
            size_t entry_cursor = 0;
            for (size_t compact_row = 0; compact_row < tier->token_rows.size(); ++compact_row)
            {
                const int token_row = tier->token_rows[compact_row];
                outbound.row_ids_host[compact_row] = token_row;
                std::memcpy(outbound.hidden_rows_fp32 + compact_row * static_cast<size_t>(params_.d_model),
                            hidden + static_cast<size_t>(token_row) * static_cast<size_t>(params_.d_model),
                            static_cast<size_t>(params_.d_model) * sizeof(float));
                outbound.entry_offsets_host[compact_row] = static_cast<int32_t>(entry_cursor);

                for (const auto &entry : tier->entries)
                {
                    if (entry.token_row != token_row)
                        continue;
                    outbound.expert_ids_host[entry_cursor] = entry.expert_id;
                    outbound.route_weights_host[entry_cursor] = entry.route_weight;
                    ++entry_cursor;
                }
            }
            outbound.live_row_count = tier->token_rows.size();
            outbound.live_entry_count = entry_cursor;
            if (entry_cursor != tier->entries.size())
            {
                LOG_ERROR("[MoESparseDispatchStage] Tier descriptor token_rows did not cover every routed entry: packed="
                          << entry_cursor << " entries=" << tier->entries.size());
                return false;
            }
            outbound.entry_offsets_host[outbound.live_row_count] = static_cast<int32_t>(entry_cursor);
        }

        std::chrono::steady_clock::time_point t_dispatch_start;
        if (MoEExpertOverlayProfiler::isEnabled())
            t_dispatch_start = std::chrono::steady_clock::now();

        auto result = params_.collective_context->dispatch(runtime_key, outbound, params_.inbound_rows, ctx);
        if (!result.ok)
        {
            LOG_ERROR("[MoESparseDispatchStage] Dispatch collective failed: " << result.error);
            return false;
        }

        if (MoEExpertOverlayProfiler::isEnabled())
        {
            const size_t compact_bytes = compactMoEOverlayDispatchBytes(outbound);
            const size_t dense_bytes = denseMoEOverlayDispatchBytes(params_.seq_len, params_.top_k, params_.d_model);
            const size_t inbound_row_count = params_.inbound_rows ? params_.inbound_rows->live_row_count : 0;
            const double dispatch_wait_ms = std::chrono::duration<double, std::milli>(
                                                std::chrono::steady_clock::now() - t_dispatch_start)
                                                .count();
            MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
                params_.key.layer_idx,
                runtime_key.toString(),
                params_.source_participant,
                params_.target_participant,
                outbound.live_row_count,
                outbound.live_entry_count,
                inbound_row_count,
                compact_bytes,
                dense_bytes,
                dispatch_wait_ms);
        }

        return true;
    }

    bool MoESparseDispatchStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements MoESparseDispatchStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.hidden)
            reqs.addInput("hidden", params_.hidden->shape(), toBufferTensorType(params_.hidden->native_type()));
        if (params_.routing_indices)
            reqs.addInput("routing_indices", params_.routing_indices->shape(), toBufferTensorType(params_.routing_indices->native_type()));
        if (params_.routing_weights)
            reqs.addInput("routing_weights", params_.routing_weights->shape(), toBufferTensorType(params_.routing_weights->native_type()));
        return reqs;
    }

    StageDumpInfo MoESparseDispatchStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.hidden)
            info.addInput("hidden", params_.hidden, static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model));
        if (params_.routing_indices)
            info.addInput("routing_indices", params_.routing_indices, static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        if (params_.routing_weights)
            info.addInput("routing_weights", params_.routing_weights, static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.top_k));
        info.addScalarInt("source_participant", params_.source_participant);
        info.addScalarInt("target_participant", params_.target_participant);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("top_k", params_.top_k);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarInt("tier_index", params_.tier_index);
        info.addScalarBool("replicated_hidden_export", params_.replicated_hidden_export);
        info.addScalarInt("logical_continuation_root_participant", params_.logical_continuation_root_participant);
        return info;
    }

} // namespace llaminar2