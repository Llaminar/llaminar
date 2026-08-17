/**
 * @file MoERankBatchSparseStages.cpp
 * @brief Production graph lowering for rank-batched sparse ExpertOverlay transport.
 *
 * The root stage performs deterministic descriptor filtering into stable
 * participant workspaces before one direct MPI send.  The return stage decodes
 * one envelope and accumulates participants in the same ascending-id order used
 * by the previous scalar protocol, preserving its floating-point addition
 * order while removing repeated network transactions.
 */

#include "MoERankBatchSparseStages.h"

#include "../../../collective/ITPContext.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Validate that a rank-batch boundary is executing on a CPU context. */
        bool validateHostBoundary(
            IDeviceContext *ctx,
            DeviceId device,
            const char *stage_name)
        {
            if (!ctx || device != DeviceId::cpu())
            {
                LOG_ERROR("[" << stage_name
                               << "] rank-batch transport requires a CPU context and CPU stage device");
                return false;
            }
            return true;
        }

        /** @brief Validate one FP32 activation matrix used by a non-ticket source. */
        bool validateFP32Matrix(
            const TensorBase *tensor,
            int rows,
            int columns,
            const char *name)
        {
            if (!tensor || tensor->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[MoERankBatchDispatchStage] " << name
                                                          << " must be a non-null FP32 tensor");
                return false;
            }
            const auto &shape = tensor->shape();
            if (shape.size() != 2 ||
                shape[0] < static_cast<size_t>(rows) ||
                shape[1] != static_cast<size_t>(columns))
            {
                LOG_ERROR("[MoERankBatchDispatchStage] " << name
                                                          << " has incompatible matrix geometry");
                return false;
            }
            return true;
        }

        /** @brief Resolve one typed graph role into a complete wire namespace. */
        bool applyRuntimeSemantics(
            const IComputeStage::MoEOverlayCollectiveRuntimeParams &params,
            MoEOverlayRankBatchKey *key,
            std::string *error)
        {
            if (!key)
            {
                if (error)
                    *error = "rank-batch runtime key is null";
                return false;
            }
            using Semantics = IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics;
            switch (params.execution_semantics)
            {
            case Semantics::Decode:
            case Semantics::Prefill:
                if (key->key_namespace !=
                        MoEOverlayCollectiveNamespace::Main ||
                    params.mtp_depth != -1)
                {
                    if (error)
                        *error = "main rank-batch semantics require the Main namespace and mtp_depth=-1";
                    return false;
                }
                *key = makeMoEOverlayRankBatchKey(
                    key->generation_id,
                    key->step_id,
                    params.execution_semantics == Semantics::Prefill
                        ? ExpertHistogramSource::PrefillChunk
                        : ExpertHistogramSource::DecodeToken,
                    key->layer_idx,
                    key->tier_idx,
                    key->domain_ordinal,
                    key->source_world_rank,
                    key->target_world_rank,
                    key->direction);
                return true;
            case Semantics::MTPDraft:
                if (key->key_namespace !=
                        MoEOverlayCollectiveNamespace::MTP ||
                    params.mtp_depth < 0 ||
                    key->mtp_depth != params.mtp_depth)
                {
                    if (error)
                        *error = "MTP draft rank-batch semantics require the matching retained sidecar namespace depth";
                    return false;
                }
                *key = makeMTPMoEOverlayRankBatchKey(
                    key->generation_id,
                    key->step_id,
                    params.mtp_depth,
                    key->layer_idx,
                    key->tier_idx,
                    key->domain_ordinal,
                    key->source_world_rank,
                    key->target_world_rank,
                    key->direction);
                return true;
            case Semantics::GroupedVerifier:
                if (key->key_namespace !=
                        MoEOverlayCollectiveNamespace::Main ||
                    params.mtp_depth <= 0)
                {
                    if (error)
                        *error = "grouped-verifier rank-batch semantics require a Main graph and positive admitted draft depth";
                    return false;
                }
                *key = makeMTPMoEOverlayRankBatchKey(
                    key->generation_id,
                    key->step_id,
                    params.mtp_depth,
                    key->layer_idx,
                    key->tier_idx,
                    key->domain_ordinal,
                    key->source_world_rank,
                    key->target_world_rank,
                    key->direction);
                return true;
            case Semantics::Unspecified:
                break;
            }
            if (error)
                *error = "rank-batch sparse stage received unspecified execution semantics";
            return false;
        }

        /** @brief Stamp graph-invariant rank topology with one live request identity. */
        bool makeRuntimeKey(
            const MoEOverlayRankBatchKey &base,
            const IComputeStage::MoEOverlayCollectiveRuntimeParams &runtime,
            bool require_identity,
            bool require_semantics,
            uint64_t *local_execution_count,
            MoEOverlayRankBatchKey *result,
            std::string *error)
        {
            if (!result || !local_execution_count)
            {
                if (error)
                    *error = "rank-batch runtime key output is null";
                return false;
            }
            *result = base;
            if (require_identity)
            {
                if (!runtime.valid())
                {
                    if (error)
                        *error = "rank-batch boundary has no runner-stamped transaction identity";
                    return false;
                }
                result->generation_id = runtime.generation_id;
                result->step_id = runtime.step_id;
            }
            else
            {
                result->step_id = (*local_execution_count)++;
            }
            if (require_semantics)
            {
                if (!runtime.valid() || !runtime.hasExecutionSemantics())
                {
                    if (error)
                        *error = "rank-batch boundary has no typed execution semantics";
                    return false;
                }
                if (!applyRuntimeSemantics(runtime, result, error))
                    return false;
            }
            if (!result->isValid())
            {
                if (error)
                    *error = "runner-stamped rank-batch key is invalid: " + result->toString();
                return false;
            }
            return true;
        }

        /** @brief Ensure graph and transport retain exactly the same participant order. */
        bool participantOrderMatches(
            const std::vector<int> &stage_participants,
            const IMoEOverlayRankBatchTransport &transport)
        {
            return stage_participants == transport.participantIds();
        }

    } // namespace

    MoERankBatchDispatchStage::MoERankBatchDispatchStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.device_id != DeviceId::cpu() || !params_.transport ||
            params_.participant_ids.empty() ||
            !participantOrderMatches(
                params_.participant_ids, *params_.transport) ||
            params_.key.direction != MoEOverlayCollectiveDirection::Dispatch ||
            params_.source_participant < 0 || params_.seq_len <= 0 ||
            params_.top_k <= 0 || params_.d_model <= 0 ||
            params_.tier_index < 0)
        {
            throw std::invalid_argument(
                "rank-batch dispatch requires CPU placement, exact transport topology, and positive geometry");
        }

        const bool source = params_.endpoint_role ==
                            MoERankBatchEndpointRole::ContinuationSource;
        const int expected_rank = source
                                      ? params_.transport->sourceWorldRank()
                                      : params_.transport->targetWorldRank();
        if (params_.transport->localWorldRank() != expected_rank)
        {
            throw std::invalid_argument(
                "rank-batch dispatch endpoint role does not match the local MPI rank");
        }

        if (source)
        {
            const bool shared_rows =
                params_.transport->hasSharedRowStorage();
            if ((shared_rows &&
                 !params_.participant_workspaces.empty()) ||
                (!shared_rows &&
                 params_.participant_workspaces.size() !=
                     params_.participant_ids.size()) ||
                !params_.inbound_rows.empty() || !params_.dispatch_output ||
                (params_.ticket_storage == nullptr &&
                 (!params_.hidden || !params_.routing_indices ||
                  !params_.routing_weights)))
            {
                throw std::invalid_argument(
                    "rank-batch dispatch source has incomplete or ambiguous payload ownership");
            }
            outbound_rows_.reserve(params_.participant_ids.size());
            if (shared_rows)
            {
                /*
                 * The source packer writes directly into the POSIX mapping.
                 * Binding a second collective workspace here would silently
                 * reintroduce the bulk copy that this transport exists to
                 * remove, so shared and MPI ownership are mutually exclusive.
                 */
                for (const int participant : params_.participant_ids)
                {
                    outbound_rows_.push_back(
                        params_.transport->sharedDispatchRows(participant));
                }
            }
            else
            {
                for (const auto &workspace : params_.participant_workspaces)
                {
                    if (!workspace)
                    {
                        throw std::invalid_argument(
                            "rank-batch dispatch source has a null participant workspace");
                    }
                    outbound_rows_.push_back(workspace->localExpertInput(
                        params_.key.layer_idx, params_.tier_index));
                }
            }
            outbound_views_.reserve(outbound_rows_.size());
            for (const auto &rows : outbound_rows_)
                outbound_views_.push_back(&rows);
        }
        else
        {
            if (params_.inbound_rows.size() != params_.participant_ids.size() ||
                !params_.participant_workspaces.empty() || params_.hidden ||
                params_.routing_indices || params_.routing_weights ||
                params_.dispatch_output || params_.ticket_storage ||
                params_.hidden_buffer_id || params_.routing_indices_buffer_id ||
                params_.routing_weights_buffer_id)
            {
                throw std::invalid_argument(
                    "rank-batch dispatch target cannot own source payload bindings");
            }
            inbound_views_.reserve(params_.inbound_rows.size());
            for (const auto &rows : params_.inbound_rows)
            {
                if (!rows)
                {
                    throw std::invalid_argument(
                        "rank-batch dispatch target has a null participant receive view");
                }
                inbound_views_.push_back(rows.get());
            }
        }
    }

    const MoEExpertTierDispatch *
    MoERankBatchDispatchStage::resolveTierDispatch() const
    {
        if (!params_.dispatch_output || params_.tier_index < 0 ||
            params_.tier_index >=
                static_cast<int>(params_.dispatch_output->tiers.size()) ||
            params_.dispatch_output->seq_len != params_.seq_len ||
            params_.dispatch_output->top_k != params_.top_k ||
            params_.dispatch_output->d_model != params_.d_model)
        {
            return nullptr;
        }
        return &params_.dispatch_output->tiers[
            static_cast<size_t>(params_.tier_index)];
    }

    bool MoERankBatchDispatchStage::packParticipant(
        const MoEOverlayRankBatchKey &runtime_key,
        const MoEExpertTierDispatch &tier,
        const float *hidden,
        int logical_seq_len,
        size_t participant_index,
        uint64_t residency_epoch)
    {
        if (participant_index >= outbound_rows_.size() || !hidden ||
            logical_seq_len <= 0)
        {
            LOG_ERROR("[MoERankBatchDispatchStage] Invalid participant pack request");
            return false;
        }
        const int participant = params_.participant_ids[participant_index];
        auto &outbound = outbound_rows_[participant_index];
        outbound.key =
            runtime_key.key_namespace == MoEOverlayCollectiveNamespace::MTP
                ? makeMTPMoEOverlayCollectiveKey(
                      runtime_key.generation_id,
                      runtime_key.step_id,
                      runtime_key.mtp_depth,
                      runtime_key.layer_idx,
                      runtime_key.tier_idx,
                      participant,
                      participant,
                      MoEOverlayCollectiveDirection::Dispatch)
                : makeMoEOverlayCollectiveKey(
                      runtime_key.generation_id,
                      runtime_key.step_id,
                      runtime_key.layer_idx,
                      runtime_key.tier_idx,
                      participant,
                      participant,
                      MoEOverlayCollectiveDirection::Dispatch);
        outbound.key.histogram_source = runtime_key.histogram_source;
        outbound.residency_epoch = residency_epoch;
        outbound.source_participant = params_.source_participant;
        outbound.target_participant = participant;
        outbound.d_model = params_.d_model;
        outbound.top_k = params_.top_k;
        outbound.live_row_count = 0;
        outbound.live_entry_count = 0;
        outbound.entry_offsets_host[0] = 0;

        const size_t expected_entries = static_cast<size_t>(std::count_if(
            tier.entries.begin(),
            tier.entries.end(),
            [participant](const MoEExpertDispatchEntry &entry)
            {
                return entry.destination_participant == participant;
            }));
        if (expected_entries > outbound.entry_capacity)
        {
            LOG_ERROR("[MoERankBatchDispatchStage] Participant " << participant
                                                                  << " entries exceed fixed capacity");
            return false;
        }

        size_t entry_cursor = 0;
        size_t compact_row = 0;
        for (const int token_row : tier.token_rows)
        {
            if (token_row < 0 || token_row >= logical_seq_len)
            {
                LOG_ERROR("[MoERankBatchDispatchStage] Tier token row exceeds live sequence length");
                return false;
            }
            const size_t row_begin = entry_cursor;
            for (const auto &entry : tier.entries)
            {
                if (entry.token_row != token_row ||
                    entry.destination_participant != participant)
                {
                    continue;
                }
                if (entry.route_slot < 0 || entry.route_slot >= params_.top_k ||
                    entry.expert_id < 0 || !std::isfinite(entry.route_weight))
                {
                    LOG_ERROR("[MoERankBatchDispatchStage] Invalid targeted route descriptor entry");
                    return false;
                }
                outbound.expert_ids_host[entry_cursor] = entry.expert_id;
                outbound.route_weights_host[entry_cursor] = entry.route_weight;
                ++entry_cursor;
            }
            if (entry_cursor == row_begin)
                continue;
            if (compact_row >= outbound.row_capacity)
            {
                LOG_ERROR("[MoERankBatchDispatchStage] Participant compact rows exceed fixed capacity");
                return false;
            }
            outbound.row_ids_host[compact_row] = token_row;
            outbound.entry_offsets_host[compact_row] =
                static_cast<int32_t>(row_begin);
            std::memcpy(
                outbound.hidden_rows_fp32 +
                    compact_row * static_cast<size_t>(params_.d_model),
                hidden + static_cast<size_t>(token_row) *
                             static_cast<size_t>(params_.d_model),
                static_cast<size_t>(params_.d_model) * sizeof(float));
            ++compact_row;
        }
        if (entry_cursor != expected_entries)
        {
            LOG_ERROR("[MoERankBatchDispatchStage] Tier token rows do not cover every targeted route");
            return false;
        }
        outbound.live_row_count = compact_row;
        outbound.live_entry_count = entry_cursor;
        outbound.entry_offsets_host[compact_row] =
            static_cast<int32_t>(entry_cursor);
        return true;
    }

    void MoERankBatchDispatchStage::updateMoEOverlayCollectiveRuntimeParams(
        const MoEOverlayCollectiveRuntimeParams &params)
    {
        runtime_params_ = params;
    }

    bool MoERankBatchDispatchStage::execute(IDeviceContext *ctx)
    {
        last_result_ = {};
        if (!validateHostBoundary(
                ctx, params_.device_id, "MoERankBatchDispatchStage"))
        {
            return false;
        }

        MoEOverlayRankBatchKey runtime_key;
        std::string key_error;
        if (!makeRuntimeKey(
                params_.key,
                runtime_params_,
                params_.require_explicit_transaction_identity,
                params_.require_explicit_execution_semantics,
                &local_execution_count_,
                &runtime_key,
                &key_error))
        {
            LOG_ERROR("[MoERankBatchDispatchStage] " << key_error);
            return false;
        }

        const auto start = MoEExpertOverlayProfiler::isEnabled()
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        if (params_.endpoint_role ==
            MoERankBatchEndpointRole::ContinuationSource)
        {
            const auto *tier = resolveTierDispatch();
            if (!tier || params_.dispatch_output->residency_epoch == 0)
            {
                LOG_ERROR("[MoERankBatchDispatchStage] Source dispatch descriptor or residency epoch is invalid");
                return false;
            }
            if (std::any_of(
                    tier->entries.begin(),
                    tier->entries.end(),
                    [](const MoEExpertDispatchEntry &entry)
                    {
                        return entry.destination_participant < 0;
                    }))
            {
                LOG_ERROR("[MoERankBatchDispatchStage] Rank batching requires explicitly participant-targeted routes");
                return false;
            }

            const MoEOverlayDispatchTicket *ticket = nullptr;
            if (params_.ticket_storage)
            {
                ticket = &params_.ticket_storage->ticket();
                if (!params_.ticket_storage->hasValidBoundIdentity() ||
                    !ticket->isValid() ||
                    params_.dispatch_output->ticket_lifetime !=
                        params_.ticket_storage ||
                    ticket->header->bucket_row_capacity != params_.seq_len ||
                    ticket->header->top_k != params_.top_k ||
                    ticket->header->d_model != params_.d_model ||
                    (ticket->header->residency_epoch != 0 &&
                     ticket->header->residency_epoch !=
                         params_.dispatch_output->residency_epoch))
                {
                    LOG_ERROR(
                        "[MoERankBatchDispatchStage] Captured dispatch ticket is not certified by its preceding host dispatch descriptor");
                    return false;
                }
            }
            else if (!validateFP32Matrix(
                         params_.hidden, params_.seq_len, params_.d_model, "hidden") ||
                     !validateFP32Matrix(
                         params_.routing_indices,
                         params_.seq_len,
                         params_.top_k,
                         "routing_indices") ||
                     !validateFP32Matrix(
                         params_.routing_weights,
                         params_.seq_len,
                         params_.top_k,
                         "routing_weights"))
            {
                return false;
            }

            const float *hidden = ticket ? ticket->hidden_rows_fp32
                                         : params_.hidden->data();
            const int logical_seq_len =
                ticket ? ticket->header->logical_row_count : params_.seq_len;
            for (size_t index = 0; index < params_.participant_ids.size(); ++index)
            {
                if (!packParticipant(
                        runtime_key,
                        *tier,
                        hidden,
                        logical_seq_len,
                        index,
                        params_.dispatch_output->residency_epoch))
                {
                    return false;
                }
            }
            last_result_ = params_.transport->exchangeDispatch(
                runtime_key, outbound_views_, {});
        }
        else
        {
            last_result_ = params_.transport->exchangeDispatch(
                runtime_key, {}, inbound_views_);
        }
        if (!last_result_.ok)
        {
            LOG_ERROR("[MoERankBatchDispatchStage] " << last_result_.error);
            return false;
        }

        if (MoEExpertOverlayProfiler::isEnabled() &&
            params_.endpoint_role ==
                MoERankBatchEndpointRole::ContinuationSource)
        {
            const double batch_wait_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start)
                    .count();
            const double participant_wait_ms =
                batch_wait_ms /
                static_cast<double>(std::max<size_t>(outbound_rows_.size(), 1u));
            for (const auto &rows : outbound_rows_)
            {
                MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
                    runtime_key.layer_idx,
                    runtime_key.tier_idx,
                    runtime_key.toString(),
                    rows.source_participant,
                    rows.target_participant,
                    rows.live_row_count,
                    rows.live_entry_count,
                    /*inbound_rows=*/0,
                    compactMoEOverlayDispatchBytes(rows),
                    denseMoEOverlayDispatchBytes(
                        params_.seq_len, params_.top_k, params_.d_model),
                    participant_wait_ms);
            }
        }
        return true;
    }

    bool MoERankBatchDispatchStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements MoERankBatchDispatchStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        if (params_.endpoint_role !=
                MoERankBatchEndpointRole::ContinuationSource ||
            params_.ticket_storage)
        {
            return requirements;
        }
        if (params_.hidden)
            requirements.addInput(
                "hidden",
                params_.hidden->shape(),
                toBufferTensorType(params_.hidden->native_type()));
        if (params_.routing_indices)
            requirements.addInput(
                "routing_indices",
                params_.routing_indices->shape(),
                toBufferTensorType(params_.routing_indices->native_type()));
        if (params_.routing_weights)
            requirements.addInput(
                "routing_weights",
                params_.routing_weights->shape(),
                toBufferTensorType(params_.routing_weights->native_type()));
        return requirements;
    }

    StageBufferContract MoERankBatchDispatchStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.endpoint_role !=
                MoERankBatchEndpointRole::ContinuationSource ||
            params_.ticket_storage)
        {
            return contract;
        }
        if (params_.hidden && params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id, "FP32");
        if (params_.routing_indices && params_.routing_indices_buffer_id)
            contract.addInput(*params_.routing_indices_buffer_id, "FP32");
        if (params_.routing_weights && params_.routing_weights_buffer_id)
            contract.addInput(*params_.routing_weights_buffer_id, "FP32");
        return contract;
    }

    StageDumpInfo MoERankBatchDispatchStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.hidden)
        {
            info.addInput(
                "hidden",
                params_.hidden,
                static_cast<size_t>(params_.seq_len),
                static_cast<size_t>(params_.d_model));
        }
        info.addScalarInt("participant_count", params_.participant_ids.size());
        info.addScalarInt("source_world_rank", params_.key.source_world_rank);
        info.addScalarInt("target_world_rank", params_.key.target_world_rank);
        info.addScalarInt("tier_index", params_.tier_index);
        info.addScalarInt(
            "endpoint_role", static_cast<int>(params_.endpoint_role));
        info.addScalarBool("captured_ticket", params_.ticket_storage != nullptr);
        return info;
    }

    MoERankBatchReturnReduceStage::MoERankBatchReturnReduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (params_.device_id != DeviceId::cpu() || !params_.transport ||
            params_.participant_ids.empty() ||
            !participantOrderMatches(
                params_.participant_ids, *params_.transport) ||
            params_.key.direction !=
                MoEOverlayCollectiveDirection::ReturnReduce ||
            params_.continuation_participant < 0 || params_.seq_len <= 0 ||
            params_.d_model <= 0)
        {
            throw std::invalid_argument(
                "rank-batch return requires CPU placement, exact topology, and positive geometry");
        }

        const bool source = params_.endpoint_role ==
                            MoERankBatchEndpointRole::ContinuationSource;
        const int expected_rank = source
                                      ? params_.transport->sourceWorldRank()
                                      : params_.transport->targetWorldRank();
        if (params_.transport->localWorldRank() != expected_rank)
        {
            throw std::invalid_argument(
                "rank-batch return endpoint role does not match the local MPI rank");
        }

        if (source)
        {
            const bool has_dense = params_.dense_output != nullptr;
            const bool has_ticket = params_.ticket_storage != nullptr;
            if (params_.inbound_rows.size() != params_.participant_ids.size() ||
                !params_.outbound_rows.empty() || has_dense == has_ticket ||
                (has_ticket && params_.broadcast_after_scatter))
            {
                throw std::invalid_argument(
                    "rank-batch return source has incomplete or ambiguous output ownership");
            }
            inbound_views_.reserve(params_.inbound_rows.size());
            for (const auto &rows : params_.inbound_rows)
            {
                if (!rows)
                {
                    throw std::invalid_argument(
                        "rank-batch return source has a null receive view");
                }
                inbound_views_.push_back(rows.get());
            }
        }
        else
        {
            if (params_.outbound_rows.size() != params_.participant_ids.size() ||
                !params_.inbound_rows.empty() || params_.dense_output ||
                params_.dense_output_buffer_id || params_.ticket_storage ||
                params_.clear_output_before_scatter ||
                params_.publish_ticket_completion ||
                params_.continuation_tp_context ||
                params_.broadcast_after_scatter || params_.dispatch_output ||
                params_.release_residency_lease_on_completion)
            {
                throw std::invalid_argument(
                    "rank-batch return target cannot own continuation output state");
            }
            outbound_views_.reserve(params_.outbound_rows.size());
            for (const auto &rows : params_.outbound_rows)
            {
                if (!rows)
                {
                    throw std::invalid_argument(
                        "rank-batch return target has a null participant output view");
                }
                outbound_views_.push_back(rows.get());
            }
        }
    }

    void MoERankBatchReturnReduceStage::updateMoEOverlayCollectiveRuntimeParams(
        const MoEOverlayCollectiveRuntimeParams &params)
    {
        runtime_params_ = params;
    }

    bool MoERankBatchReturnReduceStage::scatterReceivedRows()
    {
        MoEOverlayDispatchTicket *ticket = nullptr;
        if (params_.ticket_storage)
        {
            ticket = &params_.ticket_storage->ticket();
            if (!params_.ticket_storage->hasValidBoundIdentity() ||
                !ticket->isValid() ||
                ticket->header->bucket_row_capacity != params_.seq_len ||
                ticket->header->d_model != params_.d_model)
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Captured return ticket identity is invalid");
                return false;
            }
        }
        else
        {
            if (!params_.dense_output ||
                params_.dense_output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Dense continuation output must be FP32");
                return false;
            }
            const auto &shape = params_.dense_output->shape();
            if (shape.size() != 2 ||
                shape[0] < static_cast<size_t>(params_.seq_len) ||
                shape[1] != static_cast<size_t>(params_.d_model))
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Dense continuation output geometry is invalid");
                return false;
            }
        }

        const int logical_seq_len =
            ticket ? ticket->header->logical_row_count : params_.seq_len;
        float *dense = ticket ? ticket->return_rows_fp32
                              : params_.dense_output->mutable_data();
        if (params_.clear_output_before_scatter)
        {
            std::fill_n(
                dense,
                static_cast<size_t>(params_.seq_len) *
                    static_cast<size_t>(params_.d_model),
                0.0f);
        }

        const uint64_t expected_epoch =
            params_.dispatch_output ? params_.dispatch_output->residency_epoch : 0;
        if (params_.dispatch_output && expected_epoch == 0)
        {
            LOG_ERROR("[MoERankBatchReturnReduceStage] Dispatch authority has no residency epoch");
            return false;
        }

        /*
         * `inbound_views_` is constructed from the transport's ascending
         * participant list.  Never consume in device completion order: this
         * loop is the numerical-order contract retained by rank batching.
         */
        for (size_t participant_index = 0;
             participant_index < inbound_views_.size();
             ++participant_index)
        {
            const auto &rows = *inbound_views_[participant_index];
            const int expected_participant =
                params_.participant_ids[participant_index];
            if (rows.source_participant != expected_participant ||
                rows.target_participant != params_.continuation_participant ||
                rows.d_model != params_.d_model ||
                rows.live_row_count > rows.row_capacity ||
                !rows.row_ids_host || !rows.output_rows_fp32 ||
                (rows.live_row_count != 0 && rows.residency_epoch == 0) ||
                (rows.live_row_count != 0 && expected_epoch != 0 &&
                 rows.residency_epoch != expected_epoch))
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Returned participant rows violate identity, epoch, or capacity");
                return false;
            }

            for (size_t compact_row = 0;
                 compact_row < rows.live_row_count;
                 ++compact_row)
            {
                const int row_id = rows.row_ids_host[compact_row];
                if (row_id < 0 || row_id >= logical_seq_len)
                {
                    LOG_ERROR("[MoERankBatchReturnReduceStage] Returned row id exceeds live sequence length");
                    return false;
                }
                const float *source =
                    rows.output_rows_fp32 +
                    compact_row * static_cast<size_t>(params_.d_model);
                float *destination =
                    dense + static_cast<size_t>(row_id) *
                                static_cast<size_t>(params_.d_model);
                for (int column = 0; column < params_.d_model; ++column)
                    destination[column] += source[column];
            }
        }

        if (params_.broadcast_after_scatter)
        {
            if (!params_.continuation_tp_context ||
                params_.continuation_root_tp_index < 0 ||
                params_.continuation_root_tp_index >=
                    params_.continuation_tp_context->degree() ||
                !params_.continuation_tp_context->broadcast(
                    params_.dense_output,
                    params_.continuation_root_tp_index))
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Continuation broadcast failed");
                return false;
            }
        }

        if (params_.publish_ticket_completion)
        {
            if (!ticket)
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Ticket completion requested without ticket storage");
                return false;
            }
            ticket->header->return_logical_row_count = logical_seq_len;
        }

        if (params_.release_residency_lease_on_completion)
        {
            if (!params_.dispatch_output ||
                !params_.dispatch_output->residency_lease ||
                params_.dispatch_output->residency_epoch == 0)
            {
                LOG_ERROR("[MoERankBatchReturnReduceStage] Final batch cannot release a missing residency lease");
                return false;
            }
            const uint64_t epoch = params_.dispatch_output->residency_epoch;
            params_.dispatch_output->residency_lease.reset();
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dispatch_epoch_releases",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.toString(),
                {{"epoch", std::to_string(epoch)},
                 {"layer", std::to_string(params_.key.layer_idx)}});
        }
        return true;
    }

    bool MoERankBatchReturnReduceStage::execute(IDeviceContext *ctx)
    {
        last_result_ = {};
        if (!validateHostBoundary(
                ctx, params_.device_id, "MoERankBatchReturnReduceStage"))
        {
            return false;
        }

        MoEOverlayRankBatchKey runtime_key;
        std::string key_error;
        if (!makeRuntimeKey(
                params_.key,
                runtime_params_,
                params_.require_explicit_transaction_identity,
                params_.require_explicit_execution_semantics,
                &local_execution_count_,
                &runtime_key,
                &key_error))
        {
            LOG_ERROR("[MoERankBatchReturnReduceStage] " << key_error);
            return false;
        }

        const auto start = MoEExpertOverlayProfiler::isEnabled()
                               ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        if (params_.endpoint_role == MoERankBatchEndpointRole::RemoteTarget)
        {
            last_result_ = params_.transport->exchangeReturn(
                runtime_key, outbound_views_, {});
        }
        else
        {
            last_result_ = params_.transport->exchangeReturn(
                runtime_key, {}, inbound_views_);
        }
        if (!last_result_.ok)
        {
            LOG_ERROR("[MoERankBatchReturnReduceStage] " << last_result_.error);
            return false;
        }

        double wait_ms = 0.0;
        if (MoEExpertOverlayProfiler::isEnabled())
        {
            wait_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
        }

        const auto scatter_start = MoEExpertOverlayProfiler::isEnabled()
                                       ? std::chrono::steady_clock::now()
                                       : std::chrono::steady_clock::time_point{};
        if (params_.endpoint_role ==
                MoERankBatchEndpointRole::ContinuationSource &&
            !scatterReceivedRows())
        {
            return false;
        }
        const double scatter_ms = MoEExpertOverlayProfiler::isEnabled()
                                      ? std::chrono::duration<double, std::milli>(
                                            std::chrono::steady_clock::now() -
                                            scatter_start)
                                            .count()
                                      : 0.0;

        if (MoEExpertOverlayProfiler::isEnabled() &&
            params_.endpoint_role ==
                MoERankBatchEndpointRole::ContinuationSource)
        {
            const double divisor = static_cast<double>(
                std::max<size_t>(inbound_views_.size(), 1u));
            for (const auto *rows : inbound_views_)
            {
                MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
                    runtime_key.layer_idx,
                    runtime_key.tier_idx,
                    runtime_key.toString(),
                    rows->source_participant,
                    rows->target_participant,
                    rows->live_row_count,
                    rows->live_row_count,
                    compactMoEOverlayReturnBytes(*rows),
                    denseMoEOverlayReturnBytes(
                        params_.seq_len, params_.d_model),
                    wait_ms / divisor,
                    scatter_ms / divisor,
                    /*broadcast_ms=*/0.0);
            }
        }
        return true;
    }

    bool MoERankBatchReturnReduceStage::manualGraphBoundaryComplete() const
    {
        if (!last_result_.ok || !last_result_.collective_complete)
            return false;
        if (params_.endpoint_role !=
                MoERankBatchEndpointRole::ContinuationSource ||
            !params_.publish_ticket_completion || !params_.ticket_storage)
        {
            return true;
        }
        return params_.ticket_storage->ticket().returnPayloadReady();
    }

    bool MoERankBatchReturnReduceStage::supportsBackend(
        ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements
    MoERankBatchReturnReduceStage::getBufferRequirements() const
    {
        StageBufferRequirements requirements;
        if (params_.dense_output)
        {
            requirements.addOutput(
                "dense_output",
                params_.dense_output->shape(),
                toBufferTensorType(params_.dense_output->native_type()));
        }
        return requirements;
    }

    StageBufferContract MoERankBatchReturnReduceStage::bufferContract() const
    {
        auto contract = StageBufferContract::build();
        if (params_.ticket_storage || !params_.dense_output_buffer_id)
            return contract;
        if (params_.clear_output_before_scatter)
            contract.addOutput(*params_.dense_output_buffer_id);
        else
            contract.addInOut(*params_.dense_output_buffer_id);
        return contract;
    }

    StageDumpInfo MoERankBatchReturnReduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.dense_output)
        {
            info.addOutput(
                "dense_output",
                params_.dense_output,
                static_cast<size_t>(params_.seq_len),
                static_cast<size_t>(params_.d_model));
        }
        info.addScalarInt("participant_count", params_.participant_ids.size());
        info.addScalarInt("source_world_rank", params_.key.source_world_rank);
        info.addScalarInt("target_world_rank", params_.key.target_world_rank);
        info.addScalarInt(
            "endpoint_role", static_cast<int>(params_.endpoint_role));
        info.addScalarBool("clear_output_before_scatter",
                           params_.clear_output_before_scatter);
        info.addScalarBool("captured_return_ticket",
                           params_.ticket_storage != nullptr);
        info.addScalarBool("publish_ticket_completion",
                           params_.publish_ticket_completion);
        info.addScalarBool("release_residency_lease_on_completion",
                           params_.release_residency_lease_on_completion);
        return info;
    }

} // namespace llaminar2
