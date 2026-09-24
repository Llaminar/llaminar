/**
 * @file MoESparseReturnReduceStage.cpp
 * @brief Implementation of graph-native sparse MoE return/reduce payload stage.
 *
 * The graph-bound return layout distinguishes dense token accumulation from
 * raw canonical-route gathering. A canonical boundary performs no weighted
 * sum: the final ordered reducer authenticates complete router-slot coverage
 * before the continuation broadcast. Transport completion and residency lease
 * retirement retain their existing explicit graph edges.
 */

#include "MoESparseReturnReduceStage.h"
#include "../../moe/MoEOverlayCanonicalHostReturn.h"

#include "../../../collective/ITPContext.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>

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
                LOG_ERROR("[" << stage_name << "] Host-staged sparse return/reduce requires CPU stage device, got "
                              << device.to_string());
                return false;
            }
            return true;
        }

        bool validateReturnRows(const MoEOverlayReturnRows &rows, int d_model)
        {
            if (rows.d_model != d_model || !rows.row_ids_host || !rows.output_rows_fp32)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Invalid return rows view");
                return false;
            }
            if (rows.live_row_count > rows.row_capacity)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Return rows live count exceeds capacity");
                return false;
            }
            if (rows.live_row_count != 0 && rows.residency_epoch == 0)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Non-empty return rows are missing their residency epoch");
                return false;
            }
            return true;
        }

        bool validateDenseOutput(TensorBase *output, int seq_len, int d_model)
        {
            if (!output)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Null dense output tensor");
                return false;
            }
            if (output->native_type() != TensorType::FP32)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] dense_output must be FP32, got " << output->dtype_name());
                return false;
            }
            const auto &shape = output->shape();
            if (shape.size() != 2 || shape[0] < static_cast<size_t>(seq_len) || shape[1] != static_cast<size_t>(d_model))
            {
                LOG_ERROR("[MoESparseReturnReduceStage] dense_output shape mismatch");
                return false;
            }
            return true;
        }
    } // namespace

    MoESparseReturnReduceStage::MoESparseReturnReduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.collective_context && params_.collective_context_lifetime)
            params_.collective_context = params_.collective_context_lifetime.get();
        if (!params_.outbound_rows && params_.outbound_rows_lifetime)
            params_.outbound_rows = params_.outbound_rows_lifetime.get();
        if (!params_.inbound_rows && params_.inbound_rows_lifetime)
            params_.inbound_rows = params_.inbound_rows_lifetime.get();
        if (!isValidMoEOverlayReturnLayout(params_.return_layout) ||
            (params_.return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes &&
             (params_.ticket_storage || params_.canonical_route_ticket_storage ||
              params_.broadcast_after_scatter || params_.publish_ticket_completion)))
            throw std::invalid_argument("Canonical host gathering cannot own a dense ticket or broadcast before the final ordered fold");
    }

    /**
     * @brief Receive the request/chunk identity selected by the overlay runner.
     *
     * The paired dispatch and return boundaries are updated before one graph
     * execution, guaranteeing that a returned row cannot be confused with a
     * prior captured graph invocation.
     */
    void MoESparseReturnReduceStage::updateMoEOverlayCollectiveRuntimeParams(
        const MoEOverlayCollectiveRuntimeParams &params)
    {
        runtime_params_ = params;
    }

    bool MoESparseReturnReduceStage::execute(IDeviceContext *ctx)
    {
        last_collective_result_ = {};
        const bool protocol_participant =
            params_.inbound_consumer_role ==
            InboundConsumerRole::ProtocolParticipant;
        const bool canonical_ticket_completion =
            params_.inbound_consumer_role ==
            InboundConsumerRole::CanonicalRouteTicketCompletion;
        const bool skips_dense_scatter =
            protocol_participant || canonical_ticket_completion;

        if (!validateHostStagedStage(ctx, params_.device_id, "MoESparseReturnReduceStage"))
            return false;
        if (!params_.collective_context || !params_.outbound_rows || !params_.inbound_rows)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Missing collective context or return row views");
            return false;
        }
        if (params_.seq_len <= 0 || params_.d_model <= 0)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Invalid dimensions seq_len=" << params_.seq_len
                                                                                 << " d_model=" << params_.d_model);
            return false;
        }
        MoEOverlayCollectiveKey runtime_key = params_.key;
        if (params_.require_explicit_transaction_identity)
        {
            if (!runtime_params_.valid())
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Distributed graph-native return "
                          "started without a runner-stamped transaction identity");
                return false;
            }
            runtime_key.generation_id = runtime_params_.generation_id;
            runtime_key.step_id = runtime_params_.step_id;
        }
        else
        {
            /* Local isolated collectives intentionally retain their own counter. */
            runtime_key.step_id = execution_count_++;
        }
        if (params_.require_explicit_execution_semantics)
        {
            if (!runtime_params_.valid() ||
                !runtime_params_.hasExecutionSemantics())
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Production sparse "
                          "return started without typed execution semantics");
                return false;
            }
            using Semantics =
                MoEOverlayCollectiveRuntimeParams::ExecutionSemantics;
            const auto semantics = runtime_params_.execution_semantics;
            if (semantics == Semantics::Decode ||
                semantics == Semantics::Prefill)
            {
                if (runtime_key.key_namespace !=
                        MoEOverlayCollectiveNamespace::Main ||
                    runtime_params_.mtp_depth != -1)
                {
                    LOG_ERROR("[MoESparseReturnReduceStage] Main execution semantics require the Main namespace and mtp_depth=-1");
                    return false;
                }
                runtime_key.histogram_source =
                    semantics == Semantics::Prefill
                        ? ExpertHistogramSource::PrefillChunk
                        : ExpertHistogramSource::DecodeToken;
            }
            else if (semantics == Semantics::MTPDraft)
            {
                if (runtime_key.key_namespace !=
                        MoEOverlayCollectiveNamespace::MTP ||
                    runtime_params_.mtp_depth < 0 ||
                    runtime_key.mtp_depth != runtime_params_.mtp_depth)
                {
                    LOG_ERROR("[MoESparseReturnReduceStage] MTP draft semantics require the matching retained sidecar namespace depth");
                    return false;
                }
                runtime_key = makeMTPMoEOverlayCollectiveKey(
                    runtime_key.generation_id,
                    runtime_key.step_id,
                    runtime_params_.mtp_depth,
                    runtime_key.layer_idx,
                    runtime_key.tier_idx,
                    runtime_key.domain_id,
                    runtime_key.participant_id,
                    runtime_key.direction);
            }
            else if (semantics == Semantics::GroupedVerifier)
            {
                if (runtime_key.key_namespace !=
                        MoEOverlayCollectiveNamespace::Main ||
                    runtime_params_.mtp_depth <= 0)
                {
                    LOG_ERROR("[MoESparseReturnReduceStage] Grouped verifier semantics require a Main graph and positive admitted draft depth");
                    return false;
                }
                runtime_key = makeMTPMoEOverlayCollectiveKey(
                    runtime_key.generation_id,
                    runtime_key.step_id,
                    runtime_params_.mtp_depth,
                    runtime_key.layer_idx,
                    runtime_key.tier_idx,
                    runtime_key.domain_id,
                    runtime_key.participant_id,
                    runtime_key.direction);
            }
            else
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Unsupported execution semantics");
                return false;
            }
        }

        if (runtime_key.direction != MoEOverlayCollectiveDirection::ReturnReduce || !runtime_key.isValid())
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Invalid return key " << runtime_key.toString());
            return false;
        }
        if (params_.source_participant < 0 || params_.target_participant < 0)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Invalid participant ids source=" << params_.source_participant
                                                                                     << " target=" << params_.target_participant);
            return false;
        }
        if (!validateReturnRows(*params_.outbound_rows, params_.d_model))
        {
            return false;
        }
        MoEOverlayDispatchTicket *ticket = nullptr;
        if (params_.ticket_storage &&
            params_.canonical_route_ticket_storage)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] One return boundary cannot own both dense and canonical tickets");
            return false;
        }
        if (protocol_participant &&
            (params_.dense_output || params_.ticket_storage ||
             params_.canonical_route_ticket_storage ||
             params_.dense_output_buffer_id ||
             params_.clear_output_before_scatter ||
             params_.broadcast_after_scatter ||
             params_.publish_ticket_completion ||
             params_.residency_lease_terminal ==
                 MoEOverlayHostDispatchLeaseTerminal::Release))
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Protocol-only participant "
                      "cannot own dense output, a captured ticket, broadcast, "
                      "or residency-lease retirement");
            return false;
        }
        if (canonical_ticket_completion)
        {
            const auto &canonical =
                params_.canonical_route_ticket_storage;
            if (!canonical || !canonical->hasValidBoundIdentity() ||
                canonical->layerIndex() != params_.key.layer_idx ||
                canonical->dModel() != params_.d_model ||
                params_.dense_output || params_.ticket_storage ||
                params_.dense_output_buffer_id ||
                params_.clear_output_before_scatter ||
                params_.broadcast_after_scatter ||
                params_.publish_ticket_completion)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Canonical-route completion requires one matching sparse ticket and no dense authority");
                return false;
            }
        }
        else if (params_.canonical_route_ticket_storage)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Canonical-route ticket storage requires its typed completion role");
            return false;
        }
        if (params_.ticket_storage)
        {
            ticket = &params_.ticket_storage->ticket();
            if (!params_.ticket_storage->hasValidBoundIdentity() ||
                !ticket->isValid() ||
                ticket->header->bucket_row_capacity != params_.seq_len ||
                ticket->header->d_model != params_.d_model ||
                params_.dense_output || params_.broadcast_after_scatter)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Invalid or ambiguous fixed-capacity return ticket contract");
                return false;
            }
        }
        else if (!skips_dense_scatter &&
                 params_.return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes)
        {
            if (!params_.dense_output ||
                params_.dense_output->native_type() != TensorType::FP32 ||
                params_.broadcast_after_scatter || params_.publish_ticket_completion ||
                canonical_moe_route_record::recordCapacity(
                    params_.dense_output->numel(), params_.d_model) == 0)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Canonical gather requires a packed bank and a separate final ordered reducer");
                return false;
            }
        }
        else if (!skips_dense_scatter && !validateDenseOutput(
                     params_.dense_output,
                     params_.seq_len,
                     params_.d_model))
        {
            return false;
        }

        MoEOverlayReturnRows outbound = *params_.outbound_rows;
        outbound.key = runtime_key;
        outbound.source_participant = params_.source_participant;
        outbound.target_participant = params_.target_participant;
        outbound.d_model = params_.d_model;

        std::chrono::steady_clock::time_point t_return_start;
        if (MoEExpertOverlayProfiler::isEnabled())
            t_return_start = std::chrono::steady_clock::now();

        last_collective_result_ = params_.collective_context->returnReduce(runtime_key, outbound, params_.inbound_rows, ctx);
        if (!last_collective_result_.ok)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Return-reduce collective failed: " << last_collective_result_.error);
            return false;
        }

        double prof_return_wait_ms = 0.0;
        if (MoEExpertOverlayProfiler::isEnabled())
            prof_return_wait_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t_return_start)
                                      .count();

        if (!validateReturnRows(*params_.inbound_rows, params_.d_model))
            return false;
        if (!skips_dense_scatter && params_.inbound_rows->layout != params_.return_layout)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Return arithmetic differs from the graph-bound consumer contract");
            return false;
        }

        /*
         * MPI ranks enter every return key, but only the rank that owns the
         * continuation participant receives rows. Treating an unexpected
         * local payload as discardable would hide a topology error, so the
         * protocol-only role is valid only when the addressed receive is empty.
         */
        if (protocol_participant && params_.inbound_rows->live_row_count != 0)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Protocol-only participant "
                      "received continuation-owned return rows");
            return false;
        }
        if (canonical_ticket_completion &&
            !params_.canonical_route_ticket_storage->publicationSucceededFor(
                params_.outbound_rows->residency_epoch))
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Colocated CPU canonical ticket was not successfully published for the returned residency epoch");
            return false;
        }

        if (params_.dispatch_output_lifetime)
        {
            const uint64_t expected_epoch =
                params_.dispatch_output_lifetime->residency_epoch;
            if (expected_epoch == 0 ||
                params_.inbound_rows->residency_epoch != expected_epoch)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Return residency epoch mismatch: expected="
                          << expected_epoch << " received="
                          << params_.inbound_rows->residency_epoch);
                return false;
            }
        }

        const int logical_seq_len = ticket
                                        ? ticket->header->logical_row_count
                                        : params_.seq_len;
        float *dense = skips_dense_scatter
                           ? nullptr
                           : (ticket
                                  ? ticket->return_rows_fp32
                                  : params_.dense_output->mutable_data());
        const size_t dense_count = skips_dense_scatter
                                       ? 0u
                                       : static_cast<size_t>(params_.seq_len) *
                                             static_cast<size_t>(params_.d_model);
        const bool canonical_gather = !skips_dense_scatter &&
            params_.return_layout == MoEOverlayReturnLayout::CanonicalExpertRoutes;
        if (canonical_gather)
        {
            if (!gatherMoEOverlayCanonicalHostReturn(
                    *params_.inbound_rows,
                    {dense, params_.dense_output->numel()},
                    params_.clear_output_before_scatter
                        ? MoEOverlayCanonicalGatherBoundary::Begin
                        : MoEOverlayCanonicalGatherBoundary::Append))
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Canonical return exceeds or violates the graph-owned packed bank");
                return false;
            }
        }
        else if (!skips_dense_scatter && params_.clear_output_before_scatter)
            std::fill_n(dense, dense_count, 0.0f);

        std::chrono::steady_clock::time_point t_scatter_start;
        if (MoEExpertOverlayProfiler::isEnabled())
            t_scatter_start = std::chrono::steady_clock::now();

        for (size_t compact_row = 0;
             !skips_dense_scatter &&
             !canonical_gather &&
             compact_row < params_.inbound_rows->live_row_count;
             ++compact_row)
        {
            const int row_id = params_.inbound_rows->row_ids_host[compact_row];
            if (row_id < 0 || row_id >= logical_seq_len)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Returned row id " << row_id
                                                                          << " outside logical_seq_len=" << logical_seq_len);
                return false;
            }
            const float *src = params_.inbound_rows->output_rows_fp32 + compact_row * static_cast<size_t>(params_.d_model);
            float *dst = dense + static_cast<size_t>(row_id) * static_cast<size_t>(params_.d_model);
            for (int col = 0; col < params_.d_model; ++col)
                dst[col] += src[col];
        }

        double prof_scatter_ms = 0.0;
        if (MoEExpertOverlayProfiler::isEnabled())
            prof_scatter_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - t_scatter_start)
                                  .count();

        double prof_broadcast_ms = 0.0;

        if (params_.broadcast_after_scatter && last_collective_result_.collective_complete)
        {
            if (!params_.continuation_tp_context)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] broadcast_after_scatter requires continuation_tp_context");
                return false;
            }
            if (params_.continuation_root_tp_index < 0 ||
                params_.continuation_root_tp_index >= params_.continuation_tp_context->degree())
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Invalid continuation_root_tp_index="
                          << params_.continuation_root_tp_index
                          << " for TP degree=" << params_.continuation_tp_context->degree());
                return false;
            }
            std::chrono::steady_clock::time_point t_broadcast_start;
            if (MoEExpertOverlayProfiler::isEnabled())
                t_broadcast_start = std::chrono::steady_clock::now();
            if (!params_.continuation_tp_context->broadcast(params_.dense_output,
                                                            params_.continuation_root_tp_index))
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Continuation TP broadcast failed from root index "
                          << params_.continuation_root_tp_index);
                return false;
            }
            if (MoEExpertOverlayProfiler::isEnabled())
            {
                prof_broadcast_ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - t_broadcast_start)
                                        .count();
            }
        }

        if (params_.publish_ticket_completion && !ticket)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Ticket completion was requested without ticket storage");
            return false;
        }
        if (ticket && params_.publish_ticket_completion &&
            last_collective_result_.collective_complete)
        {
            ticket->header->return_logical_row_count = logical_seq_len;
        }

        if (MoEExpertOverlayProfiler::isEnabled())
        {
            const size_t compact_bytes = compactMoEOverlayReturnBytes(*params_.outbound_rows);
            const size_t dense_bytes =
                skips_dense_scatter
                    ? 0u
                    : denseMoEOverlayReturnBytes(
                          params_.seq_len, params_.d_model);
            MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
                params_.key.layer_idx,
                runtime_key.tier_idx,
                MoEOverlayProfileEdge{
                    .source_participant = params_.source_participant,
                    .target_participant = params_.target_participant,
                },
                outbound.live_row_count,
                params_.inbound_rows->live_row_count,
                compact_bytes,
                dense_bytes,
                prof_return_wait_ms,
                prof_scatter_ms,
                prof_broadcast_ms);
        }

        if (params_.residency_lease_terminal ==
            MoEOverlayHostDispatchLeaseTerminal::Release)
        {
            if (!last_collective_result_.collective_complete ||
                !params_.dispatch_output_lifetime ||
                !params_.dispatch_output_lifetime->residency_lease ||
                params_.dispatch_output_lifetime->residency_epoch == 0)
            {
                LOG_ERROR(
                    "[MoESparseReturnReduceStage] Final return cannot release "
                    "a missing or incomplete residency lease"
                    << " collective_complete="
                    << last_collective_result_.collective_complete
                    << " dispatch_output="
                    << static_cast<const void *>(
                           params_.dispatch_output_lifetime.get())
                    << " lease_present="
                    << (params_.dispatch_output_lifetime &&
                        params_.dispatch_output_lifetime->residency_lease
                            ? "true"
                            : "false")
                    << " epoch="
                    << (params_.dispatch_output_lifetime
                            ? params_.dispatch_output_lifetime
                                  ->residency_epoch
                            : 0));
                return false;
            }

            const uint64_t released_epoch =
                params_.dispatch_output_lifetime->residency_epoch;
            params_.dispatch_output_lifetime->residency_lease.reset();
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "dispatch_epoch_releases",
                1.0,
                params_.seq_len == 1 ? "decode" : "prefill",
                params_.device_id.toString(),
                {
                    {"epoch", std::to_string(released_epoch)},
                    {"layer", std::to_string(params_.key.layer_idx)},
                });
        }

        return true;
    }

    bool MoESparseReturnReduceStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements MoESparseReturnReduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.dense_output)
            reqs.addOutput("dense_output", params_.dense_output->shape(), toBufferTensorType(params_.dense_output->native_type()));
        return reqs;
    }

    StageBufferContract MoESparseReturnReduceStage::bufferContract() const
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

    StageDumpInfo MoESparseReturnReduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.dense_output)
            info.addOutput("dense_output", params_.dense_output, static_cast<size_t>(params_.seq_len), static_cast<size_t>(params_.d_model));
        info.addScalarInt("source_participant", params_.source_participant);
        info.addScalarInt("target_participant", params_.target_participant);
        info.addScalarInt("seq_len", params_.seq_len);
        info.addScalarInt("d_model", params_.d_model);
        info.addScalarBool("clear_output_before_scatter", params_.clear_output_before_scatter);
        info.addScalarBool("broadcast_after_scatter", params_.broadcast_after_scatter);
        info.addScalarBool(
            "protocol_participant",
            params_.inbound_consumer_role ==
                InboundConsumerRole::ProtocolParticipant);
        info.addScalarBool(
            "canonical_route_ticket_completion",
            params_.inbound_consumer_role ==
                InboundConsumerRole::CanonicalRouteTicketCompletion);
        info.addScalarBool("captured_return_ticket", params_.ticket_storage != nullptr);
        info.addScalarBool(
            "canonical_route_ticket",
            params_.canonical_route_ticket_storage != nullptr);
        info.addScalarBool("publish_ticket_completion",
                           params_.publish_ticket_completion);
        info.addScalarBool("release_residency_lease_on_completion",
                           params_.residency_lease_terminal ==
                               MoEOverlayHostDispatchLeaseTerminal::Release);
        info.addScalarInt("continuation_root_tp_index", params_.continuation_root_tp_index);
        if (params_.continuation_tp_context)
        {
            info.addScalarInt("continuation_tp_degree", params_.continuation_tp_context->degree());
            info.addScalarInt("continuation_tp_scope", static_cast<int>(params_.continuation_tp_context->scope()));
        }
        return info;
    }

} // namespace llaminar2
