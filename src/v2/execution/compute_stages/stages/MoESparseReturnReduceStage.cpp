/**
 * @file MoESparseReturnReduceStage.cpp
 * @brief Implementation of graph-native sparse MoE return/reduce payload stage.
 */

#include "MoESparseReturnReduceStage.h"

#include "../../../collective/ITPContext.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"

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
    }

    bool MoESparseReturnReduceStage::execute(IDeviceContext *ctx)
    {
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
        runtime_key.step_id = execution_count_++;

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
        if (!validateReturnRows(*params_.outbound_rows, params_.d_model) ||
            !validateDenseOutput(params_.dense_output, params_.seq_len, params_.d_model))
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

        auto result = params_.collective_context->returnReduce(runtime_key, outbound, params_.inbound_rows, ctx);
        if (!result.ok)
        {
            LOG_ERROR("[MoESparseReturnReduceStage] Return-reduce collective failed: " << result.error);
            return false;
        }

        double prof_return_wait_ms = 0.0;
        if (MoEExpertOverlayProfiler::isEnabled())
            prof_return_wait_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t_return_start)
                                      .count();

        if (!validateReturnRows(*params_.inbound_rows, params_.d_model))
            return false;

        float *dense = params_.dense_output->mutable_data();
        const size_t dense_count = static_cast<size_t>(params_.seq_len) * static_cast<size_t>(params_.d_model);
        if (params_.clear_output_before_scatter)
            std::fill_n(dense, dense_count, 0.0f);

        std::chrono::steady_clock::time_point t_scatter_start;
        if (MoEExpertOverlayProfiler::isEnabled())
            t_scatter_start = std::chrono::steady_clock::now();

        for (size_t compact_row = 0; compact_row < params_.inbound_rows->live_row_count; ++compact_row)
        {
            const int row_id = params_.inbound_rows->row_ids_host[compact_row];
            if (row_id < 0 || row_id >= params_.seq_len)
            {
                LOG_ERROR("[MoESparseReturnReduceStage] Returned row id " << row_id
                                                                          << " outside seq_len=" << params_.seq_len);
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

        if (params_.broadcast_after_scatter && result.collective_complete)
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

        if (MoEExpertOverlayProfiler::isEnabled())
        {
            const size_t compact_bytes = compactMoEOverlayReturnBytes(*params_.outbound_rows);
            const size_t dense_bytes = denseMoEOverlayReturnBytes(params_.seq_len, params_.d_model);
            MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
                params_.key.layer_idx,
                runtime_key.toString(),
                params_.source_participant,
                params_.target_participant,
                outbound.live_row_count,
                params_.inbound_rows->live_row_count,
                compact_bytes,
                dense_bytes,
                prof_return_wait_ms,
                prof_scatter_ms,
                prof_broadcast_ms);
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
        info.addScalarInt("continuation_root_tp_index", params_.continuation_root_tp_index);
        if (params_.continuation_tp_context)
        {
            info.addScalarInt("continuation_tp_degree", params_.continuation_tp_context->degree());
            info.addScalarInt("continuation_tp_scope", static_cast<int>(params_.continuation_tp_context->scope()));
        }
        return info;
    }

} // namespace llaminar2