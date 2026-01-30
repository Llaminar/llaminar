/**
 * @file AllGatherVStage.cpp
 * @brief Implementation of AllGatherVStage with CollectiveContext support
 *
 * Supports two execution modes:
 * 1. CollectiveContext (preferred): Uses new collective infrastructure
 * 2. Direct MPI (legacy): Falls back to MPI for backward compatibility
 */

#include "AllGatherVStage.h"
#include "../ComputeStageUtils.h"
#include "../../../execution/local_execution/collective/CollectiveContext.h"
#include "../../../collective/ICollectiveBackend.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/MPIContext.h"
#include <mpi.h>
#include <chrono>
#include <numeric>

namespace llaminar2
{

    // =============================================================================
    // AllGatherVStage Implementation
    // =============================================================================

    AllGatherVStage::AllGatherVStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AllGatherVStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.local_input)
        {
            LOG_ERROR("[AllGatherVStage] Null local_input buffer");
            return false;
        }

        if (!params_.full_output)
        {
            LOG_ERROR("[AllGatherVStage] Null full_output buffer");
            return false;
        }

        if (params_.recv_counts.empty())
        {
            LOG_ERROR("[AllGatherVStage] Empty recv_counts array");
            return false;
        }

        if (params_.displacements.empty())
        {
            LOG_ERROR("[AllGatherVStage] Empty displacements array");
            return false;
        }

        if (params_.recv_counts.size() != params_.displacements.size())
        {
            LOG_ERROR("[AllGatherVStage] recv_counts.size() != displacements.size()");
            return false;
        }

        // Prefer CollectiveContext if available
        if (params_.collective_ctx)
        {
            return executeViaCollectiveContext();
        }

        // Fall back to direct MPI
        return executeViaMPI();
    }

    bool AllGatherVStage::executeViaCollectiveContext()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        // Get shapes
        const auto &local_shape = params_.local_input->shape();
        const auto &full_shape = params_.full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("[AllGatherVStage] Expected 2D tensors, got local_shape.size()="
                      << local_shape.size() << " full_shape.size()=" << full_shape.size());
            return false;
        }

        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : buffer_seq_len;
        size_t local_dim = local_shape[1];

        // Calculate total expected elements
        size_t total_recv = std::accumulate(params_.recv_counts.begin(),
                                            params_.recv_counts.end(), 0);

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherVStage] Using CollectiveContext, seq_len=" << seq_len
                                                                           << " local_dim=" << local_dim
                                                                           << " total_recv_elements=" << total_recv);
        }

        // Determine device where tensor resides
        DeviceId tensor_device = DeviceId::cpu(); // Default to CPU
        // TODO: Query tensor for actual device once we have multi-device support

        // Delegate to CollectiveContext
        bool success = params_.collective_ctx->executeAllgatherv(
            params_.local_input,
            params_.full_output,
            params_.recv_counts,
            params_.displacements,
            seq_len,
            tensor_device);

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherVStage] CollectiveContext result=" << (success ? "SUCCESS" : "FAILED"));
        }

        return success;
    }

    bool AllGatherVStage::executeViaMPI()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[AllGatherVStage] Null MPI context");
            return false;
        }

        int world_size = params_.mpi_ctx->world_size();
        int rank = params_.mpi_ctx->rank();

        if (world_size <= 0)
        {
            LOG_ERROR("[AllGatherVStage] Invalid world_size=" << world_size);
            return false;
        }

        if (static_cast<int>(params_.recv_counts.size()) != world_size)
        {
            LOG_ERROR("[AllGatherVStage] recv_counts.size() (" << params_.recv_counts.size()
                                                               << ") != world_size (" << world_size << ")");
            return false;
        }

        // Get shapes
        const auto &local_shape = params_.local_input->shape();
        const auto &full_shape = params_.full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("[AllGatherVStage] Expected 2D tensors");
            return false;
        }

        // Use actual_seq_len if provided, otherwise use buffer shape
        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : buffer_seq_len;
        size_t local_dim = local_shape[1];

        // Calculate send count for this rank
        size_t send_count = seq_len * local_dim;

        // Scale recv_counts and displacements by seq_len
        std::vector<int> scaled_recv_counts(world_size);
        std::vector<int> scaled_displacements(world_size);
        for (int i = 0; i < world_size; ++i)
        {
            scaled_recv_counts[i] = static_cast<int>(seq_len) * params_.recv_counts[i];
            scaled_displacements[i] = static_cast<int>(seq_len) * params_.displacements[i];
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherVStage] MPI_Allgatherv rank=" << rank
                                                              << " send_count=" << send_count
                                                              << " seq_len=" << seq_len
                                                              << " local_dim=" << local_dim);
        }

        std::chrono::high_resolution_clock::time_point start;
        if (mpi_env.log_timing)
        {
            start = std::chrono::high_resolution_clock::now();
        }

        // Get data pointers
        const float *send_data = params_.local_input->data();
        float *recv_data = params_.full_output->mutable_data();

        // Execute MPI_Allgatherv
        int result = MPI_Allgatherv(
            send_data,
            static_cast<int>(send_count),
            MPI_FLOAT,
            recv_data,
            scaled_recv_counts.data(),
            scaled_displacements.data(),
            MPI_FLOAT,
            params_.mpi_ctx->comm());

        if (mpi_env.log_timing)
        {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
            LOG_INFO("[AllGatherVStage] MPI_Allgatherv duration=" << duration_us << " us");
        }

        if (result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherVStage] MPI_Allgatherv failed with code " << result);
            return false;
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherVStage] MPI_Allgatherv completed successfully");
        }

        return true;
    }

    bool AllGatherVStage::supportsBackend(ComputeBackendType backend) const
    {
        // AllGatherV works on any backend (MPI operates on CPU-staged data)
        (void)backend;
        return true;
    }

    StageBufferRequirements AllGatherVStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // AllGatherV has separate input and output buffers
        if (params_.local_input)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.local_input->native_type());
            reqs.addInput("local_input", params_.local_input->shape(), buf_type);
        }

        if (params_.full_output)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.full_output->native_type());
            reqs.addOutput("full_output", params_.full_output->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo AllGatherVStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Local input slice (what this rank contributes)
        if (params_.local_input)
        {
            size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : params_.local_input->rows();
            info.addInput("local_input", params_.local_input, seq_len, params_.local_input->cols());
        }

        // Full gathered output (replicated on all ranks)
        if (params_.full_output)
        {
            size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : params_.full_output->rows();
            info.addOutput("full_output", params_.full_output, seq_len, params_.full_output->cols());
        }

        info.addScalarInt("world_size", params_.mpi_ctx ? params_.mpi_ctx->world_size() : 0);
        info.addScalarInt("actual_seq_len", static_cast<int>(params_.actual_seq_len));
        info.addScalarInt("recv_counts_size", static_cast<int>(params_.recv_counts.size()));

        return info;
    }

} // namespace llaminar2
