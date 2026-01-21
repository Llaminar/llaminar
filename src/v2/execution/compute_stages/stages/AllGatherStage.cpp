/**
 * @file AllGatherStage.cpp
 * @brief Implementation of AllGatherStage with CollectiveContext support
 *
 * Supports two execution modes:
 * 1. CollectiveContext (preferred): Uses new collective infrastructure
 * 2. Direct MPI (legacy): Falls back to MPI for backward compatibility
 */

#include "AllGatherStage.h"
#include "../ComputeStageUtils.h"
#include "../../../execution/CollectiveContext.h"
#include "../../../collective/ICollectiveBackend.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/MPIContext.h"
#include <mpi.h>
#include <chrono>

namespace llaminar2
{

    // =============================================================================
    // AllGatherStage Implementation
    // =============================================================================

    AllGatherStage::AllGatherStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AllGatherStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.local_input)
        {
            LOG_ERROR("[AllGatherStage] Null local_input buffer");
            return false;
        }

        if (!params_.full_output)
        {
            LOG_ERROR("[AllGatherStage] Null full_output buffer");
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

    bool AllGatherStage::executeViaCollectiveContext()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        // Get shapes
        const auto &local_shape = params_.local_input->shape();
        const auto &full_shape = params_.full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("[AllGatherStage] Expected 2D tensors, got local_shape.size()="
                      << local_shape.size() << " full_shape.size()=" << full_shape.size());
            return false;
        }

        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : buffer_seq_len;

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherStage] Using CollectiveContext, seq_len=" << seq_len
                                                                          << " local_dim=" << local_shape[1]
                                                                          << " full_dim=" << full_shape[1]);
        }

        // Determine device where tensor resides
        DeviceId tensor_device = DeviceId::cpu(); // Default to CPU
        // TODO: Query tensor for actual device once we have multi-device support

        // Delegate to CollectiveContext
        bool success = params_.collective_ctx->executeAllgather(
            params_.local_input,
            params_.full_output,
            seq_len,
            tensor_device);

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[AllGatherStage] CollectiveContext result=" << (success ? "SUCCESS" : "FAILED"));
        }

        return success;
    }

    bool AllGatherStage::executeViaMPI()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        if (!params_.local_input)
        {
            LOG_ERROR("[AllGatherStage] Null local_input buffer");
            return false;
        }

        if (!params_.full_output)
        {
            LOG_ERROR("[AllGatherStage] Null full_output buffer");
            return false;
        }

        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[AllGatherStage] Null MPI context");
            return false;
        }

        int world_size = params_.mpi_ctx->world_size();
        if (world_size <= 0)
        {
            LOG_ERROR("[AllGatherStage] Invalid world_size=" << world_size);
            return false;
        }

        // Get shapes: local_input = [seq_len, vocab_local], full_output = [seq_len, vocab_size]
        const auto &local_shape = params_.local_input->shape();
        const auto &full_shape = params_.full_output->shape();

        if (local_shape.size() != 2 || full_shape.size() != 2)
        {
            LOG_ERROR("[AllGatherStage] Expected 2D tensors, got local_shape.size()="
                      << local_shape.size() << " full_shape.size()=" << full_shape.size());
            return false;
        }

        // Use actual_seq_len if provided, otherwise use buffer shape
        size_t buffer_seq_len = local_shape[0];
        size_t seq_len = params_.actual_seq_len > 0 ? params_.actual_seq_len : buffer_seq_len;
        size_t vocab_local = local_shape[1];
        size_t vocab_full = full_shape[0] == buffer_seq_len ? full_shape[1] : full_shape[0];

        // Validate seq_len doesn't exceed buffer capacity
        if (seq_len > buffer_seq_len)
        {
            LOG_ERROR("[AllGatherStage] actual_seq_len=" << seq_len
                                                         << " exceeds buffer capacity=" << buffer_seq_len);
            return false;
        }

        size_t expected_vocab_full = vocab_local * static_cast<size_t>(world_size);
        if (vocab_full != expected_vocab_full)
        {
            LOG_WARN("[AllGatherStage] vocab_full=" << vocab_full
                                                    << " expected=" << expected_vocab_full
                                                    << " - proceeding anyway");
        }

        LOG_DEBUG("[AllGatherStage] Execute: seq_len=" << seq_len
                                                       << " vocab_local=" << vocab_local
                                                       << " vocab_full=" << vocab_full
                                                       << " world_size=" << world_size);

        // Get MPI communicator - prefer domain-specific communicator if available
        MPI_Comm comm = MPI_COMM_NULL;
        if (params_.domain && params_.domain->communicator != MPI_COMM_NULL)
        {
            comm = params_.domain->communicator;
            LOG_DEBUG("[AllGatherStage] Using domain communicator: " << params_.domain->name
                                                                     << " (size=" << params_.domain->domain_size << ")");
        }
        else
        {
            comm = params_.mpi_ctx->comm();
            LOG_DEBUG("[AllGatherStage] Using MPI context communicator (legacy path)");
        }

        // Get data pointers based on tensor type
        const float *send_ptr = nullptr;
        float *recv_ptr = nullptr;

        if (params_.local_input->native_type() == TensorType::FP32)
        {
            auto *input_fp32 = dynamic_cast<FP32Tensor *>(params_.local_input);
            auto *output_fp32 = dynamic_cast<FP32Tensor *>(params_.full_output);
            if (input_fp32 && output_fp32)
            {
                send_ptr = input_fp32->data();
                recv_ptr = output_fp32->mutable_data();
            }
        }

        if (!send_ptr || !recv_ptr)
        {
            LOG_ERROR("[AllGatherStage] Unsupported tensor type for allgather");
            return false;
        }

        // =========================================================================
        // Optimized AllGather using MPI_Type_vector for strided data
        // =========================================================================
        // Each rank has [seq_len, vocab_local] data laid out contiguously.
        // We need output [seq_len, vocab_full] where each row interleaves all ranks' data:
        //   row_i: [rank0's vocab_local elements, rank1's vocab_local elements, ...]
        //
        // Strategy: Use MPI_Type_vector to describe strided placement in output buffer.
        // - Send type: seq_len blocks of vocab_local, stride vocab_local (contiguous)
        // - Recv type: seq_len blocks of vocab_local, stride vocab_full (strided)
        //
        // MPI_Allgather with derived types places each rank's recv_type starting at:
        //   rank * recvcount * sizeof(recv_type_extent)
        // But we need placement at: rank * vocab_local elements within each row.
        //
        // Solution: Create a resized recv type where the extent equals vocab_local elements,
        // so rank r's data lands at offset r * vocab_local within the vocab dimension.
        // =========================================================================

        // Step 1: Create the basic strided vector type
        // seq_len blocks of vocab_local floats, each block starting vocab_full floats apart
        MPI_Datatype strided_type;
        int mpi_result = MPI_Type_vector(
            static_cast<int>(seq_len),     // count: number of blocks (rows)
            static_cast<int>(vocab_local), // blocklength: elements per block
            static_cast<int>(vocab_full),  // stride: distance between block starts (in elements)
            MPI_FLOAT,
            &strided_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_vector failed with error " << mpi_result);
            return false;
        }

        // Step 2: Resize the type so its extent = vocab_local * sizeof(float)
        // This ensures consecutive ranks' data are placed vocab_local apart in each row
        MPI_Datatype resized_type;
        MPI_Aint lb = 0;
        MPI_Aint extent = static_cast<MPI_Aint>(vocab_local) * sizeof(float);

        mpi_result = MPI_Type_create_resized(strided_type, lb, extent, &resized_type);
        MPI_Type_free(&strided_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_create_resized failed with error " << mpi_result);
            return false;
        }

        mpi_result = MPI_Type_commit(&resized_type);
        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Type_commit failed with error " << mpi_result);
            MPI_Type_free(&resized_type);
            return false;
        }

        // Log MPI collective start
        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllGather START: seq_len=" << seq_len
                                                       << " vocab_local=" << vocab_local
                                                       << " vocab_full=" << vocab_full
                                                       << " world_size=" << world_size);
        }

        // Start timing if enabled
        auto start_time = std::chrono::high_resolution_clock::now();

        // Step 3: Use MPI_Allgather with the resized type
        // Send: seq_len * vocab_local contiguous floats
        // Recv: 1 resized_type per rank (each type covers seq_len strided blocks)
        mpi_result = MPI_Allgather(
            send_ptr,
            static_cast<int>(seq_len * vocab_local),
            MPI_FLOAT,
            recv_ptr,
            1,
            resized_type,
            comm);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Clean up the derived type
        MPI_Type_free(&resized_type);

        if (mpi_result != MPI_SUCCESS)
        {
            LOG_ERROR("[AllGatherStage] MPI_Allgather (strided) failed with error " << mpi_result);
            return false;
        }

        // Log timing if enabled
        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            size_t total_elements = seq_len * vocab_local;
            double bytes = total_elements * sizeof(float) * world_size; // Total bytes gathered
            double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO("[MPI] AllGather timing: " << ms << " ms for " << total_elements
                                                << " elements/rank (" << bandwidth_gbps << " GB/s aggregate)");
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllGather END: result=SUCCESS");
        }

        // Debug: dump gathered logits at last position
        LOG_DEBUG("[AllGatherStage] Gathered logits (last row, first 8): "
                  << recv_ptr[(seq_len - 1) * vocab_full + 0] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 1] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 2] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 3] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 4] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 5] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 6] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + 7]);
        // Also dump from second half (rank 1's data)
        LOG_DEBUG("[AllGatherStage] Gathered logits (last row, at vocab_local+0:+8): "
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 0] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 1] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 2] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 3] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 4] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 5] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 6] << ","
                  << recv_ptr[(seq_len - 1) * vocab_full + vocab_local + 7]);

        LOG_DEBUG("[AllGatherStage] Optimized AllGather (MPI_Type_vector) completed for "
                  << seq_len << " rows");
        return true;
    }

    bool AllGatherStage::supportsBackend(ComputeBackendType backend) const
    {
        // AllGather is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    StageBufferRequirements AllGatherStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // AllGather has separate input and output buffers
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

    StageDumpInfo AllGatherStage::getDumpInfo() const
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

        return info;
    }

} // namespace llaminar2
