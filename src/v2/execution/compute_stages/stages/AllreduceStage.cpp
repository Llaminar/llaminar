/**
 * @file AllreduceStage.cpp
 * @brief Implementation of AllreduceStage with CollectiveContext support
 *
 * Supports two execution modes:
 * 1. CollectiveContext (preferred): Uses new collective infrastructure
 * 2. Direct MPI (legacy): Falls back to MPI for backward compatibility
 */

#include "AllreduceStage.h"
#include "../ComputeStageUtils.h"
#include "../../../execution/local_execution/collective/CollectiveContext.h"
#include "../../../collective/ICollectiveBackend.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/MPIContext.h"
#include <mpi.h>
#include <chrono>

namespace llaminar2
{

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        if (!params_.buffer)
        {
            LOG_ERROR("[AllreduceStage] Null buffer");
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

    bool AllreduceStage::executeViaCollectiveContext()
    {
        KERNEL_PROFILE_SCOPE(KernelType::ALLREDUCE);

        const auto &mpi_env = debugEnv().mpi_logging;

        size_t count = params_.count > 0 ? params_.count : params_.buffer->numel();

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[AllreduceStage] Using CollectiveContext, count=" << count
                                                                        << " dtype=" << params_.buffer->dtype_name());
        }

        // Determine device where tensor resides
        DeviceId tensor_device = DeviceId::cpu(); // Default to CPU
        // TODO: Query tensor for actual device once we have multi-device support

        // Delegate to CollectiveContext
        bool success = params_.collective_ctx->executeAllreduce(
            params_.buffer,
            count,
            tensor_device,
            CollectiveOp::ALLREDUCE_SUM);

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[AllreduceStage] CollectiveContext result=" << (success ? "SUCCESS" : "FAILED"));
        }

        return success;
    }

    bool AllreduceStage::executeViaMPI()
    {
        KERNEL_PROFILE_SCOPE(KernelType::ALLREDUCE);

        const auto &mpi_env = debugEnv().mpi_logging;

        if (!params_.buffer)
        {
            LOG_ERROR("[AllreduceStage] Null buffer");
            return false;
        }

        // Use explicit count if provided, otherwise use buffer size
        size_t count = params_.count > 0 ? params_.count : params_.buffer->numel();

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[MPI] AllReduce START: count=" << count
                                                     << " dtype=" << params_.buffer->dtype_name());
        }

        LOG_DEBUG("[AllreduceStage] Execute: buffer=" << params_.buffer
                                                      << " count=" << count << " has_mpi_ctx=" << (params_.mpi_ctx != nullptr));
        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[AllreduceStage] Null MPI context");
            return false;
        }

        // Get MPI communicator - prefer domain-specific communicator if available
        MPI_Comm comm = MPI_COMM_NULL;
        if (params_.domain && params_.domain->communicator != MPI_COMM_NULL)
        {
            comm = params_.domain->communicator;
            LOG_DEBUG("[AllreduceStage] Using domain communicator: " << params_.domain->name
                                                                     << " (size=" << params_.domain->domain_size << ")");
        }
        else
        {
            comm = params_.mpi_ctx->communicator();
            LOG_DEBUG("[AllreduceStage] Using MPI context communicator (legacy path)");
        }

        // Start timing if enabled
        auto start_time = std::chrono::high_resolution_clock::now();

        bool success = false;

        // Handle different tensor types
        if (params_.buffer->native_type() == TensorType::FP32)
        {
            // FP32: Direct MPI_Allreduce
            auto *fp32_tensor = dynamic_cast<FP32Tensor *>(params_.buffer);
            if (fp32_tensor)
            {
                float *data_ptr = fp32_tensor->mutable_data();

                LOG_DEBUG("[AllreduceStage] FP32 path: before data[0:4]="
                          << data_ptr[0] << "," << data_ptr[1] << "," << data_ptr[2] << "," << data_ptr[3]);

                int result = MPI_Allreduce(
                    MPI_IN_PLACE,
                    data_ptr,
                    static_cast<int>(count),
                    MPI_FLOAT,
                    MPI_SUM,
                    comm);

                LOG_DEBUG("[AllreduceStage] FP32 path: after data[0:4]="
                          << data_ptr[0] << "," << data_ptr[1] << "," << data_ptr[2] << "," << data_ptr[3]);

                success = (result == MPI_SUCCESS);
            }
        }
        else if (params_.buffer->native_type() == TensorType::Q16_1)
        {
            // Q16_1: Use IMPIContext::allreduce_q16_1_inplace for efficient SIMD reduction
            auto *q16_tensor = dynamic_cast<Q16_1Tensor *>(params_.buffer);
            if (q16_tensor)
            {
                Q16_1Block *blocks = q16_tensor->mutable_typed_data();
                size_t n_blocks = (count + 31) / 32;

                LOG_DEBUG("[AllreduceStage] Q16_1 path: n_blocks=" << n_blocks
                                                                   << " using IMPIContext allreduce");

                params_.mpi_ctx->allreduce_q16_1_inplace(blocks, n_blocks);
                success = true;
            }
        }
        else if (params_.buffer->native_type() == TensorType::Q8_1)
        {
            // Q8_1: Use IMPIContext::allreduce_q8_1_inplace for efficient SIMD reduction
            auto *q8_tensor = dynamic_cast<Q8_1Tensor *>(params_.buffer);
            if (q8_tensor)
            {
                Q8_1Block *blocks = q8_tensor->mutable_typed_data();
                size_t n_blocks = (count + 31) / 32;

                LOG_DEBUG("[AllreduceStage] Q8_1 path: n_blocks=" << n_blocks
                                                                  << " using IMPIContext allreduce");

                params_.mpi_ctx->allreduce_q8_1_inplace(blocks, n_blocks);
                success = true;
            }
        }
        else
        {
            LOG_ERROR("[AllreduceStage] Unsupported tensor type for allreduce: "
                      << params_.buffer->dtype_name());
            return false;
        }

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Log timing if enabled
        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double bytes = count * sizeof(float); // Approximate
            double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            LOG_DEBUG("[MPI] AllReduce timing: " << ms << " ms for " << count
                                                << " elements (" << bandwidth_gbps << " GB/s)");
        }

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[MPI] AllReduce END: result=" << (success ? "SUCCESS" : "FAILED"));
        }

        return success;
    }

    bool AllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // Allreduce is backend-agnostic (works with any device that has MPI support)
        (void)backend;
        return true;
    }

    StageBufferRequirements AllreduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        // Allreduce operates in-place on a single buffer
        if (params_.buffer)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.buffer->native_type());
            reqs.addInout("buffer", params_.buffer->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo AllreduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        // Allreduce operates in-place - buffer is both input and output
        if (params_.buffer)
        {
            size_t count = params_.count > 0 ? params_.count : params_.buffer->numel();
            info.addInput("buffer", params_.buffer, params_.buffer->rows(), params_.buffer->cols());
            info.addOutput("buffer", params_.buffer, params_.buffer->rows(), params_.buffer->cols());
            info.addScalarInt("count", static_cast<int>(count));
        }

        return info;
    }

} // namespace llaminar2
