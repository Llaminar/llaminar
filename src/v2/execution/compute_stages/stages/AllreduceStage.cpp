/**
 * @file AllreduceStage.cpp
 * @brief Implementation of AllreduceStage
 */

#include "AllreduceStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/MPIContext.h"
#include <mpi.h>

namespace llaminar2
{

    // =============================================================================
    // AllreduceStage Implementation
    // =============================================================================

    AllreduceStage::AllreduceStage(Params params) : params_(std::move(params)) {}

    bool AllreduceStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;
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
            LOG_INFO("[MPI] AllReduce START: count=" << count
                                                     << " dtype=" << params_.buffer->dtype_name());
        }

        LOG_DEBUG("[AllreduceStage] Execute: buffer=" << params_.buffer
                                                      << " count=" << count << " has_comm=" << (params_.mpi_comm != nullptr));
        if (!params_.mpi_comm)
        {
            LOG_ERROR("[AllreduceStage] Null MPI communicator");
            return false;
        }

        MPI_Comm comm = static_cast<MPI_Comm>(params_.mpi_comm);

        // Get mutable data pointer based on tensor type
        void *data_ptr = nullptr;
        MPI_Datatype mpi_type = MPI_FLOAT;

        if (params_.buffer->native_type() == TensorType::FP32)
        {
            auto *fp32_tensor = dynamic_cast<FP32Tensor *>(params_.buffer);
            if (fp32_tensor)
            {
                data_ptr = fp32_tensor->mutable_data();
                mpi_type = MPI_FLOAT;

                // Debug: dump values before AllReduce
                float *f = static_cast<float *>(data_ptr);
                LOG_DEBUG("[AllreduceStage] Before AllReduce: data[0:4]="
                          << f[0] << "," << f[1] << "," << f[2] << "," << f[3]);
            }
        }
        // Add other types as needed (BF16, FP16, etc.)

        if (!data_ptr)
        {
            LOG_ERROR("[AllreduceStage] Unsupported tensor type for allreduce");
            return false;
        }

        // Compute checksum before if verification enabled
        float checksum_before = 0.0f;
        if (mpi_env.verify_checksums && mpi_type == MPI_FLOAT)
        {
            const float *f = static_cast<const float *>(data_ptr);
            for (size_t i = 0; i < count; ++i)
            {
                checksum_before += f[i];
            }
            LOG_INFO("[MPI] AllReduce checksum BEFORE: " << checksum_before);
        }

        LOG_DEBUG("[AllreduceStage] Calling MPI_Allreduce with count=" << count);

        // Start timing if enabled
        auto start_time = std::chrono::high_resolution_clock::now();

        int result = MPI_Allreduce(
            MPI_IN_PLACE,
            data_ptr,
            static_cast<int>(count),
            mpi_type,
            MPI_SUM,
            comm);

        // End timing
        auto end_time = std::chrono::high_resolution_clock::now();

        // Debug: dump values after AllReduce
        if (mpi_type == MPI_FLOAT)
        {
            float *f = static_cast<float *>(data_ptr);
            LOG_DEBUG("[AllreduceStage] After AllReduce: data[0:4]="
                      << f[0] << "," << f[1] << "," << f[2] << "," << f[3]);
        }

        // Log timing if enabled
        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double bytes = count * sizeof(float);
            double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
            LOG_INFO("[MPI] AllReduce timing: " << ms << " ms for " << count
                                                << " elements (" << bandwidth_gbps << " GB/s)");
        }

        // Verify checksum after if enabled
        if (mpi_env.verify_checksums && mpi_type == MPI_FLOAT)
        {
            float checksum_after = 0.0f;
            const float *f = static_cast<const float *>(data_ptr);
            for (size_t i = 0; i < count; ++i)
            {
                checksum_after += f[i];
            }
            LOG_INFO("[MPI] AllReduce checksum AFTER: " << checksum_after);
        }

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[MPI] AllReduce END: result=" << (result == MPI_SUCCESS ? "SUCCESS" : "FAILED"));
        }

        LOG_DEBUG("[AllreduceStage] MPI_Allreduce returned " << result);
        return result == MPI_SUCCESS;
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

    StageDumpInfo AllreduceStage::getDumpInfo() const
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
