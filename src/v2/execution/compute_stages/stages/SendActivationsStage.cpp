/**
 * @file SendActivationsStage.cpp
 * @brief Implementation of SendActivationsStage for pipeline parallelism
 *
 * Sends activations to the next pipeline stage via MPI point-to-point
 * communication. Supports both synchronous and asynchronous modes.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "SendActivationsStage.h"
#include "../../../utils/MPIContext.h"
#include "../../../utils/MPITags.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include <chrono>
#include <sstream>

namespace llaminar2
{

    SendActivationsStage::SendActivationsStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Generate stage name
        if (!params_.stage_name.empty())
        {
            name_ = params_.stage_name;
        }
        else
        {
            std::ostringstream oss;
            oss << "send_activations_to_rank" << params_.dest_rank << "_tag" << params_.tag;
            name_ = oss.str();
        }
    }

    bool SendActivationsStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // MPI operations don't use device context

        if (!params_.buffer)
        {
            LOG_ERROR("[SendActivationsStage] Null buffer");
            return false;
        }

        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[SendActivationsStage] Null MPI context");
            return false;
        }

        if (params_.dest_rank < 0)
        {
            LOG_ERROR("[SendActivationsStage] Invalid destination rank: " << params_.dest_rank);
            return false;
        }

        const auto &mpi_env = debugEnv().mpi_logging;

        // Get data from tensor - ensure it's on host for MPI
        // data() triggers GPU→host sync if tensor is device-dirty
        const float *data = params_.buffer->data();
        size_t count = params_.buffer->numel();
        size_t bytes = count * sizeof(float);

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[SendActivationsStage] " << name_ << ": sending " << count
                                               << " floats (" << bytes << " bytes) to rank " << params_.dest_rank
                                               << " tag=" << params_.tag << " async=" << params_.async);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        if (params_.async)
        {
            // Non-blocking send
            pending_request_ = params_.mpi_ctx->isend(
                data, count, MPI_FLOAT, params_.dest_rank, params_.tag);

            if (mpi_env.log_collectives)
            {
                LOG_INFO("[SendActivationsStage] " << name_ << ": isend initiated");
            }
        }
        else
        {
            // Blocking send
            params_.mpi_ctx->send(data, count, MPI_FLOAT, params_.dest_rank, params_.tag);

            auto end_time = std::chrono::high_resolution_clock::now();

            if (mpi_env.log_timing)
            {
                double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
                LOG_INFO("[SendActivationsStage] " << name_ << ": send completed in "
                                                   << ms << " ms (" << bandwidth_gbps << " GB/s)");
            }
            else if (mpi_env.log_collectives)
            {
                LOG_INFO("[SendActivationsStage] " << name_ << ": send completed");
            }
        }

        return true;
    }

    bool SendActivationsStage::isComplete() const
    {
        if (pending_request_ == MPI_REQUEST_NULL)
        {
            return true; // No pending operation
        }

        int flag = 0;
        MPI_Test(const_cast<MPI_Request *>(&pending_request_), &flag, MPI_STATUS_IGNORE);
        return flag != 0;
    }

    void SendActivationsStage::wait()
    {
        if (pending_request_ != MPI_REQUEST_NULL)
        {
            params_.mpi_ctx->wait(&pending_request_);

            const auto &mpi_env = debugEnv().mpi_logging;
            if (mpi_env.log_collectives)
            {
                LOG_INFO("[SendActivationsStage] " << name_ << ": async send completed via wait()");
            }
        }
    }

    bool SendActivationsStage::supportsBackend(ComputeBackendType backend) const
    {
        // MPI send is backend-agnostic (works with any device that can provide host data)
        (void)backend;
        return true;
    }

    StageBufferRequirements SendActivationsStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.buffer)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.buffer->native_type());
            reqs.addInput("buffer", params_.buffer->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo SendActivationsStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.buffer)
        {
            info.addInput("buffer", params_.buffer, params_.buffer->rows(), params_.buffer->cols());
        }

        info.addScalarInt("dest_rank", params_.dest_rank);
        info.addScalarInt("tag", params_.tag);
        info.addScalarBool("async", params_.async);

        return info;
    }

} // namespace llaminar2
