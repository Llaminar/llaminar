/**
 * @file ReceiveActivationsStage.cpp
 * @brief Implementation of ReceiveActivationsStage for pipeline parallelism
 *
 * Receives activations from the previous pipeline stage via MPI point-to-point
 * communication. Supports both synchronous and asynchronous modes.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "ReceiveActivationsStage.h"
#include "../../../utils/MPIContext.h"
#include "../../../utils/MPITags.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include <chrono>
#include <sstream>

namespace llaminar2
{

    ReceiveActivationsStage::ReceiveActivationsStage(Params params)
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
            if (params_.src_rank >= 0)
            {
                oss << "recv_activations_from_rank" << params_.src_rank << "_tag" << params_.tag;
            }
            else
            {
                oss << "recv_activations_any_source_tag" << params_.tag;
            }
            name_ = oss.str();
        }
    }

    bool ReceiveActivationsStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // MPI operations don't use device context

        if (!params_.buffer)
        {
            LOG_ERROR("[ReceiveActivationsStage] Null buffer");
            return false;
        }

        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[ReceiveActivationsStage] Null MPI context");
            return false;
        }

        const auto &mpi_env = debugEnv().mpi_logging;

        // Get mutable data pointer for receiving
        // For receive, we need mutable access - this also ensures tensor is on host
        float *data = params_.buffer->mutable_data();
        size_t count = params_.buffer->numel();
        size_t bytes = count * sizeof(float);

        // Convert -1 to MPI wildcards
        int src_rank = (params_.src_rank < 0) ? MPI_ANY_SOURCE : params_.src_rank;
        int tag = (params_.tag < 0) ? MPI_ANY_TAG : params_.tag;

        if (mpi_env.log_collectives)
        {
            LOG_INFO("[ReceiveActivationsStage] " << name_ << ": receiving " << count
                                                  << " floats (" << bytes << " bytes) from rank "
                                                  << (params_.src_rank < 0 ? "ANY" : std::to_string(params_.src_rank))
                                                  << " tag=" << params_.tag << " async=" << params_.async);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        if (params_.async)
        {
            // Non-blocking receive
            pending_request_ = params_.mpi_ctx->irecv(
                data, count, MPI_FLOAT, src_rank, tag);

            if (mpi_env.log_collectives)
            {
                LOG_INFO("[ReceiveActivationsStage] " << name_ << ": irecv initiated");
            }
        }
        else
        {
            // Blocking receive
            params_.mpi_ctx->recv(data, count, MPI_FLOAT, src_rank, tag);

            // Mark tensor as CPU-authoritative after receiving data
            params_.buffer->mark_host_dirty();

            auto end_time = std::chrono::high_resolution_clock::now();

            if (mpi_env.log_timing)
            {
                double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                double bandwidth_gbps = (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0);
                LOG_INFO("[ReceiveActivationsStage] " << name_ << ": recv completed in "
                                                      << ms << " ms (" << bandwidth_gbps << " GB/s)");
            }
            else if (mpi_env.log_collectives)
            {
                LOG_INFO("[ReceiveActivationsStage] " << name_ << ": recv completed");
            }
        }

        return true;
    }

    bool ReceiveActivationsStage::isComplete() const
    {
        if (pending_request_ == MPI_REQUEST_NULL)
        {
            return true; // No pending operation
        }

        int flag = 0;
        MPI_Test(const_cast<MPI_Request *>(&pending_request_), &flag, MPI_STATUS_IGNORE);
        return flag != 0;
    }

    void ReceiveActivationsStage::wait()
    {
        if (pending_request_ != MPI_REQUEST_NULL)
        {
            params_.mpi_ctx->wait(&pending_request_);

            // Mark tensor as CPU-authoritative after receiving data
            if (params_.buffer)
            {
                params_.buffer->mark_host_dirty();
            }

            const auto &mpi_env = debugEnv().mpi_logging;
            if (mpi_env.log_collectives)
            {
                LOG_INFO("[ReceiveActivationsStage] " << name_ << ": async recv completed via wait()");
            }
        }
    }

    bool ReceiveActivationsStage::supportsBackend(ComputeBackendType backend) const
    {
        // MPI receive is backend-agnostic (works with any device that can provide host buffer)
        (void)backend;
        return true;
    }

    StageBufferRequirements ReceiveActivationsStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.buffer)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.buffer->native_type());
            reqs.addOutput("buffer", params_.buffer->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo ReceiveActivationsStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.buffer)
        {
            info.addOutput("buffer", params_.buffer, params_.buffer->rows(), params_.buffer->cols());
        }

        info.addScalarInt("src_rank", params_.src_rank);
        info.addScalarInt("tag", params_.tag);
        info.addScalarBool("async", params_.async);

        return info;
    }

} // namespace llaminar2
