/**
 * @file GlobalPPTransferStage.cpp
 * @brief Implementation of activation transfer stage for GLOBAL pipeline parallelism
 *
 * Cross-rank MPI point-to-point transfer of activations between pipeline stages.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include "GlobalPPTransferStage.h"
#include "../../../utils/MPIContext.h"
#include "../../../utils/Logger.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/TensorClasses.h"
#include <chrono>
#include <sstream>
#include <mpi.h>

namespace llaminar2
{

    // =========================================================================
    // Construction
    // =========================================================================

    GlobalPPTransferStage::GlobalPPTransferStage(Params params)
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
            if (params_.direction == GlobalPPTransferParams::Direction::SEND)
            {
                oss << "gpp_send_to_rank" << params_.peer_rank << "_tag" << params_.tag;
            }
            else
            {
                oss << "gpp_recv_from_rank" << params_.peer_rank << "_tag" << params_.tag;
            }
            name_ = oss.str();
        }
    }

    // =========================================================================
    // IComputeStage Implementation
    // =========================================================================

    bool GlobalPPTransferStage::execute(IDeviceContext *ctx)
    {
        (void)ctx; // MPI operations don't use device context

        // Validate common parameters
        if (!params_.tensor)
        {
            LOG_ERROR("[GlobalPPTransferStage] " << name_ << ": null tensor");
            return false;
        }

        if (!params_.mpi_ctx)
        {
            LOG_ERROR("[GlobalPPTransferStage] " << name_ << ": null MPI context");
            return false;
        }

        if (params_.peer_rank < 0)
        {
            LOG_ERROR("[GlobalPPTransferStage] " << name_ << ": invalid peer rank: " << params_.peer_rank);
            return false;
        }

        // Refuse to communicate with self
        if (params_.peer_rank == params_.mpi_ctx->rank())
        {
            LOG_ERROR("[GlobalPPTransferStage] " << name_ << ": peer_rank == my rank (" << params_.peer_rank << "), use LocalPPTransfer instead");
            return false;
        }

        if (params_.direction == GlobalPPTransferParams::Direction::SEND)
        {
            return executeSend();
        }
        else
        {
            return executeRecv();
        }
    }

    bool GlobalPPTransferStage::executeSend()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        // data() triggers GPU→host sync if tensor is device-dirty
        const float *data = params_.tensor->data();
        const size_t count = (params_.count > 0) ? params_.count : params_.tensor->numel();
        const size_t bytes = count * sizeof(float);

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": sending " << count
                                                << " floats (" << bytes << " bytes) to rank " << params_.peer_rank
                                                << " tag=" << params_.tag);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Synchronous blocking send
        params_.mpi_ctx->send(data, count, MPI_FLOAT, params_.peer_rank, params_.tag);

        auto end_time = std::chrono::high_resolution_clock::now();

        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double bandwidth_gbps = (ms > 0) ? (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0) : 0.0;
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": send completed in "
                                                << ms << " ms (" << bandwidth_gbps << " GB/s)");
        }
        else if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": send completed");
        }

        return true;
    }

    bool GlobalPPTransferStage::executeRecv()
    {
        const auto &mpi_env = debugEnv().mpi_logging;

        // mutable_data() ensures tensor is on host and marks it as host-authoritative
        float *data = params_.tensor->mutable_data();
        const size_t count = (params_.count > 0) ? params_.count : params_.tensor->numel();
        const size_t bytes = count * sizeof(float);

        if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": receiving " << count
                                                << " floats (" << bytes << " bytes) from rank " << params_.peer_rank
                                                << " tag=" << params_.tag);
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        // Synchronous blocking receive
        params_.mpi_ctx->recv(data, count, MPI_FLOAT, params_.peer_rank, params_.tag);

        // Mark tensor as host-authoritative after receiving new data
        params_.tensor->mark_host_dirty();

        auto end_time = std::chrono::high_resolution_clock::now();

        if (mpi_env.log_timing)
        {
            double ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            double bandwidth_gbps = (ms > 0) ? (bytes / (ms / 1000.0)) / (1024.0 * 1024.0 * 1024.0) : 0.0;
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": recv completed in "
                                                << ms << " ms (" << bandwidth_gbps << " GB/s)");
        }
        else if (mpi_env.log_collectives)
        {
            LOG_DEBUG("[GlobalPPTransferStage] " << name_ << ": recv completed");
        }

        return true;
    }

    std::string GlobalPPTransferStage::name() const
    {
        return name_;
    }

    bool GlobalPPTransferStage::supportsBackend(ComputeBackendType backend) const
    {
        // MPI handles communication — we accept any compute backend
        (void)backend;
        return true;
    }

    StageBufferRequirements GlobalPPTransferStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.tensor)
        {
            BufferTensorType buf_type = toBufferTensorType(params_.tensor->native_type());

            if (params_.direction == GlobalPPTransferParams::Direction::SEND)
            {
                reqs.addInput("tensor", params_.tensor->shape(), buf_type);
            }
            else
            {
                reqs.addOutput("tensor", params_.tensor->shape(), buf_type);
            }
        }

        return reqs;
    }

    StageDumpInfo GlobalPPTransferStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.tensor)
        {
            const size_t rows = params_.tensor->rows();
            const size_t cols = params_.tensor->cols();

            if (params_.direction == GlobalPPTransferParams::Direction::SEND)
            {
                info.addInput("tensor", params_.tensor, rows, cols);
            }
            else
            {
                info.addOutput("tensor", params_.tensor, rows, cols);
            }
        }

        // Add transfer metadata
        info.addScalarInt("direction", static_cast<int>(params_.direction));
        info.addScalarInt("peer_rank", params_.peer_rank);
        info.addScalarInt("tag", params_.tag);

        size_t count = (params_.count > 0) ? params_.count
                       : (params_.tensor ? params_.tensor->numel() : 0);
        info.addScalarInt("count", static_cast<int>(count));

        if (params_.mpi_ctx)
        {
            info.addScalarInt("my_rank", params_.mpi_ctx->rank());
            info.addScalarInt("world_size", params_.mpi_ctx->world_size());
        }

        return info;
    }

    void GlobalPPTransferStage::setParams(const Params &params)
    {
        params_ = params;

        // Regenerate name
        if (!params_.stage_name.empty())
        {
            name_ = params_.stage_name;
        }
        else
        {
            std::ostringstream oss;
            if (params_.direction == GlobalPPTransferParams::Direction::SEND)
            {
                oss << "gpp_send_to_rank" << params_.peer_rank << "_tag" << params_.tag;
            }
            else
            {
                oss << "gpp_recv_from_rank" << params_.peer_rank << "_tag" << params_.tag;
            }
            name_ = oss.str();
        }
    }

} // namespace llaminar2
