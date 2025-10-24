/**
 * @file PipelineBase.cpp
 * @brief Base pipeline implementation
 * @author David Sanftenberg
 */

#include "PipelineBase.h"
#include <iostream>

namespace llaminar2
{

    PipelineBase::PipelineBase(std::shared_ptr<ModelContext> model_ctx,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               int device_idx)
        : model_ctx_(model_ctx), mpi_ctx_(mpi_ctx), device_idx_(device_idx)
    {
        if (!model_ctx_)
        {
            throw std::runtime_error("PipelineBase: model_ctx cannot be null");
        }

        model_path_ = model_ctx_->path();

        std::cout << "[PipelineBase] Initializing with model: " << model_path_ << "\n";

        if (mpi_ctx_)
        {
            std::cout << "[PipelineBase] MPI context provided, rank "
                      << mpi_ctx_->rank() << "/" << mpi_ctx_->world_size() << "\n";
        }

        if (device_idx_ >= 0)
        {
            std::cout << "[PipelineBase] Device index: " << device_idx_ << " (GPU)\n";
        }
        else
        {
            std::cout << "[PipelineBase] Device index: " << device_idx_ << " (CPU)\n";
        }
    }

} // namespace llaminar2
