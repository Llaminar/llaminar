/**
 * @file MTPConcatStage.cpp
 * @brief Implementation of the MTP embedding/hidden concat stage.
 */

#include "MTPConcatStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../transfer/TransferEngine.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#ifdef HAVE_CUDA
#include "../../../kernels/cuda/ops/CUDARowSelectKernels.h"
#endif

#ifdef HAVE_ROCM
#include "../../../kernels/rocm/ops/ROCmRowSelectKernels.h"
#endif

#include <algorithm>
#include <cstring>
#include <utility>

namespace llaminar2
{

    MTPConcatStage::MTPConcatStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool MTPConcatStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "MTPConcatStage"))
            return false;

        if (!ensureRequiredPointers("MTPConcatStage",
                                    {{"hidden", params_.hidden},
                                     {"embedding", params_.embedding},
                                     {"output", params_.output}}))
            return false;

        auto *hidden = requireTensorBasePtr(params_.hidden, "hidden");
        auto *embedding = requireTensorBasePtr(params_.embedding, "embedding");
        auto *output = requireTensorBasePtr(params_.output, "output");
        if (!hidden || !embedding || !output)
            return false;

        const int rows = params_.num_tokens > 0
                             ? params_.num_tokens
                             : static_cast<int>(hidden->rows());
        const int hidden_dim = params_.hidden_dim > 0
                                   ? params_.hidden_dim
                                   : static_cast<int>(hidden->cols());

        if (params_.device_id.is_gpu())
        {
            auto hidden_upload = TransferEngine::instance().upload(hidden, params_.device_id);
            auto embedding_upload = TransferEngine::instance().upload(embedding, params_.device_id);
            if (!hidden_upload.success || !embedding_upload.success ||
                !output->allocateOnDevice(params_.device_id))
            {
                LOG_ERROR("[MTPConcatStage] Failed to prepare GPU tensors on "
                          << params_.device_id.toString()
                          << " hidden_upload=" << hidden_upload.error
                          << " embedding_upload=" << embedding_upload.error);
                return false;
            }

            const float *hidden_data = static_cast<const float *>(hidden->active_data_ptr());
            const float *embedding_data = static_cast<const float *>(embedding->active_data_ptr());
            float *out_data = static_cast<float *>(output->active_mutable_data_ptr());
            if (!hidden_data || !embedding_data || !out_data)
            {
                LOG_ERROR("[MTPConcatStage] Null GPU data pointer");
                return false;
            }

            void *stream = gpuStream();
            bool launched = false;

            if (params_.device_id.is_cuda())
            {
#ifdef HAVE_CUDA
                launched = cuda::launchMTPConcatFP32(
                    hidden_data,
                    embedding_data,
                    out_data,
                    rows,
                    hidden_dim,
                    stream);
#else
                LOG_ERROR("[MTPConcatStage] CUDA backend requested but HAVE_CUDA is disabled");
                return false;
#endif
            }

            if (params_.device_id.is_rocm())
            {
#ifdef HAVE_ROCM
                launched = rocm::launchMTPConcatFP32(
                    hidden_data,
                    embedding_data,
                    out_data,
                    rows,
                    hidden_dim,
                    stream);
#else
                LOG_ERROR("[MTPConcatStage] ROCm backend requested but HAVE_ROCM is disabled");
                return false;
#endif
            }

            if (!launched)
            {
                LOG_ERROR("[MTPConcatStage] GPU concat launch failed on " << params_.device_id.toString());
                return false;
            }
            output->transitionToWithEvent(TensorCoherenceState::DEVICE_AUTHORITATIVE,
                                          params_.device_id,
                                          stream);
            return true;
        }

        const float *hidden_data = hidden->data();
        const float *embedding_data = embedding->data();
        float *out_data = output->mutable_data();
        if (!hidden_data || !embedding_data || !out_data)
        {
            LOG_ERROR("[MTPConcatStage] Null FP32 data pointer");
            return false;
        }

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int row = 0; row < rows; ++row)
            {
                const size_t src_offset = static_cast<size_t>(row) * hidden_dim;
                const size_t dst_offset = static_cast<size_t>(row) * hidden_dim * 2;
                std::memcpy(out_data + dst_offset,
                            embedding_data + src_offset,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
                std::memcpy(out_data + dst_offset + hidden_dim,
                            hidden_data + src_offset,
                            static_cast<size_t>(hidden_dim) * sizeof(float));
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

    size_t MTPConcatStage::estimatedMemoryBytes() const
    {
        const size_t rows = static_cast<size_t>(std::max(params_.num_tokens, 0));
        const size_t dim = static_cast<size_t>(std::max(params_.hidden_dim, 0));
        return rows * dim * 4 * sizeof(float);
    }

    bool MTPConcatStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:
            return true;
#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
            return true;
#endif
#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
            return true;
#endif
        default:
            return false;
        }
    }

    StageDumpInfo MTPConcatStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        const size_t rows = static_cast<size_t>(std::max(params_.num_tokens, 0));
        const size_t dim = static_cast<size_t>(std::max(params_.hidden_dim, 0));
        info.addInput("embedding", params_.embedding, rows, dim);
        info.addInput("hidden", params_.hidden, rows, dim);
        info.addOutput("output", params_.output, rows, dim * 2);
        return info;
    }

    StageBufferContract MTPConcatStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.hidden_buffer_id)
            contract.addInput(*params_.hidden_buffer_id);
        if (params_.embedding_buffer_id)
            contract.addInput(*params_.embedding_buffer_id);
        if (params_.output_buffer_id)
            contract.addOutput(*params_.output_buffer_id);
        return contract;
    }

} // namespace llaminar2
