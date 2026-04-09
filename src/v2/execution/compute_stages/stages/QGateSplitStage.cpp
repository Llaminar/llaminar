/**
 * @file QGateSplitStage.cpp
 * @brief Implementation of Q+gate deinterleave stage for Qwen 3.5 FA layers
 */

#include "QGateSplitStage.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../utils/OpenMPUtils.h"

#include <cstring>

// GPU kernel declarations
#ifdef HAVE_CUDA
extern "C" bool cudaGDN_q_gate_split(
    const float *input, float *output_q, float *output_gate,
    int seq_len, int n_heads, int head_dim,
    int device_idx, void *stream);
#endif
#ifdef HAVE_ROCM
extern "C" bool rocmGDN_q_gate_split(
    const float *input, float *output_q, float *output_gate,
    int seq_len, int n_heads, int head_dim,
    int device_idx, void *stream);
#endif

namespace llaminar2
{

    QGateSplitStage::QGateSplitStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
    }

    bool QGateSplitStage::execute(IDeviceContext *ctx)
    {
        if (!ensureContext(ctx, "QGateSplitStage"))
            return false;

        if (!ensureRequiredPointers("QGateSplitStage",
                                    {{"input", params_.input},
                                     {"output_q", params_.output_q},
                                     {"output_gate", params_.output_gate}}))
            return false;

        auto *input_base = requireTensorBasePtr(params_.input, "input");
        auto *q_base = requireTensorBasePtr(params_.output_q, "output_q");
        auto *gate_base = requireTensorBasePtr(params_.output_gate, "output_gate");

        if (!input_base || !q_base || !gate_base)
        {
            LOG_ERROR("[QGateSplitStage] Tensor cast failed");
            return false;
        }

        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(input_base->rows());
        const int n_heads = params_.n_heads;
        const int head_dim = params_.head_dim;
        const int q_dim = n_heads * head_dim;         // Output width per row
        const int input_dim = n_heads * head_dim * 2; // Input width per row

        // GPU dispatch path
        if (params_.device_id.is_gpu())
        {
            // Ensure all tensors are on device
            const_cast<TensorBase *>(input_base)->ensureOnDevice(params_.device_id);
            q_base->allocateOnDevice(params_.device_id);
            gate_base->allocateOnDevice(params_.device_id);

            const float *src_gpu = static_cast<const float *>(input_base->active_data_ptr());
            float *dst_q_gpu = static_cast<float *>(q_base->active_mutable_data_ptr());
            float *dst_gate_gpu = static_cast<float *>(gate_base->active_mutable_data_ptr());
            int dev_idx = params_.device_id.toKernelDeviceIndex();
            void *stream = gpuStream();

#ifdef HAVE_CUDA
            if (params_.device_id.is_cuda())
            {
                bool ok = cudaGDN_q_gate_split(
                    src_gpu, dst_q_gpu, dst_gate_gpu,
                    seq_len, n_heads, head_dim,
                    dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[QGateSplitStage] CUDA kernel failed");
                    return false;
                }
                return true;
            }
#endif
#ifdef HAVE_ROCM
            if (params_.device_id.is_rocm())
            {
                bool ok = rocmGDN_q_gate_split(
                    src_gpu, dst_q_gpu, dst_gate_gpu,
                    seq_len, n_heads, head_dim,
                    dev_idx, stream);
                if (!ok)
                {
                    LOG_ERROR("[QGateSplitStage] ROCm kernel failed");
                    return false;
                }
                return true;
            }
#endif
        }

        const float *src = input_base->data();
        float *dst_q = q_base->mutable_data();
        float *dst_gate = gate_base->mutable_data();

        if (!src || !dst_q || !dst_gate)
        {
            LOG_ERROR("[QGateSplitStage] Null data pointer");
            return false;
        }

        // Per-head interleaved layout: each head has [query_head_dim, gate_head_dim]
        // contiguous in the Q projection output. For n_heads=8, head_dim=256:
        //   [q0_256, g0_256, q1_256, g1_256, ..., q7_256, g7_256]
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                const float *row = src + static_cast<size_t>(t) * input_dim;
                float *q_row = dst_q + static_cast<size_t>(t) * q_dim;
                float *g_row = dst_gate + static_cast<size_t>(t) * q_dim;

                for (int h = 0; h < n_heads; ++h)
                {
                    const float *head_block = row + h * (head_dim * 2);
                    std::memcpy(q_row + h * head_dim, head_block, head_dim * sizeof(float));
                    std::memcpy(g_row + h * head_dim, head_block + head_dim, head_dim * sizeof(float));
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        LOG_DEBUG("[QGateSplitStage] Executed: seq_len=" << seq_len
                                                         << " n_heads=" << n_heads
                                                         << " head_dim=" << head_dim
                                                         << " input_dim=" << input_dim
                                                         << " q_dim=" << q_dim);
        return true;
    }

    size_t QGateSplitStage::estimatedFlops() const
    {
        return 0; // Pure memory operation, no FLOPs
    }

    size_t QGateSplitStage::estimatedMemoryBytes() const
    {
        const size_t q_dim = static_cast<size_t>(params_.n_heads) * params_.head_dim;
        const size_t total = static_cast<size_t>(params_.seq_len) * q_dim;
        // Read input (2× q_dim per row), write Q + gate (each q_dim per row)
        return total * 4 * sizeof(float);
    }

    bool QGateSplitStage::supportsBackend(ComputeBackendType backend) const
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

    StageDumpInfo QGateSplitStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        auto *q_base = dynamic_cast<const TensorBase *>(params_.output_q);
        auto *gate_base = dynamic_cast<const TensorBase *>(params_.output_gate);

        if (q_base)
            info.addOutput("output_q", params_.output_q, q_base->rows(), q_base->cols());
        if (gate_base)
            info.addOutput("output_gate", params_.output_gate, gate_base->rows(), gate_base->cols());

        return info;
    }

    StageBufferRequirements QGateSplitStage::getBufferRequirements() const
    {
        return {};
    }

    StageBufferContract QGateSplitStage::bufferContract() const
    {
        StageBufferContract contract;
        if (params_.input_buffer_id)
            contract.addInput(*params_.input_buffer_id);
        if (params_.output_q_buffer_id)
            contract.addOutput(*params_.output_q_buffer_id);
        if (params_.output_gate_buffer_id)
            contract.addOutput(*params_.output_gate_buffer_id);
        return contract;
    }

} // namespace llaminar2
