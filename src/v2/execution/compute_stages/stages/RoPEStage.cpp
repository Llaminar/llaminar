/**
 * @file RoPEStage.cpp
 * @brief Implementation of RoPEStage
 */

#include "RoPEStage.h"
#include "../ComputeStageUtils.h"
#include "../../../utils/DebugEnv.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/Logger.h"
#include "../../../kernels/KernelFactory.h"

namespace llaminar2
{

    // =============================================================================
    // RoPEStage Implementation (Type-Safe via KernelFactory)
    // =============================================================================

    RoPEStage::RoPEStage(Params params) : params_(std::move(params)) {}

    bool RoPEStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[RoPEStage] Null device context");
            return false;
        }

        if (!params_.Q)
        {
            LOG_ERROR("[RoPEStage] Null Q tensor");
            return false;
        }

        // Get seq_len: use explicit param if set (for pre-allocated buffers), else from tensor
        // This is critical for decode where buffer is [max_seq_len, dim] but we only process 1 token
        const int seq_len = (params_.seq_len > 0)
                                ? params_.seq_len
                                : static_cast<int>(params_.Q->rows());

        // Detect Hybrid mode: Q8_1 input with FP32 output buffers
        const bool hybrid_mode = (params_.Q_out != nullptr) &&
                                 (params_.Q->native_type() == TensorType::Q8_1) &&
                                 (params_.Q_out->native_type() == TensorType::FP32);

        // Detect HybridQ16 mode: Q8_1 input with Q16_1 output buffers
        const bool hybrid_q16_mode = (params_.Q_out != nullptr) &&
                                     (params_.Q->native_type() == TensorType::Q8_1) &&
                                     (params_.Q_out->native_type() == TensorType::Q16_1);

        LOG_DEBUG("[RoPEStage] Execute: seq_len=" << seq_len
                                                  << " n_heads=" << params_.n_heads
                                                  << " n_kv_heads=" << params_.n_kv_heads
                                                  << " head_dim=" << params_.head_dim
                                                  << " pos_offset=" << params_.pos_offset
                                                  << " tensor_type=" << params_.Q->dtype_name()
                                                  << " hybrid_mode=" << (hybrid_mode ? "true" : "false")
                                                  << " hybrid_q16_mode=" << (hybrid_q16_mode ? "true" : "false"));

        // Create kernel via KernelFactory with automatic type dispatch
        auto dev_type = llaminar::v2::kernels::KernelFactory::getDeviceType(params_.device_idx);
        auto kernel = llaminar::v2::kernels::KernelFactory::createRoPE(params_.Q, dev_type);
        if (!kernel)
        {
            LOG_ERROR("[RoPEStage] Failed to create RoPE kernel for type "
                      << params_.Q->dtype_name());
            return false;
        }

        // Generate position_ids array [pos_offset, pos_offset+1, ..., pos_offset+seq_len-1]
        std::vector<int> position_ids(seq_len);
        for (int i = 0; i < seq_len; ++i)
        {
            position_ids[i] = params_.pos_offset + i;
        }

        const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;

        // Hybrid mode: use apply_q8_1_to_fp32() for Q8_1 → FP32 with no requantization
        if (hybrid_mode)
        {
            return kernel->apply_q8_1_to_fp32(
                params_.Q,
                params_.K,
                params_.Q_out,
                params_.K_out,
                position_ids.data(),
                seq_len,
                params_.n_heads,
                n_kv_heads,
                params_.head_dim,
                params_.theta_base,
                params_.mpi_ctx,
                params_.device_idx);
        }

        // HybridQ16 mode: use apply_q8_1_to_q16_1() for Q8_1 → Q16_1 with higher precision
        if (hybrid_q16_mode)
        {
            LOG_DEBUG("[RoPEStage] Using HybridQ16 mode: Q8_1 → Q16_1");
            return kernel->apply_q8_1_to_q16_1(
                params_.Q,
                params_.K,
                params_.Q_out,
                params_.K_out,
                position_ids.data(),
                seq_len,
                params_.n_heads,
                n_kv_heads,
                params_.head_dim,
                params_.theta_base,
                params_.mpi_ctx,
                params_.device_idx);
        }

        // Standard path: Apply RoPE via kernel's apply_tensor method (in-place)
        return kernel->apply_tensor(
            params_.Q,
            params_.K, // May be nullptr
            position_ids.data(),
            seq_len,
            params_.n_heads,
            n_kv_heads,
            params_.head_dim,
            params_.theta_base,
            params_.mpi_ctx,
            params_.device_idx);
    }

    size_t RoPEStage::estimatedFlops() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        // Per position per head: head_dim/2 rotations, each ~10 FLOPs (sin, cos, 4 muls, 2 adds)
        size_t flops = static_cast<size_t>(10) * seq_len * params_.n_heads * (params_.head_dim / 2);
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            flops += static_cast<size_t>(10) * seq_len * n_kv_heads * (params_.head_dim / 2);
        }
        return flops;
    }

    size_t RoPEStage::estimatedMemoryBytes() const
    {
        if (!params_.Q)
            return 0;

        const int seq_len = static_cast<int>(params_.Q->rows());
        size_t bytes = static_cast<size_t>(2) * seq_len * params_.n_heads *
                       params_.head_dim * sizeof(float); // Q read + write
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            bytes += static_cast<size_t>(2) * seq_len * n_kv_heads * params_.head_dim * sizeof(float);
        }
        return bytes;
    }

    bool RoPEStage::supportsBackend(ComputeBackendType backend) const
    {
        switch (backend)
        {
        case ComputeBackendType::CPU:

            return true;
        default:
            return false;
        }
    }

    StageDumpInfo RoPEStage::getDumpInfo() const
    {
        StageDumpInfo info;
        if (!params_.Q)
            return info;

        // Use explicit seq_len if provided, otherwise derive from tensor
        const int seq_len = (params_.seq_len > 0) ? params_.seq_len : static_cast<int>(params_.Q->rows());

        // Detect Hybrid mode: Q8_1 input with FP32 output buffers
        const bool hybrid_mode = (params_.Q_out != nullptr) &&
                                 (params_.Q->native_type() == TensorType::Q8_1) &&
                                 (params_.Q_out->native_type() == TensorType::FP32);

        // Q input
        const float *q_input_data = getSafeFp32Data(params_.Q);
        if (q_input_data)
        {
            info.addInput("Q", q_input_data,
                          seq_len, params_.n_heads * params_.head_dim);
        }

        // Q output - use Q_out in Hybrid mode, otherwise same as input (in-place)
        if (hybrid_mode && params_.Q_out)
        {
            const float *q_out_data = getSafeFp32Data(params_.Q_out);
            if (q_out_data)
            {
                info.addOutput("Q", q_out_data,
                               seq_len, params_.n_heads * params_.head_dim);
            }
        }
        else if (q_input_data)
        {
            info.addOutput("Q", q_input_data,
                           seq_len, params_.n_heads * params_.head_dim);
        }

        // K tensor (optional)
        if (params_.K)
        {
            int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;

            // K input
            const float *k_input_data = getSafeFp32Data(params_.K);
            if (k_input_data)
            {
                info.addInput("K", k_input_data,
                              seq_len, n_kv_heads * params_.head_dim);
            }

            // K output - use K_out in Hybrid mode, otherwise same as input (in-place)
            if (hybrid_mode && params_.K_out)
            {
                const float *k_out_data = getSafeFp32Data(params_.K_out);
                if (k_out_data)
                {
                    info.addOutput("K", k_out_data,
                                   seq_len, n_kv_heads * params_.head_dim);
                }
            }
            else if (k_input_data)
            {
                info.addOutput("K", k_input_data,
                               seq_len, n_kv_heads * params_.head_dim);
            }
        }

        // Scalar params
        info.addScalarInt("seq_len", seq_len);
        info.addScalarInt("n_heads", params_.n_heads);
        info.addScalarInt("n_kv_heads", params_.n_kv_heads);
        info.addScalarInt("head_dim", params_.head_dim);
        info.addScalarInt("pos_offset", params_.pos_offset);
        info.addScalar("theta_base", params_.theta_base);

        return info;
    }

    StageBufferRequirements RoPEStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (!params_.Q)
            return reqs; // Empty if tensors not set

        // Get dimensions from tensors
        const size_t seq_len = params_.Q->rows();
        const size_t q_dim = static_cast<size_t>(params_.n_heads * params_.head_dim);

        // Convert tensor type to buffer tensor type
        BufferTensorType buf_type = toBufferTensorType(params_.Q->native_type());

        // Q is INOUT (in-place operation)
        reqs.addInout("Q", {seq_len, q_dim}, buf_type);

        // K is optional INOUT (in-place operation)
        if (params_.K)
        {
            const int n_kv_heads = params_.n_kv_heads > 0 ? params_.n_kv_heads : params_.n_heads;
            const size_t k_dim = static_cast<size_t>(n_kv_heads * params_.head_dim);
            reqs.addInout("K", {seq_len, k_dim}, buf_type);
        }

        return reqs;
    }

} // namespace llaminar2
