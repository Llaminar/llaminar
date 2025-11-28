/**
 * @file CPURMSNormKernelT.cpp
 * @brief CPU RMSNorm kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPURMSNormKernelT.h"
#include "../primitives/RMSNormPrimitives.h"
#include "../../../tensors/Tensors.h"
#include <cmath>
#include <type_traits>

namespace llaminar2
{

    template <typename TensorT>
    bool CPURMSNormKernelT<TensorT>::apply(
        const float *input, const float *weight, float *output,
        int rows, int cols,
        float epsilon,
        bool use_bf16,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if (!input || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, FP32Tensor> || std::is_same_v<TensorT, Q8_1Tensor>)
        {
            (void)device_idx;
            (void)use_bf16; // Should be false for FP32Tensor/Q8_1Tensor

            // Use vectorized primitives
            primitives::rmsnorm_fused_vectorized(input, weight, output, rows, cols, epsilon);

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPURMSNormKernelT<TensorT>::apply_bf16(
        const uint16_t *input, const float *weight, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx)
    {
        if (!input || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            (void)device_idx;
            primitives::rmsnorm_fused_bf16_vectorized(input, weight, output, rows, cols, epsilon);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPURMSNormKernelT<TensorT>::apply_fp16(
        const uint16_t *input, const float *weight, uint16_t *output,
        int rows, int cols, float epsilon, int device_idx)
    {
        if (!input || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            (void)device_idx;
            primitives::rmsnorm_fused_fp16_vectorized(input, weight, output, rows, cols, epsilon);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPURMSNormKernelT<TensorT>::apply_int32_to_int8(
        const int32_t *input, const float *weight, int8_t *output,
        float *scales, int rows, int cols, float epsilon, int device_idx)
    {
        if (!input || !output || !scales || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, INT32Tensor>)
        {
            (void)device_idx;
            primitives::rmsnorm_fused_int32_to_int8_vectorized(input, weight, output, scales, rows, cols, epsilon);
            return true;
        }
        else
        {
            return false;
        }
    }

    // Explicit instantiations
    template class CPURMSNormKernelT<FP32Tensor>;
    template class CPURMSNormKernelT<BF16Tensor>;
    template class CPURMSNormKernelT<FP16Tensor>;
    template class CPURMSNormKernelT<INT32Tensor>;
    template class CPURMSNormKernelT<Q8_1Tensor>;

} // namespace llaminar2
