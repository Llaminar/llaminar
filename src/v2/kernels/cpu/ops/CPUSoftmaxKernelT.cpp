/**
 * @file CPUSoftmaxKernelT.cpp
 * @brief Implementation of templated CPU Softmax kernel
 *
 * @author David Sanftenberg
 */

#include "CPUSoftmaxKernelT.h"
#include "../../../tensors/Tensors.h"
#include <omp.h>
#include <cstring>

namespace llaminar2
{

    template <typename TensorT>
    bool CPUSoftmaxKernelT<TensorT>::apply(
        const float *input, float *output,
        int rows, int cols,
        bool use_causal_mask,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        // If input != output, copy first (primitives are in-place)
        if (input != output)
        {
            std::memcpy(output, input, rows * cols * sizeof(float));
        }

// Parallelize over rows
#pragma omp parallel for if (rows * cols > 8192)
        for (int i = 0; i < rows; ++i)
        {
            float *row_ptr = output + i * cols;
            primitives::softmax_row_fp32(row_ptr, cols, use_causal_mask, 1.0f, i);
        }

        return true;
    }

    template <typename TensorT>
    bool CPUSoftmaxKernelT<TensorT>::apply_bf16(
        const uint16_t *input, uint16_t *output,
        int rows, int cols,
        bool use_causal_mask,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        if (input != output)
        {
            std::memcpy(output, input, rows * cols * sizeof(uint16_t));
        }

#pragma omp parallel for if (rows * cols > 8192)
        for (int i = 0; i < rows; ++i)
        {
            uint16_t *row_ptr = output + i * cols;
            primitives::softmax_row_bf16(row_ptr, cols, use_causal_mask, 1.0f, i);
        }

        return true;
    }

    template <typename TensorT>
    bool CPUSoftmaxKernelT<TensorT>::apply_fp16(
        const uint16_t *input, uint16_t *output,
        int rows, int cols,
        bool use_causal_mask,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        (void)mpi_ctx;
        (void)device_idx;

        if (input != output)
        {
            std::memcpy(output, input, rows * cols * sizeof(uint16_t));
        }

#pragma omp parallel for if (rows * cols > 8192)
        for (int i = 0; i < rows; ++i)
        {
            uint16_t *row_ptr = output + i * cols;
            primitives::softmax_row_fp16(row_ptr, cols, use_causal_mask, 1.0f, i);
        }

        return true;
    }

    // Explicit instantiations
    template class CPUSoftmaxKernelT<FP32Tensor>;
    template class CPUSoftmaxKernelT<BF16Tensor>;
    template class CPUSoftmaxKernelT<FP16Tensor>;
    template class CPUSoftmaxKernelT<Q8_1Tensor>;
    template class CPUSoftmaxKernelT<INT32Tensor>;

} // namespace llaminar2
