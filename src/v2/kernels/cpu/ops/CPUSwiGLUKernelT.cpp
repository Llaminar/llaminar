/**
 * @file CPUSwiGLUKernelT.cpp
 * @brief CPU SwiGLU kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPUSwiGLUKernelT.h"
#include "../primitives/SwiGLUPrimitives.h"
#include "../../../tensors/Tensors.h"
#include <cmath>
#include <omp.h>
#include <type_traits>

namespace llaminar2
{

    template <typename TensorT>
    bool CPUSwiGLUKernelT<TensorT>::apply(
        const float *gate, const float *up, float *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP32Tensor>)
        {
            (void)device_idx;   // Device index ignored - always operates on CPU buffers
            (void)add_residual; // Not implemented yet

            const int total_elements = rows * cols;

            // Use vectorized primitives
            primitives::compute_swiglu(gate, up, output, total_elements);

            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUSwiGLUKernelT<TensorT>::apply_bf16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, BF16Tensor>)
        {
            (void)device_idx;
            (void)add_residual;
            const int total_elements = rows * cols;
            primitives::compute_swiglu_bf16(gate, up, output, total_elements);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUSwiGLUKernelT<TensorT>::apply_fp16(
        const uint16_t *gate, const uint16_t *up, uint16_t *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, FP16Tensor>)
        {
            (void)device_idx;
            (void)add_residual;
            const int total_elements = rows * cols;
            primitives::compute_swiglu_fp16(gate, up, output, total_elements);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename TensorT>
    bool CPUSwiGLUKernelT<TensorT>::apply_q8_1(
        const void *gate, const void *up, void *output,
        int rows, int cols,
        bool add_residual,
        const MPIContext *mpi_ctx,
        int device_idx)
    {
        if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            (void)device_idx;
            (void)add_residual;
            const int total_elements = rows * cols;
            primitives::compute_swiglu_q8_1(gate, up, output, total_elements);
            return true;
        }
        else
        {
            return false;
        }
    }

    // Explicit instantiations
    template class CPUSwiGLUKernelT<FP32Tensor>;
    template class CPUSwiGLUKernelT<BF16Tensor>;
    template class CPUSwiGLUKernelT<FP16Tensor>;
    template class CPUSwiGLUKernelT<Q8_1Tensor>;

} // namespace llaminar2
