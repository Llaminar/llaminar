/**
 * @file CPURMSNormKernelT.cpp
 * @brief CPU RMSNorm kernel implementation
 *
 * @author David Sanftenberg
 */

#include "CPURMSNormKernelT.h"
#include "../primitives/RMSNormPrimitives.h"
#include "../../../tensors/SIMDHelpers.h" // For dequantize/quantize Q8_1
#include "../../../tensors/Tensors.h"
#include <cmath>
#include <type_traits>
#include <array>

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

    template <typename TensorT>
    bool CPURMSNormKernelT<TensorT>::apply_q8_1(
        const Q8_1Block *input, const float *weight, Q8_1Block *output,
        int rows, int cols, float epsilon, int device_idx)
    {
        if (!input || !weight || !output || rows <= 0 || cols <= 0)
        {
            return false;
        }

        // Q8_1 requires cols to be multiple of 32 (block size)
        if (cols % 32 != 0)
        {
            return false;
        }

        if constexpr (std::is_same_v<TensorT, Q8_1Tensor>)
        {
            (void)device_idx;

            const size_t ucols = static_cast<size_t>(cols);
            const size_t blocks_per_row = ucols / 32;

            // Use optimized SIMD path: dequant → RMSNorm → quant
            // This is faster than the integer-space approach (see changelog/2025-12-04)
            static constexpr size_t MAX_STACK_ROW_SIZE = 8192;

            // Lambda for processing a single row
            auto process_row = [&](int row, float *fp32_in, float *fp32_out)
            {
                const Q8_1Block *in_row = input + row * blocks_per_row;
                Q8_1Block *out_row = output + row * blocks_per_row;

                // 1. Dequantize Q8_1 → FP32 (one row)
                simd::dequantize_q8_1_to_fp32(in_row, fp32_in, ucols);

                // 2. RMSNorm in FP32 (one row)
                primitives::rmsnorm_fused_row_avx512(fp32_in, weight, fp32_out, ucols, epsilon);

                // 3. Quantize FP32 → Q8_1 (one row)
                simd::quantize_fp32_to_q8_1_blocks(fp32_out, out_row, ucols);
            };

            // Parallelization threshold (from want_parallel in typed kernel)
            const bool use_parallel = (rows >= 8) && (static_cast<size_t>(rows) * ucols >= 65536);

            if (use_parallel)
            {
#ifdef _OPENMP
#pragma omp parallel
#endif
                {
                    std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
                    std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                    std::vector<float> heap_fp32_in, heap_fp32_out;
                    float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                         ? stack_fp32_in.data()
                                         : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
                    float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                          ? stack_fp32_out.data()
                                          : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

#ifdef _OPENMP
#pragma omp for
#endif
                    for (int row = 0; row < rows; ++row)
                    {
                        process_row(row, fp32_in, fp32_out);
                    }
                }
            }
            else
            {
                // Serial path for small workloads
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_in;
                std::array<float, MAX_STACK_ROW_SIZE> stack_fp32_out;
                std::vector<float> heap_fp32_in, heap_fp32_out;
                float *fp32_in = (ucols <= MAX_STACK_ROW_SIZE)
                                     ? stack_fp32_in.data()
                                     : (heap_fp32_in.resize(ucols), heap_fp32_in.data());
                float *fp32_out = (ucols <= MAX_STACK_ROW_SIZE)
                                      ? stack_fp32_out.data()
                                      : (heap_fp32_out.resize(ucols), heap_fp32_out.data());

                for (int row = 0; row < rows; ++row)
                {
                    process_row(row, fp32_in, fp32_out);
                }
            }

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
