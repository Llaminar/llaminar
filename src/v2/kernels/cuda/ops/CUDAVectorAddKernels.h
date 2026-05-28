/**
 * @file CUDAVectorAddKernels.h
 * @brief Host interface for CUDA vector addition kernels
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include <cstddef>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#else
using cudaStream_t = void *;
#endif

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Element-wise vector addition: output = input1 + input2
         *
         * @param output Output buffer (device memory)
         * @param input1 First input buffer (device memory)
         * @param input2 Second input buffer (device memory)
         * @param count Number of float elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAdd_f32(
            float *output,
            const float *input1,
            const float *input2,
            size_t count,
            cudaStream_t stream);

        /**
         * @brief In-place vector addition: output += input
         *
         * @param output Buffer to accumulate into (device memory)
         * @param input Input buffer to add (device memory)
         * @param count Number of float elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAddInplace_f32(
            float *output,
            const float *input,
            size_t count,
            cudaStream_t stream);

        /**
         * @brief In-place FP16 vector addition: output += input
         *
         * Uses half2 vectorized operations when count is even.
         *
         * @param output Buffer to accumulate into (device memory, __half*)
         * @param input Input buffer to add (device memory, __half*)
         * @param count Number of FP16 elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAddInplace_f16(
            void *output,
            const void *input,
            size_t count,
            cudaStream_t stream);

        /**
         * @brief In-place BF16 vector addition: output += input
         *
         * Uses bfloat162 vectorized operations when count is even.
         *
         * @param output Buffer to accumulate into (device memory, __nv_bfloat16*)
         * @param input Input buffer to add (device memory, __nv_bfloat16*)
         * @param count Number of BF16 elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAddInplace_bf16(
            void *output,
            const void *input,
            size_t count,
            cudaStream_t stream);

        /**
         * @brief In-place INT8 vector addition with saturation: output += input
         *
         * Uses saturating arithmetic to clamp results to [-128, 127].
         * Uses char4 vectorized operations when count is divisible by 4.
         *
         * @param output Buffer to accumulate into (device memory)
         * @param input Input buffer to add (device memory)
         * @param count Number of int8 elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAddInplace_i8(
            int8_t *output,
            const int8_t *input,
            size_t count,
            cudaStream_t stream);

        /**
         * @brief In-place INT32 vector addition: output += input
         *
         * Uses int4 vectorized operations when count is divisible by 4.
         *
         * @param output Buffer to accumulate into (device memory)
         * @param input Input buffer to add (device memory)
         * @param count Number of int32 elements
         * @param stream CUDA stream for kernel execution
         * @return true on success
         */
        bool launchVectorAddInplace_i32(
            int32_t *output,
            const int32_t *input,
            size_t count,
            cudaStream_t stream);

    } // namespace cuda
} // namespace llaminar2
