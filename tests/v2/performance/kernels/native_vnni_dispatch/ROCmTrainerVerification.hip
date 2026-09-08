/**
 * @file ROCmTrainerVerification.hip
 * @brief HIP implementation of device-resident trainer byte comparison.
 */

#include "GPUTrainerVerification.h"

#include <hip/hip_runtime.h>

#include <algorithm>

namespace llaminar2::test
{
    /** One grid-stride raw-bit comparison over an FP32 output tensor. */
    __global__ void rocmTrainerFP32ByteComparisonKernel(
        const float *__restrict__ actual,
        const float *__restrict__ expected,
        size_t count,
        unsigned long long *__restrict__ mismatch_count,
        unsigned long long *__restrict__ first_mismatch)
    {
        const size_t thread =
            static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t stride =
            static_cast<size_t>(blockDim.x) * gridDim.x;
        for (size_t index = thread; index < count; index += stride)
        {
            const uint32_t difference =
                __float_as_uint(actual[index]) ^
                __float_as_uint(expected[index]);
            if (difference == 0)
                continue;

            const unsigned int differing_bytes =
                static_cast<unsigned int>((difference & 0x000000ffu) != 0) +
                static_cast<unsigned int>((difference & 0x0000ff00u) != 0) +
                static_cast<unsigned int>((difference & 0x00ff0000u) != 0) +
                static_cast<unsigned int>((difference & 0xff000000u) != 0);
            const unsigned int first_byte_in_word =
                static_cast<unsigned int>(__ffs(difference) - 1) / 8u;
            atomicAdd(
                mismatch_count,
                static_cast<unsigned long long>(differing_bytes));
            atomicMin(
                first_mismatch,
                static_cast<unsigned long long>(index * sizeof(float) +
                                                first_byte_in_word));
        }
    }

    bool enqueueROCmFP32ByteComparison(
        const float *actual,
        const float *expected,
        size_t count,
        uint64_t *mismatch_count,
        uint64_t *first_mismatch,
        void *stream)
    {
        if (!actual || !expected || count == 0 || !mismatch_count ||
            !first_mismatch || !stream)
        {
            return false;
        }

        auto hip_stream = static_cast<hipStream_t>(stream);
        if (hipMemsetAsync(
                mismatch_count,
                0,
                sizeof(*mismatch_count),
                hip_stream) != hipSuccess ||
            hipMemsetAsync(
                first_mismatch,
                0xff,
                sizeof(*first_mismatch),
                hip_stream) != hipSuccess)
        {
            return false;
        }

        constexpr int kThreads = 256;
        const size_t required_blocks =
            (count + static_cast<size_t>(kThreads) - 1) /
            static_cast<size_t>(kThreads);
        const int blocks = static_cast<int>(
            std::min<size_t>(required_blocks, 1024));
        hipLaunchKernelGGL(
            rocmTrainerFP32ByteComparisonKernel,
            dim3(blocks),
            dim3(kThreads),
            0,
            hip_stream,
            actual,
            expected,
            count,
            reinterpret_cast<unsigned long long *>(mismatch_count),
            reinterpret_cast<unsigned long long *>(first_mismatch));
        return hipGetLastError() == hipSuccess;
    }
}
