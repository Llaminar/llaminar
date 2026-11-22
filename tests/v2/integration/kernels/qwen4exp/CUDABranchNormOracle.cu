/**
 * @file CUDABranchNormOracle.cu
 * @brief Freeze the original CUDA normalization tree and staged Q8 publication.
 *
 * These launches are linked into the native test owner only. The fixture
 * records original bytes before replay and supplies an explicit native stream.
 */
#include <cuda_runtime.h>
#include <cmath>

#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/qwen4exp/Qwen4ExpKernels.h"

namespace
{
    using llaminar2::qwen4exp::BranchRMSNormArgs;
    using llaminar2::qwen4exp::BranchRMSNormSingleQ8Args;
    constexpr unsigned threads = 256;

    /** @brief Original branch norm with all eight reduction joins in shared memory. */
    __global__ __launch_bounds__(threads, 2) void originalBranchNorm(BranchRMSNormArgs args)
    {
        const auto task = static_cast<std::size_t>(blockIdx.x);
        const auto live = args.extent.liveRows ? *args.extent.liveRows : args.extent.rows;
        if (task / 4 >= static_cast<std::size_t>(live))
            return;

        const auto base = task * args.hidden;
        const auto gainBase = task % 4 * args.hidden;
        const unsigned lane = threadIdx.x;
        float partial = 0.0f;
        for (unsigned channel = lane; channel < static_cast<unsigned>(args.hidden); channel += threads)
        {
            const float value = args.input[base + channel];
            partial = fmaf(value, value, partial);
        }
        __shared__ float sums[threads];
        sums[lane] = partial;
        __syncthreads();
#pragma unroll
        for (unsigned stride = threads / 2; stride > 0; stride /= 2)
        {
            if (lane < stride)
                sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
            __syncthreads();
        }
        const float inverse = 1.0f / sqrtf(sums[0] / static_cast<float>(args.hidden) + args.epsilon);
        for (unsigned channel = lane; channel < static_cast<unsigned>(args.hidden); channel += threads)
        {
            const auto value = __fmul_rn(args.input[base + channel], inverse);
            args.output[base + channel] = __fmul_rn(value, args.gain[gainBase + channel]);
        }
    }

    /** @brief Original Q8 arithmetic over stored FP32 values, after the norm producer retires. */
    __global__ void originalStoredQ8(BranchRMSNormSingleQ8Args args)
    {
        const auto block = blockIdx.x;
        const auto offset = block * 32;
        const auto lane = threadIdx.x;
        llaminar2::original_single_q8::publish(args.norm.output[offset + lane], true,
            args.codes + offset, args.scales + block, nullptr, lane);
    }
}

/** @brief Record the original norm and optional separate exact K32 oracle on the test stream. */
extern "C" bool testOriginalCUDABranchNorm(BranchRMSNormSingleQ8Args args, void* stream)
{
    const auto native = static_cast<cudaStream_t>(stream);
    if (!native || native == cudaStreamLegacy || native == cudaStreamPerThread)
        return false;
    originalBranchNorm<<<static_cast<unsigned>(args.norm.extent.rows) * 4, threads, 0, native>>>(args.norm);
    if (cudaGetLastError() != cudaSuccess)
        return false;
    if (args.codes)
    {
        originalStoredQ8<<<4 * args.norm.hidden / 32, 32, 0, native>>>(args);
        if (cudaGetLastError() != cudaSuccess)
            return false;
    }
    return true;
}
