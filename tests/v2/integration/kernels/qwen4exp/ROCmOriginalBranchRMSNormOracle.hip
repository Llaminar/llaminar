/**
 * @file ROCmOriginalBranchRMSNormOracle.hip
 * @brief Independent test-only frozen original QFN branch RMSNorm arithmetic.
 *
 * The original 256-lane FMA loop, eight LDS addition stages and all-thread
 * normalization epilogue preserve the full shared-memory reduction tree.
 * No production GPU helper is included. Qualification first calibrates this
 * oracle against the original production dispatch before testing a candidate.
 * This TU is linked only into explicit native correctness/performance targets.
 */
#include <hip/hip_runtime.h>
#include "kernels/qwen4exp/Qwen4ExpKernels.h"
#include <cmath>
#include <climits>
#include <cstddef>

namespace qfn_branch_norm_original
{
using llaminar2::qwen4exp::BranchRMSNormArgs;
using llaminar2::qwen4exp::RowExtent;
inline constexpr unsigned kThreads=256;
/** @brief Frozen original device-owned extent validation, without a host shadow. */
__device__ inline int liveRows(RowExtent extent)
{
    const int count = extent.liveRows ? *extent.liveRows : extent.rows;
    if (count < 0 || count > extent.rows) __builtin_trap();
    return count;
}
/** @brief Frozen original expression order, owning lanes and full-LDS binary tree. */
__global__ __launch_bounds__(kThreads, 2) void originalBranchRMSNorm(BranchRMSNormArgs a)
{
    const auto task = static_cast<std::size_t>(blockIdx.x);
    if (task / 4 >= static_cast<std::size_t>(liveRows(a.extent))) return;
    const auto base = task * a.hidden;
    const unsigned lane = threadIdx.x;
    float partial = 0.0f;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
    {
        const float value = a.input[base + channel];
        partial = fmaf(value, value, partial);
    }
    __shared__ float sums[kThreads];
    sums[lane] = partial;
    __syncthreads();
#pragma unroll
    for (unsigned stride = kThreads / 2; stride > 0; stride /= 2)
    {
        if (lane < stride) sums[lane] = __fadd_rn(sums[lane], sums[lane + stride]);
        __syncthreads();
    }
    const float inverse = 1.0f / sqrtf(sums[0] / static_cast<float>(a.hidden) + a.epsilon);
    const auto gainBase = (task % 4) * a.hidden;
    for (unsigned channel = lane; channel < static_cast<unsigned>(a.hidden); channel += kThreads)
        a.output[base + channel] = __fmul_rn(__fmul_rn(a.input[base + channel], inverse), a.gain[gainBase + channel]);
}
}
/** @brief Test-only original launch with unchanged grid/block and exact caller stream. */
extern "C" bool testOriginalBranchRMSNorm(llaminar2::qwen4exp::BranchRMSNormArgs args,void *stream)
{
    auto native=static_cast<hipStream_t>(stream);
    if (!native || native==hipStreamLegacy || native==hipStreamPerThread ||
        args.extent.rows<=0 || args.extent.rows>INT_MAX/4 || args.hidden<=0 || args.hidden>INT_MAX/4 ||
        !args.input || !args.gain || !args.output || args.output==args.gain || !std::isfinite(args.epsilon) || args.epsilon<=0)
        return false;
    hipLaunchKernelGGL(qfn_branch_norm_original::originalBranchRMSNorm,
        dim3(static_cast<unsigned>(args.extent.rows)*4),dim3(256),0,native,args);
    return hipPeekAtLastError()==hipSuccess;
}
