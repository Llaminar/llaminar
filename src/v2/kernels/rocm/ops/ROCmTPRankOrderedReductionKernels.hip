/**
 * @file ROCmTPRankOrderedReductionKernels.hip
 * @brief ROCm fixed-rank reduction for batch-invariant TP publication.
 *
 * The native allgather preceding this kernel transports participant partials
 * without changing their bits.  One HIP thread then owns each output element
 * and folds rank banks in ascending order with explicit round-to-nearest
 * additions.  The caller supplies all storage and the exact graph stream.
 */

#include "../../common/TPRankOrderedReductionKernels.h"

#include <hip/hip_runtime.h>

namespace llaminar2
{
    namespace
    {
        constexpr int kThreadsPerBlock = 256;

        /**
         * @brief Fold one rank-major value column in ascending rank order.
         */
        __global__ void rankOrderedSumFP32Kernel(
            const float *__restrict__ rank_banks,
            float *__restrict__ output,
            std::size_t element_count,
            int rank_count)
        {
            const std::size_t first =
                static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            const std::size_t stride =
                static_cast<std::size_t>(blockDim.x) * gridDim.x;
            for (std::size_t element = first;
                 element < element_count;
                 element += stride)
            {
                float sum = rank_banks[element];
                for (int rank = 1; rank < rank_count; ++rank)
                {
                    sum = __fadd_rn(
                        sum,
                        rank_banks[static_cast<std::size_t>(rank) *
                                       element_count +
                                   element]);
                }
                output[element] = sum;
            }
        }
    } // namespace

    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream)
    {
        if (!rank_banks || !output || element_count == 0 ||
            rank_count < 2 || device_index < 0 || !stream)
        {
            return false;
        }
        if (hipSetDevice(device_index) != hipSuccess)
            return false;

        const std::size_t blocks_needed =
            (element_count + kThreadsPerBlock - 1u) /
            static_cast<std::size_t>(kThreadsPerBlock);
        const unsigned int blocks = static_cast<unsigned int>(
            blocks_needed > 65535u ? 65535u : blocks_needed);
        hipLaunchKernelGGL(
            rankOrderedSumFP32Kernel,
            dim3(blocks),
            dim3(kThreadsPerBlock),
            0,
            static_cast<hipStream_t>(stream),
            rank_banks,
            output,
            element_count,
            rank_count);
        return hipPeekAtLastError() == hipSuccess;
    }
} // namespace llaminar2
