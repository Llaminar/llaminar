/**
 * @file TPRankOrderedReductionKernels.h
 * @brief Device launch ABI for fixed-rank tensor-parallel reductions.
 *
 * Native NCCL/RCCL all-reduce is free to choose a different reduction tree
 * when the message contains one decode row versus several grouped verifier
 * rows.  The launchers declared here perform only the arithmetic half of the
 * batch-invariant protocol: an earlier raw allgather publishes immutable
 * rank-major FP32 banks, then one device kernel folds those banks in ascending
 * participant order.  Transport therefore moves bytes without arithmetic and
 * the visible sum has one stable order for every runtime row count.
 */

#pragma once

#include <cstddef>

namespace llaminar2
{
    /**
     * @brief Fold rank-major FP32 banks on a CUDA device.
     *
     * @param rank_banks Device input shaped `[rank_count, element_count]`.
     * @param output Device destination with `element_count` FP32 values.
     * @param element_count Number of values contributed by every participant.
     * @param rank_count Number of participant banks to fold in ascending order.
     * @param device_index CUDA ordinal that owns both allocations.
     * @param stream Exact non-default CUDA stream ordered after the allgather.
     * @return true when the kernel was enqueued successfully.
     */
    bool launchCUDATPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream);

    /**
     * @brief Fold rank-major FP32 banks on a ROCm device.
     *
     * @param rank_banks Device input shaped `[rank_count, element_count]`.
     * @param output Device destination with `element_count` FP32 values.
     * @param element_count Number of values contributed by every participant.
     * @param rank_count Number of participant banks to fold in ascending order.
     * @param device_index HIP ordinal that owns both allocations.
     * @param stream Exact non-default HIP stream ordered after the allgather.
     * @return true when the kernel was enqueued successfully.
     */
    bool launchROCmTPRankOrderedSumFP32(
        const float *rank_banks,
        float *output,
        std::size_t element_count,
        int rank_count,
        int device_index,
        void *stream);
} // namespace llaminar2
