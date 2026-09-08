/**
 * @file GPUTrainerVerification.h
 * @brief Device-resident byte certificates shared by GPU tuning harnesses.
 *
 * CUDA and ROCm candidate tournaments can produce very large output tensors.
 * Downloading those tensors for correctness checks wastes PCIe bandwidth and
 * makes the trainer's feedback loop scale with output size. These helpers keep
 * the comparison on the exact producer stream and materialize only two
 * terminal counters after the complete output has been examined.
 *
 * This API is diagnostic-only. It is linked into performance and integration
 * binaries, never into a production dispatch path.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace llaminar2::test
{
    /**
     * @brief Enqueue an exact FP32 comparison on a non-null CUDA stream.
     *
     * The caller owns persistent device counters. The helper resets both
     * counters asynchronously, compares raw IEEE-754 bits, and returns without
     * synchronizing. Signed zero and NaN payloads therefore remain part of the
     * certificate rather than being normalized away.
     *
     * @param actual Device output produced by the candidate graph or launch.
     * @param expected Device output retained from the certified oracle launch.
     * @param count Number of FP32 elements to compare.
     * @param mismatch_count Device scalar receiving total mismatched bytes.
     * @param first_mismatch Device scalar receiving the least byte offset,
     *        or `UINT64_MAX` when every output bit is equal.
     * @param stream Exact CUDA producer/diagnostic stream.
     * @return true when resets and the comparison launch were enqueued.
     */
    bool enqueueCudaFP32ByteComparison(
        const float *actual,
        const float *expected,
        size_t count,
        uint64_t *mismatch_count,
        uint64_t *first_mismatch,
        void *stream);

    /**
     * @brief Enqueue an exact FP32 comparison on a non-null HIP stream.
     *
     * Ownership and ordering match `enqueueCudaFP32ByteComparison`: the
     * producer stream orders the candidate output before this comparison, and
     * the caller waits only when it needs the two terminal diagnostic values.
     *
     * @param actual Device output produced by the candidate graph or launch.
     * @param expected Device output retained from the certified oracle launch.
     * @param count Number of FP32 elements to compare.
     * @param mismatch_count Device scalar receiving total mismatched bytes.
     * @param first_mismatch Device scalar receiving the least byte offset,
     *        or `UINT64_MAX` when every output bit is equal.
     * @param stream Exact HIP producer/diagnostic stream.
     * @return true when resets and the comparison launch were enqueued.
     */
    bool enqueueROCmFP32ByteComparison(
        const float *actual,
        const float *expected,
        size_t count,
        uint64_t *mismatch_count,
        uint64_t *first_mismatch,
        void *stream);
}
