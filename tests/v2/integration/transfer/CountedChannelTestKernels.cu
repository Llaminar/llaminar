/**
 * @file CountedChannelTestKernels.cu
 * @brief CUDA fixture bridge for GPU-authored counted-transfer replay inputs.
 *
 * No production entrypoint or transport policy lives here. Explicit stream
 * launches generate fixture bytes and counts inside the retained test graph.
 */
#include <cuda_runtime.h>
#include "CountedChannelTestKernels.h"
#include "CountedChannelTestKernels.inl"

namespace llaminar2::counted_channel_test
{
    bool prepareKernelsCUDA()
    {
        cudaFuncAttributes attributes{};
        return cudaFuncGetAttributes(&attributes, prepareCountedPayload) == cudaSuccess &&
            cudaFuncGetAttributes(&attributes, advanceCountedSequence) == cudaSuccess &&
            cudaFuncGetAttributes(&attributes, prepareSlicedPayload) == cudaSuccess;
    }
    bool prepareCUDA(void *payload, std::uint64_t *count, const std::uint64_t *sequence, std::size_t capacity, void *stream)
    {
        if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread || !payload || !count || !sequence || capacity < 257)
            return false;
        prepareCountedPayload<<<32, 128, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<std::uint8_t *>(payload), count, sequence, capacity);
        return cudaGetLastError() == cudaSuccess;
    }
    bool prepareSlicedCUDA(void *payload, CapturedTransferSourceSlice *slice,
        const std::uint64_t *sequence, std::size_t capacity, std::size_t source_span,
        std::uint64_t source_origin, void *stream)
    {
        if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread || !payload || !slice ||
            !sequence || capacity < 257 || source_span < capacity || source_origin > ~std::uint64_t{0} - source_span) return false;
        prepareSlicedPayload<<<32, 128, 0, static_cast<cudaStream_t>(stream)>>>(
            static_cast<std::uint8_t *>(payload), slice, sequence, capacity, source_span, source_origin);
        return cudaGetLastError() == cudaSuccess;
    }
    bool advanceCUDA(std::uint64_t *sequence, void *stream)
    {
        if (!stream || stream == cudaStreamLegacy || stream == cudaStreamPerThread || !sequence) return false;
        advanceCountedSequence<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(sequence);
        return cudaGetLastError() == cudaSuccess;
    }
}
