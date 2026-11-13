/**
 * @file CountedChannelTestKernels.hip
 * @brief HIP fixture bridge for GPU-authored counted-transfer replay inputs.
 *
 * HIP records the same fixed graph as CUDA; no conditional graph support,
 * scheduler tickets, host count reads or per-size recapture is required.
 */
#include <hip/hip_runtime.h>
#include "CountedChannelTestKernels.h"
#include "CountedChannelTestKernels.inl"

namespace llaminar2::counted_channel_test
{
    bool prepareKernelsROCm()
    {
        hipFuncAttributes attributes{};
        return hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(prepareCountedPayload)) == hipSuccess &&
            hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(advanceCountedSequence)) == hipSuccess &&
            hipFuncGetAttributes(&attributes, reinterpret_cast<const void *>(prepareSlicedPayload)) == hipSuccess;
    }
    bool prepareROCm(void *payload, std::uint64_t *count, const std::uint64_t *sequence, std::size_t capacity, void *stream)
    {
        if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread || !payload || !count || !sequence || capacity < 257)
            return false;
        hipLaunchKernelGGL(prepareCountedPayload, dim3(32), dim3(128), 0, static_cast<hipStream_t>(stream),
            static_cast<std::uint8_t *>(payload), count, sequence, capacity);
        return hipGetLastError() == hipSuccess;
    }
    bool prepareSlicedROCm(void *payload, CapturedTransferSourceSlice *slice,
        const std::uint64_t *sequence, std::size_t capacity, std::size_t source_span,
        std::uint64_t source_origin, void *stream)
    {
        if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread || !payload || !slice ||
            !sequence || capacity < 257 || source_span < capacity || source_origin > ~std::uint64_t{0} - source_span) return false;
        hipLaunchKernelGGL(prepareSlicedPayload, dim3(32), dim3(128), 0, static_cast<hipStream_t>(stream),
            static_cast<std::uint8_t *>(payload), slice, sequence, capacity, source_span, source_origin);
        return hipGetLastError() == hipSuccess;
    }
    bool advanceROCm(std::uint64_t *sequence, void *stream)
    {
        if (!stream || stream == hipStreamLegacy || stream == hipStreamPerThread || !sequence) return false;
        hipLaunchKernelGGL(advanceCountedSequence, dim3(1), dim3(64), 0, static_cast<hipStream_t>(stream), sequence);
        return hipGetLastError() == hipSuccess;
    }
}
