/**
 * @file ICUDANativeArrivalCaptureStage.h
 * @brief Stage declaration for CUDA arrival operations that require a native owner.
 *
 * The operation can never execute as an eager warmup. The forward engine
 * consumes this contract on a cache miss before it submits any stage work.
 */
#pragma once

#include <cstddef>

namespace llaminar2
{
    class ICUDANativeArrivalCaptureStage
    {
    public:
        virtual ~ICUDANativeArrivalCaptureStage() = default;
        /** @return Number of original arrival imports this stage records on CUDA. */
        [[nodiscard]] virtual std::size_t cudaLiveArrivalSites() const noexcept = 0;
    };
}
