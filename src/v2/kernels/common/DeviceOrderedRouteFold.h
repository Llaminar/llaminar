/**
 * @file DeviceOrderedRouteFold.h
 * @brief Overlap independent route loads without changing FP32 reduction order.
 *
 * Canonical MoE contributions are route-major and already weighted. Each output
 * lane owns its entire router-order sum. Issuing one load followed immediately
 * by its dependent add exposes memory latency eight times for a top-8 router;
 * a small register window lets independent loads progress together instead.
 * The additions still execute in ascending route order, including the initial
 * positive zero and the last partial window. There is no parallel reduction,
 * format conversion, workspace, synchronization, or allocation here.
 */
#pragma once

#include "kernels/common/DeviceFP32NumericalContract.h"
#include <cstddef>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ROUTE_FOLD_INLINE __device__ __forceinline__
#else
#define LLAMINAR_ROUTE_FOLD_INLINE inline
#endif

namespace llaminar2::device_ordered_route_fold
{
    /** Independent loads retained per lane; this is not the model's top-k. */
    inline constexpr int kRegisterWindow = 8;

    /**
     * @brief Sum one output's contributions with bounded memory-level parallelism.
     *
     * The caller validates its row/column and owns the complete positive route
     * extent. An eight-value window is small enough to remain register-resident
     * on supported GPUs; the mandatory compiler spill guard checks this after
     * inlining. Loading ahead does not authorize reassociation: every add uses
     * the same explicit round-to-nearest contract as the serial MoE reduction.
     *
     * @param first Contribution for route zero at the caller's output column.
     * @param route_stride Distance in FP32 elements to the next route.
     * @param route_count Positive number of live canonical routes.
     * @return Serial router-order sum, byte-identical to one load/add per route.
     */
    LLAMINAR_ROUTE_FOLD_INLINE float sum(
        const float *first, std::size_t route_stride, int route_count) noexcept
    {
        float result = 0.0f;
        int route = 0;
        for (; route <= route_count - kRegisterWindow; route += kRegisterWindow)
        {
            float values[kRegisterWindow];
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
            for (int index = 0; index < kRegisterWindow; ++index)
                values[index] = first[static_cast<std::size_t>(route + index) * route_stride];
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll
#endif
            for (int index = 0; index < kRegisterWindow; ++index)
                result = device_fp32_contract::add(result, values[index]);
        }

        // Complete the same arithmetic for an incomplete load window. Never
        // read beyond the live route extent or manufacture padded contributions.
#if defined(__CUDACC__) || defined(__HIPCC__)
#pragma unroll 1
#endif
        for (; route < route_count; ++route)
            result = device_fp32_contract::add(
                result, first[static_cast<std::size_t>(route) * route_stride]);
        return result;
    }
}

#undef LLAMINAR_ROUTE_FOLD_INLINE
