/**
 * @file HIPGraphTimelineKernels.hip
 * @brief System-scope HIP graph waits/publications for mapped epoch timelines.
 *
 * The primitive is intentionally topology-agnostic: its address is resolved by
 * TransferEngine from planner-owned mapped-region identity, and the graph
 * dependency carries all producer ordering. It performs no host callback,
 * allocation, synchronization, or replay-time parameter mutation.
 */

#include "HIPGraphTimelineKernels.h"

#ifdef HAVE_ROCM

namespace llaminar2::hip_graph_timeline
{
    namespace
    {
        /**
         * @brief Wait until a peer publishes one leased timeline threshold.
         *
         * Only one wave is resident. The short scalar sleep avoids continuously
         * saturating the node-local interconnect while the peer device works;
         * the final system-acquire load makes preceding packet writes visible
         * to dependent kernels.
         */
        __global__ void waitSystemValue64(
            const std::uint64_t *signal,
            std::uint64_t value)
        {
            if (blockIdx.x != 0u || threadIdx.x != 0u)
                return;
            while (__hip_atomic_load(
                       signal,
                       __ATOMIC_ACQUIRE,
                       __HIP_MEMORY_SCOPE_SYSTEM) < value)
            {
                __builtin_amdgcn_s_sleep(1u);
            }
        }

        /**
         * @brief Publish one timeline value with system-wide release ordering.
         *
         * The graph dependency makes every producer kernel complete first. A
         * system-scope release then makes mapped packet/control bytes visible
         * before a peer device can observe the timeline value and continue.
         */
        __global__ void publishSystemReleaseValue64(
            std::uint64_t *signal,
            std::uint64_t value)
        {
            if (blockIdx.x == 0u && threadIdx.x == 0u)
            {
                __hip_atomic_store(
                    signal,
                    value,
                    __ATOMIC_RELEASE,
                    __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
    } // namespace

    hipError_t addSystemWaitValue64Node(
        hipGraphNode_t *node,
        hipGraph_t graph,
        const hipGraphNode_t *dependencies,
        std::size_t dependency_count,
        const void *signal,
        std::uint64_t value) noexcept
    {
        if (!node || !graph || !signal || value == 0u ||
            (dependency_count > 0u && !dependencies) ||
            (reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            return hipErrorInvalidValue;
        }

        auto *signal_argument = static_cast<const std::uint64_t *>(signal);
        std::uint64_t value_argument = value;
        void *arguments[] = {&signal_argument, &value_argument};
        hipKernelNodeParams params{};
        params.func = reinterpret_cast<void *>(waitSystemValue64);
        params.gridDim = dim3(1u, 1u, 1u);
        params.blockDim = dim3(1u, 1u, 1u);
        params.sharedMemBytes = 0u;
        params.kernelParams = arguments;
        params.extra = nullptr;
        return hipGraphAddKernelNode(
            node,
            graph,
            dependencies,
            dependency_count,
            &params);
    }

    hipError_t addSystemReleaseValue64Node(
        hipGraphNode_t *node,
        hipGraph_t graph,
        const hipGraphNode_t *dependencies,
        std::size_t dependency_count,
        void *signal,
        std::uint64_t value) noexcept
    {
        if (!node || !graph || !signal || value == 0u ||
            (dependency_count > 0u && !dependencies) ||
            (reinterpret_cast<std::uintptr_t>(signal) &
             (alignof(std::uint64_t) - 1u)) != 0u)
        {
            return hipErrorInvalidValue;
        }

        auto *signal_argument = static_cast<std::uint64_t *>(signal);
        std::uint64_t value_argument = value;
        void *arguments[] = {&signal_argument, &value_argument};
        hipKernelNodeParams params{};
        params.func = reinterpret_cast<void *>(publishSystemReleaseValue64);
        params.gridDim = dim3(1u, 1u, 1u);
        params.blockDim = dim3(1u, 1u, 1u);
        params.sharedMemBytes = 0u;
        params.kernelParams = arguments;
        params.extra = nullptr;
        return hipGraphAddKernelNode(
            node,
            graph,
            dependencies,
            dependency_count,
            &params);
    }

    bool appendActiveCaptureSystemWaitValue64(
        hipStream_t stream,
        const void *signal,
        std::uint64_t value) noexcept
    {
        if (!stream || !signal || value == 0u)
            return false;
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        unsigned long long capture_id = 0u;
        hipGraph_t graph = nullptr;
        const hipGraphNode_t *dependencies = nullptr;
        std::size_t dependency_count = 0u;
        if (hipStreamGetCaptureInfo_v2(
                stream,
                &capture_status,
                &capture_id,
                &graph,
                &dependencies,
                &dependency_count) != hipSuccess ||
            capture_status != hipStreamCaptureStatusActive || !graph ||
            capture_id == 0u ||
            (dependency_count != 0u && !dependencies))
        {
            return false;
        }
        hipGraphNode_t node = nullptr;
        if (addSystemWaitValue64Node(
                &node,
                graph,
                dependencies,
                dependency_count,
                signal,
                value) != hipSuccess ||
            !node)
        {
            return false;
        }
        return hipStreamUpdateCaptureDependencies(
                   stream,
                   &node,
                   1u,
                   hipStreamSetCaptureDependencies) == hipSuccess;
    }

    bool appendActiveCaptureSystemReleaseValue64(
        hipStream_t stream,
        void *signal,
        std::uint64_t value) noexcept
    {
        if (!stream || !signal || value == 0u)
            return false;
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        unsigned long long capture_id = 0u;
        hipGraph_t graph = nullptr;
        const hipGraphNode_t *dependencies = nullptr;
        std::size_t dependency_count = 0u;
        if (hipStreamGetCaptureInfo_v2(
                stream,
                &capture_status,
                &capture_id,
                &graph,
                &dependencies,
                &dependency_count) != hipSuccess ||
            capture_status != hipStreamCaptureStatusActive || !graph ||
            capture_id == 0u ||
            (dependency_count != 0u && !dependencies))
        {
            return false;
        }
        hipGraphNode_t node = nullptr;
        if (addSystemReleaseValue64Node(
                &node,
                graph,
                dependencies,
                dependency_count,
                signal,
                value) != hipSuccess ||
            !node)
        {
            return false;
        }
        return hipStreamUpdateCaptureDependencies(
                   stream,
                   &node,
                   1u,
                   hipStreamSetCaptureDependencies) == hipSuccess;
    }
} // namespace llaminar2::hip_graph_timeline

#endif // HAVE_ROCM
