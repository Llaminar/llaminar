/**
 * @file CUDAMoEOverlayDeviceControllerKernels.cu
 * @brief CUDA launch bridge for the node-local sole-controller lifecycle.
 *
 * Every action is submitted to the caller's exact non-default stream. The
 * shared implementation performs only system-scope mapped-memory operations;
 * it does not synchronize, allocate, copy through the host, or infer topology.
 */

#include "CUDAMoEOverlayDeviceControllerKernels.h"

#include <cuda_runtime.h>

#include "../../common/MoEOverlayDeviceControllerDevice.inl"

namespace
{
    /** @return CUDA's device-wide nanosecond timer for cross-SM intervals. */
    __device__ __forceinline__ std::uint64_t serviceNowTick() noexcept
    {
        std::uint64_t tick = 0u;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tick));
        return tick;
    }
} // namespace

#define LLAMINAR_MOE_SERVICE_NOW_TICK() serviceNowTick()
#define LLAMINAR_MOE_SERVICE_TICKS_TO_NS(delta, clock_rate_khz) (delta)
#include "../../common/DeviceMoEOverlayServiceTelemetryDevice.inl"

namespace
{
    /** Bind the exact CUDA device without accepting a default stream. */
    bool selectLaunchContext(int device_ordinal, void *stream) noexcept
    {
        return device_ordinal >= 0 && stream != nullptr &&
               cudaSetDevice(device_ordinal) == cudaSuccess;
    }
} // namespace

extern "C" bool cudaMoEOverlayRunDeviceControllerAction(
    const llaminar2::MoEOverlayDeviceControllerActionLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    llaminar2::moe_overlay_controller_device::controllerActionKernel
        <<<1u,
           llaminar2::moe_overlay_controller_device::kControllerThreads,
           0u,
           reinterpret_cast<cudaStream_t>(stream)>>>(*launch);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool cudaMoEOverlayBeginServiceTelemetry(
    llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
    int device_ordinal,
    void *stream)
{
    if (!sample || !selectLaunchContext(device_ordinal, stream))
        return false;
    llaminar2::moe_overlay_service_device::beginServiceTelemetryKernel
        <<<1u, 1u, 0u, reinterpret_cast<cudaStream_t>(stream)>>>(sample);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool cudaMoEOverlayFinishServiceTelemetry(
    const void *runtime_layer,
    llaminar2::DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
    llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
    std::uint32_t num_experts,
    std::uint32_t phase_hint,
    const llaminar2::MoEOverlayInferenceGraphRole *runtime_graph_role,
    int device_ordinal,
    void *stream)
{
    if (!runtime_layer || !layer_telemetry || !sample || num_experts == 0u ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    llaminar2::moe_overlay_service_device::finishServiceTelemetryKernel
        <<<1u,
           llaminar2::moe_overlay_service_device::kTelemetryThreads,
           0u,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            runtime_layer,
            layer_telemetry,
            sample,
            num_experts,
            phase_hint,
            runtime_graph_role,
            /*clock_rate_khz=*/0u);
    return cudaGetLastError() == cudaSuccess;
}

extern "C" bool cudaMoEOverlayPublishServiceTelemetry(
    const llaminar2::DeviceMoEOverlayServiceTelemetryCell *telemetry,
    const llaminar2::DeviceMoEOverlayServiceTelemetrySample *samples,
    std::uint32_t layer_count,
    std::int32_t participant_id,
    llaminar2::MoEOverlayDeviceServiceTelemetryPublicationHeader *publication,
    int device_ordinal,
    void *stream)
{
    if (!telemetry || !samples || !publication || layer_count == 0u ||
        participant_id < 0 || !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    llaminar2::moe_overlay_service_device::publishServiceTelemetryKernel
        <<<1u,
           llaminar2::moe_overlay_service_device::kTelemetryThreads,
           0u,
           reinterpret_cast<cudaStream_t>(stream)>>>(
            telemetry,
            samples,
            layer_count,
            participant_id,
            publication);
    return cudaGetLastError() == cudaSuccess;
}
