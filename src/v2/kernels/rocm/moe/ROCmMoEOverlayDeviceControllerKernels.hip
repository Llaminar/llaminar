/**
 * @file ROCmMoEOverlayDeviceControllerKernels.hip
 * @brief HIP launch bridge for the node-local sole-controller lifecycle.
 *
 * Every action is submitted to the caller's exact non-default stream. The
 * shared implementation performs only system-scope mapped-memory operations;
 * it does not synchronize, allocate, copy through the host, or infer topology.
 */

#include "ROCmMoEOverlayDeviceControllerKernels.h"

#include <hip/hip_runtime.h>

#include "../../common/MoEOverlayDeviceControllerDevice.inl"

namespace
{
    /** @return HIP's device-wide constant-frequency wall clock. */
    __device__ __forceinline__ std::uint64_t serviceNowTick() noexcept
    {
        return static_cast<std::uint64_t>(wall_clock64());
    }

    /** Convert constant-frequency wall-clock ticks to rounded-up nanoseconds. */
    __device__ __forceinline__ std::uint64_t serviceTicksToNanoseconds(
        std::uint64_t ticks,
        std::uint64_t clock_rate_khz) noexcept
    {
        if (clock_rate_khz == 0u)
            return 0u;
        constexpr std::uint64_t kNanosecondsPerKilohertz = 1000000u;
        const std::uint64_t whole = ticks / clock_rate_khz;
        const std::uint64_t remainder = ticks % clock_rate_khz;
        return whole * kNanosecondsPerKilohertz +
               (remainder * kNanosecondsPerKilohertz +
                clock_rate_khz - 1u) /
                   clock_rate_khz;
    }
} // namespace

#define LLAMINAR_MOE_SERVICE_NOW_TICK() serviceNowTick()
#define LLAMINAR_MOE_SERVICE_TICKS_TO_NS(delta, clock_rate_khz) \
    serviceTicksToNanoseconds((delta), (clock_rate_khz))
#include "../../common/DeviceMoEOverlayServiceTelemetryDevice.inl"

namespace
{
    /** Bind the exact HIP device without accepting a default stream. */
    bool selectLaunchContext(int device_ordinal, void *stream) noexcept
    {
        return device_ordinal >= 0 && stream != nullptr &&
               hipSetDevice(device_ordinal) == hipSuccess;
    }
} // namespace

extern "C" bool hipMoEOverlayRunDeviceControllerAction(
    const llaminar2::MoEOverlayDeviceControllerActionLaunch *launch,
    int device_ordinal,
    void *stream)
{
    if (!launch || !launch->valid() ||
        !selectLaunchContext(device_ordinal, stream))
    {
        return false;
    }
    hipLaunchKernelGGL(
        llaminar2::moe_overlay_controller_device::controllerActionKernel,
        dim3(1u),
        dim3(llaminar2::moe_overlay_controller_device::kControllerThreads),
        0u,
        reinterpret_cast<hipStream_t>(stream),
        *launch);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool hipMoEOverlayBeginServiceTelemetry(
    llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
    int device_ordinal,
    void *stream)
{
    if (!sample || !selectLaunchContext(device_ordinal, stream))
        return false;
    hipLaunchKernelGGL(
        llaminar2::moe_overlay_service_device::beginServiceTelemetryKernel,
        dim3(1u),
        dim3(1u),
        0u,
        reinterpret_cast<hipStream_t>(stream),
        sample);
    return hipGetLastError() == hipSuccess;
}

extern "C" bool hipMoEOverlayFinishServiceTelemetry(
    const void *runtime_layer,
    llaminar2::DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
    llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
    std::uint32_t num_experts,
    std::uint32_t phase_hint,
    const llaminar2::MoEOverlayInferenceGraphRole *runtime_graph_role,
    int device_ordinal,
    void *stream)
{
    int wall_clock_rate_khz = 0;
    if (!runtime_layer || !layer_telemetry || !sample || num_experts == 0u ||
        !selectLaunchContext(device_ordinal, stream) ||
        hipDeviceGetAttribute(
            &wall_clock_rate_khz,
            hipDeviceAttributeWallClockRate,
            device_ordinal) != hipSuccess ||
        wall_clock_rate_khz <= 0)
    {
        return false;
    }
    hipLaunchKernelGGL(
        llaminar2::moe_overlay_service_device::finishServiceTelemetryKernel,
        dim3(1u),
        dim3(llaminar2::moe_overlay_service_device::kTelemetryThreads),
        0u,
        reinterpret_cast<hipStream_t>(stream),
        runtime_layer,
        layer_telemetry,
        sample,
        num_experts,
        phase_hint,
        runtime_graph_role,
        static_cast<std::uint64_t>(wall_clock_rate_khz));
    return hipGetLastError() == hipSuccess;
}

extern "C" bool hipMoEOverlayPublishServiceTelemetry(
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
    hipLaunchKernelGGL(
        llaminar2::moe_overlay_service_device::publishServiceTelemetryKernel,
        dim3(1u),
        dim3(llaminar2::moe_overlay_service_device::kTelemetryThreads),
        0u,
        reinterpret_cast<hipStream_t>(stream),
        telemetry,
        samples,
        layer_count,
        participant_id,
        publication);
    return hipGetLastError() == hipSuccess;
}
