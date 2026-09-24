/**
 * @file CUDAMoEOverlayDeviceControllerKernels.h
 * @brief Explicit-stream CUDA bridge for mapped ExpertOverlay control.
 */

#pragma once

#include "execution/moe/MoEOverlayDeviceControllerKernels.h"
#include "execution/moe/DeviceMoEOverlayServiceTelemetry.h"

extern "C"
{
    /** @brief Enqueue one graph-capturable controller transition. */
    bool cudaMoEOverlayRunDeviceControllerAction(
        const llaminar2::MoEOverlayDeviceControllerActionLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one device-local service start marker. */
    bool cudaMoEOverlayBeginServiceTelemetry(
        llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one device-local routed-expert service observation. */
    bool cudaMoEOverlayFinishServiceTelemetry(
        const void *runtime_layer,
        llaminar2::DeviceMoEOverlayServiceTelemetryCell *layer_telemetry,
        llaminar2::DeviceMoEOverlayServiceTelemetrySample *sample,
        std::uint32_t num_experts,
        std::uint32_t phase_hint,
        const llaminar2::MoEOverlayInferenceGraphRole *runtime_graph_role,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one finite device-to-mapped telemetry publication. */
    bool cudaMoEOverlayPublishServiceTelemetry(
        const llaminar2::DeviceMoEOverlayServiceTelemetryCell *telemetry,
        const llaminar2::DeviceMoEOverlayServiceTelemetrySample *samples,
        std::uint32_t layer_count,
        std::int32_t participant_id,
        llaminar2::MoEOverlayDeviceServiceTelemetryPublicationHeader
            *publication,
        int device_ordinal,
        void *stream);
}
