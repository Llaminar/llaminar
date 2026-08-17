/**
 * @file ROCmMoEOverlayActivationPacketKernels.h
 * @brief Explicit-stream HIP bridge for node-local ExpertOverlay packets.
 */

#pragma once

#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"

extern "C"
{
    /** @brief Enqueue deterministic continuation dispatch compaction. */
    bool hipMoEOverlayActivationPackDispatch(
        const llaminar2::MoEOverlayActivationDispatchPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue follower descriptor validation and CSR expansion. */
    bool hipMoEOverlayActivationConsumeDispatch(
        const llaminar2::MoEOverlayActivationDispatchConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue follower compact return publication. */
    bool hipMoEOverlayActivationPackReturn(
        const llaminar2::MoEOverlayActivationReturnPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue continuation validation and ordered FP32 accumulation. */
    bool hipMoEOverlayActivationConsumeReturn(
        const llaminar2::MoEOverlayActivationReturnConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row dispatch and system publication. */
    bool hipMoEOverlayActivationPackSingleRowDispatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row dispatch acquire and materialization. */
    bool hipMoEOverlayActivationConsumeSingleRowDispatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row return and system publication. */
    bool hipMoEOverlayActivationPackSingleRowReturn(
        const llaminar2::MoEOverlayActivationSingleRowReturnPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row return acquire and ordered fold. */
    bool hipMoEOverlayActivationConsumeSingleRowReturn(
        const llaminar2::MoEOverlayActivationSingleRowReturnConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one block per topology-declared one-row dispatch lane. */
    bool hipMoEOverlayActivationPackSingleRowDispatchBatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue parallel return acquisition and canonical ordered fold. */
    bool hipMoEOverlayActivationConsumeSingleRowReturnBatch(
        const llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue topology-parallel multi-row validation and one ordered fold. */
    bool hipMoEOverlayActivationConsumeMultiRowReturnBatch(
        const llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Publish only runtime-assigned canonical routes to one mapped peer lane. */
    bool hipMoEOverlayPublishNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRoutePublishLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Wait for every peer publication before root-side payload DMA. */
    bool hipMoEOverlayAcquireNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Pull only peer-owned mapped rows into stable root-device scratch. */
    bool hipMoEOverlayStageNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Validate staged peer rows, fold in route order, and acknowledge. */
    bool hipMoEOverlayFoldNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);
}
