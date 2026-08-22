/**
 * @file CUDAMoEOverlayActivationPacketKernels.h
 * @brief Explicit-stream CUDA bridge for node-local ExpertOverlay packets.
 */

#pragma once

#include "execution/moe/MoEOverlayActivationPacketABI.h"
#include "execution/moe/MoEOverlayNodeLocalRouteExchangeABI.h"

extern "C"
{
    /** @brief Enqueue deterministic continuation dispatch compaction. */
    bool cudaMoEOverlayActivationPackDispatch(
        const llaminar2::MoEOverlayActivationDispatchPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue follower descriptor validation and CSR expansion. */
    bool cudaMoEOverlayActivationConsumeDispatch(
        const llaminar2::MoEOverlayActivationDispatchConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue follower compact return publication. */
    bool cudaMoEOverlayActivationPackReturn(
        const llaminar2::MoEOverlayActivationReturnPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue continuation validation and ordered FP32 accumulation. */
    bool cudaMoEOverlayActivationConsumeReturn(
        const llaminar2::MoEOverlayActivationReturnConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row dispatch and system publication. */
    bool cudaMoEOverlayActivationPackSingleRowDispatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row dispatch acquire and materialization. */
    bool cudaMoEOverlayActivationConsumeSingleRowDispatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row return and system publication. */
    bool cudaMoEOverlayActivationPackSingleRowReturn(
        const llaminar2::MoEOverlayActivationSingleRowReturnPackLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one fused one-row return acquire and ordered fold. */
    bool cudaMoEOverlayActivationConsumeSingleRowReturn(
        const llaminar2::MoEOverlayActivationSingleRowReturnConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue one block per topology-declared one-row dispatch lane. */
    bool cudaMoEOverlayActivationPackSingleRowDispatchBatch(
        const llaminar2::MoEOverlayActivationSingleRowDispatchBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue parallel return acquisition and canonical ordered fold. */
    bool cudaMoEOverlayActivationConsumeSingleRowReturnBatch(
        const llaminar2::MoEOverlayActivationSingleRowReturnBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue topology-parallel multi-row validation and one ordered fold. */
    bool cudaMoEOverlayActivationConsumeMultiRowReturnBatch(
        const llaminar2::MoEOverlayActivationMultiRowReturnBatchLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Publish only runtime-assigned canonical routes to one mapped peer lane. */
    bool cudaMoEOverlayPublishNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRoutePublishLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Wait for every peer publication before root-side payload DMA. */
    bool cudaMoEOverlayAcquireNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Pull only peer-owned mapped rows into stable root-device scratch. */
    bool cudaMoEOverlayStageNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Validate staged peer rows, fold in route order, and acknowledge. */
    bool cudaMoEOverlayFoldNodeLocalCanonicalRoutes(
        const llaminar2::MoENodeLocalRouteConsumeLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Wait until every peer has released the prior dense payload bank. */
    bool cudaMoEOverlayBeginNodeLocalDensePublication(
        const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Publish the D2H-complete dense payload epoch from the root. */
    bool cudaMoEOverlayFinishNodeLocalDensePublication(
        const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Acquire the next dense payload epoch on one peer stream. */
    bool cudaMoEOverlayBeginNodeLocalDensePublicationConsume(
        const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
        int device_ordinal,
        void *stream);

    /** @brief Acknowledge completion of one peer's dense H2D import. */
    bool cudaMoEOverlayFinishNodeLocalDensePublicationConsume(
        const llaminar2::MoENodeLocalDensePublicationLaunch *launch,
        int device_ordinal,
        void *stream);
}
