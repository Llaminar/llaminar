/**
 * @file ROCmMoEOverlayEpochKernels.h
 * @brief Explicit-stream HIP launch ABI for ExpertOverlay epoch RCU.
 *
 * Every operation is asynchronous and graph-capturable.  Launch success and
 * semantic success are intentionally separate: the latter is published into
 * the caller-owned device status record.
 */

#pragma once

#include "execution/moe/DeviceMoEOverlayEpochABI.h"

#include <cstdint>

extern "C"
{
    /** @brief Enqueue request admission into the currently published bank. */
    bool hipMoEOverlayEpochAcquire(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        llaminar2::DeviceMoEOverlayEpochTicket *ticket,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue release of one request-lifetime reader ticket. */
    bool hipMoEOverlayEpochRelease(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        llaminar2::DeviceMoEOverlayEpochTicket *ticket,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue reservation of the reusable bank for a live device epoch. */
    bool hipMoEOverlayEpochReserveCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue Candidate-to-Ready after transfer events have been joined. */
    bool hipMoEOverlayEpochMarkCandidateReady(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue atomic publication of one ready placement epoch. */
    bool hipMoEOverlayEpochPublishCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue cancellation of one unpublished candidate epoch. */
    bool hipMoEOverlayEpochAbortCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue a non-blocking grace-period retirement attempt. */
    bool hipMoEOverlayEpochRetire(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);
}
