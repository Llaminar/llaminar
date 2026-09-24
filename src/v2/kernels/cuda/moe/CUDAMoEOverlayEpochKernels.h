/**
 * @file CUDAMoEOverlayEpochKernels.h
 * @brief Explicit-stream CUDA launch ABI for ExpertOverlay epoch RCU.
 *
 * These functions enqueue tiny graph-capturable kernels.  A true return value
 * means only that CUDA accepted the launch; callers consume the device-resident
 * status record in stream order to determine the semantic result.
 */

#pragma once

#include "execution/moe/MoEOverlayActivationPacketABI.h"

#include <cstdint>

extern "C"
{
    /** @brief Enqueue request admission into the currently published bank. */
    bool cudaMoEOverlayEpochAcquire(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        llaminar2::DeviceMoEOverlayEpochTicket *ticket,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        const std::uint64_t *external_admission_epoch,
        llaminar2::DeviceMoEOverlayEpochAdmissionBarrierBinding admission_barrier,
        llaminar2::MoEOverlayPeerPlacementEpochBinding peer_placement_epoch,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue release of one request-lifetime reader ticket. */
    bool cudaMoEOverlayEpochRelease(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        llaminar2::DeviceMoEOverlayEpochTicket *ticket,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue reservation of the reusable bank for a live device epoch. */
    bool cudaMoEOverlayEpochReserveCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue Candidate-to-Ready after transfer events have been joined. */
    bool cudaMoEOverlayEpochMarkCandidateReady(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue atomic publication of one ready placement epoch. */
    bool cudaMoEOverlayEpochPublishCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue cancellation of one unpublished candidate epoch. */
    bool cudaMoEOverlayEpochAbortCandidate(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *candidate_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);

    /** @brief Enqueue a non-blocking grace-period retirement attempt. */
    bool cudaMoEOverlayEpochRetire(
        llaminar2::DeviceMoEOverlayEpochControl *control,
        const std::uint64_t *retiring_epoch,
        llaminar2::DeviceMoEOverlayEpochStatus *status,
        int device_ordinal,
        void *stream);
}
