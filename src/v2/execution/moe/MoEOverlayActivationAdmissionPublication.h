/**
 * @file MoEOverlayActivationAdmissionPublication.h
 * @brief Ordered revalidation of an original mapped admission publication.
 *
 * The peer may close admission before replacing its generation. Re-reading
 * ready first would let that old release authorize a closed/new publication.
 * This stateless inline helper owns the generation-then-ready order for actual
 * device validation and adversarial host replay. Accessors remain exact native
 * system-acquire loads in production; no device state is mirrored on the host.
 */
#pragma once

#include "MoEOverlayActivationEpochABI.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_ADMISSION_HD __host__ __device__
#else
#define LLAMINAR_MOE_ADMISSION_HD
#endif

namespace llaminar2
{
    /**
     * @brief Reject unknown lifecycle bytes before the original native rearm CAS.
     * @param state Acquired original admission state, without normalization.
     * @return True only for the original Idle or Armed lifecycle values.
     * Failed remains absorbing, and invalid ABI state is never repaired to Idle.
     */
    [[nodiscard]] LLAMINAR_MOE_ADMISSION_HD constexpr bool
    moeOverlayNativeInvocationRearmStateAllowed(std::uint32_t state) noexcept
    {
        return state == static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Idle) ||
               state == static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Armed);
    }

    /**
     * @brief Close the original ready publication without replacing cancellation.
     * @param acquire_ready Exact system acquire of the original mapped word.
     * @param compare_ready Strong system CAS(expected, desired), with acquire/release ordering.
     * @return True only when the same ordinary publication was changed to zero.
     * Cancellation may arrive after the acquire; the CAS must reject that race.
     * The helper owns no error or lifecycle state and cannot replace either.
     */
    template <class ReadyAcquire, class ReadyCompare>
    [[nodiscard]] LLAMINAR_MOE_ADMISSION_HD inline bool
    closeMoEOverlayActivationAdmissionPublication(
        ReadyAcquire acquire_ready, ReadyCompare compare_ready) noexcept
    {
        const auto ready = acquire_ready();
        return (ready == 0u || ready == kMoEOverlayActivationAdmissionTimeline) &&
               compare_ready(ready, std::uint64_t{0});
    }

    /**
     * @brief Publish the completed immutable admission without replacing Abort.
     * @param compare_ready Strong system CAS(expected, desired), with acquire/release ordering.
     * @return True only when the original closed word was changed to ready.
     * Successful release publishes all preceding identity/status writes. A
     * cancellation that won first remains authoritative; there is no retry.
     */
    template <class ReadyCompare>
    [[nodiscard]] LLAMINAR_MOE_ADMISSION_HD inline bool
    publishMoEOverlayActivationAdmissionPublication(ReadyCompare compare_ready) noexcept
    {
        return compare_ready(std::uint64_t{0}, kMoEOverlayActivationAdmissionTimeline);
    }

    /**
     * @brief Stop an unsuccessful grant poll on the original absorbing failure.
     * @param ready Acquired original publication word after admission did not succeed.
     * @param acquire_state Exact original system acquire, evaluated only if needed.
     * @return True for Abort or Failed without changing the original error.
     * Successful grant acquisition never enters this unsuccessful-poll helper.
     */
    template <class StateAcquire>
    [[nodiscard]] LLAMINAR_MOE_ADMISSION_HD inline bool
    moeOverlayActivationAdmissionWaitAborted(
        std::uint64_t ready, StateAcquire acquire_state) noexcept
    {
        return ready == kMoEOverlayActivationAbortTimeline ||
               acquire_state() == static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Failed);
    }

    /**
     * @brief Authenticate a snapshot only through generation then final ready acquire.
     * @param snapshot_generation Generation already copied with the candidate identity.
     * @param acquire_generation Exact acquire accessor for the original identity generation.
     * @param acquire_ready Exact acquire accessor for the original final publication word.
     * @return Whether the snapshot still belongs to a fully published current admission.
     * Accessors are stack closures, inlined on the actual device. Their separate
     * statements are essential: a ready acquire must follow generation revalidation.
     */
    template <class GenerationAcquire, class ReadyAcquire>
    [[nodiscard]] LLAMINAR_MOE_ADMISSION_HD inline bool
    moeOverlayActivationAdmissionPublicationCurrent(
        std::uint64_t snapshot_generation,
        GenerationAcquire acquire_generation,
        ReadyAcquire acquire_ready) noexcept
    {
        const auto current_generation = acquire_generation();
        const auto current_ready = acquire_ready();
        return moeOverlayActivationAdmissionPublicationMatches(
            snapshot_generation, current_generation, current_ready);
    }
}

#undef LLAMINAR_MOE_ADMISSION_HD
