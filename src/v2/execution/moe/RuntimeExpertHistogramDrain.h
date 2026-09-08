/**
 * @file RuntimeExpertHistogramDrain.h
 * @brief Typed non-blocking progress for device-owned routing evidence.
 *
 * Runtime MoE kernels retain exact route counts on their owning device. The
 * maintenance worker advances those counts through an event-polled drain; it
 * must never turn temporary device work into a host stream synchronization.
 */

#pragma once

#include <string>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Authority phase governing whether route rows may enter a window.
     *
     * Service measurements initially contribute calibration evidence. Once
     * those measurements are complete, maintenance closes admission before it
     * drains and rotates every host/device bank. The resulting quarantine
     * remains closed until the next public request boundary; this prevents the
     * tail of the request that completed certification from becoming movement
     * demand. `OptimizationDemand` is the only post-certificate live phase.
     */
    enum class RuntimeExpertHistogramAdmission
    {
        CalibrationEvidence,
        CertificationQuarantine,
        OptimizationDemand,
    };

    /** @return Whether inference rows belong in the selected histogram bank. */
    [[nodiscard]] constexpr bool admitsRuntimeExpertHistogramRows(
        RuntimeExpertHistogramAdmission admission) noexcept
    {
        return admission !=
               RuntimeExpertHistogramAdmission::CertificationQuarantine;
    }

    /** @brief Lifecycle result from one non-blocking histogram-drain poll. */
    enum class RuntimeExpertHistogramDrainProgress
    {
        Pending, ///< Exact device/transfer work remains in flight.
        Ready,   ///< This source merged one complete generation exactly once.
        Failed,  ///< The source cannot preserve its evidence contract.
    };

    /** @brief Progress plus an actionable fatal diagnostic when failed. */
    struct RuntimeExpertHistogramDrainResult
    {
        RuntimeExpertHistogramDrainProgress progress =
            RuntimeExpertHistogramDrainProgress::Pending;
        std::string error;

        /** @return A pending result with no diagnostic allocation. */
        [[nodiscard]] static RuntimeExpertHistogramDrainResult pending()
        {
            return {};
        }

        /** @return A successfully merged generation. */
        [[nodiscard]] static RuntimeExpertHistogramDrainResult ready()
        {
            return {.progress =
                        RuntimeExpertHistogramDrainProgress::Ready};
        }

        /** @return A terminal failure carrying its ownership diagnostic. */
        [[nodiscard]] static RuntimeExpertHistogramDrainResult failed(
            std::string message)
        {
            return {
                .progress = RuntimeExpertHistogramDrainProgress::Failed,
                .error = std::move(message),
            };
        }
    };
} // namespace llaminar2
