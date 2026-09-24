/**
 * @file InferenceMeasurementReadiness.h
 * @brief Typed readiness contract for production performance measurement.
 *
 * Some production execution policies perform bounded topology preparation
 * before request admission, then learn service economics from ordinary live
 * traffic before enabling optimization. This header keeps the preparation
 * lifecycle visible without coupling the HTTP server or benchmark runner to a
 * concrete controller. A subsystem may block only for work that does not
 * require inference; natural service telemetry must never request synthetic
 * benchmark traffic.
 */

#pragma once

#include <cstdint>
#include <string>

namespace llaminar2
{
    /** Observable state of all mandatory pre-measurement production work. */
    enum class InferenceMeasurementReadinessState : std::uint8_t
    {
        Ready,       ///< No preparation is required, or all work is complete.
        Calibrating, ///< Bounded non-inference preparation is still active.
        Failed,      ///< Preparation failed and timing would be meaningless.
    };

    /**
     * @brief Legacy diagnostic vocabulary for generic preparation owners.
     *
     * Production readiness owners currently publish `None`: preparation must
     * not ask a caller to synthesize model traffic. The other values remain as
     * protocol diagnostics for rejecting a regressing implementation.
     */
    enum class InferenceMeasurementWorkloadKind : std::uint8_t
    {
        None,            ///< No inference transaction is currently requested.
        Prefill,         ///< One ordinary complete prefill schedule.
        Decode,          ///< Serial decode transactions after request prefill.
        GroupedVerifier, ///< MTP decode transactions containing grouped verify.
        FullRequest,     ///< Prefill plus every configured decode phase.
    };

    /**
     * @brief Immutable snapshot of one runner's measurement readiness.
     *
     * `completed_work_units` and `required_work_units` describe controller-
     * defined durable work, not attempted requests.  They may remain equal
     * while an asynchronous transfer or evidence exchange advances.  Callers
     * must therefore use @ref state as authority and treat the counters only
     * as diagnostics.
     */
    struct InferenceMeasurementReadiness
    {
        InferenceMeasurementReadinessState state =
            InferenceMeasurementReadinessState::Ready;
        std::uint64_t completed_work_units = 0u;
        std::uint64_t required_work_units = 0u;
        /** True only while one ordinary inference transaction can advance work. */
        bool inference_requested = false;
        /** Monotonic identity of the currently requested inference sample. */
        std::uint64_t inference_request_generation = 0u;
        InferenceMeasurementWorkloadKind requested_workload =
            InferenceMeasurementWorkloadKind::None;
        std::string owner;      ///< Stable subsystem name, empty when trivial.
        std::string phase;      ///< Current controller phase for diagnostics.
        std::string diagnostic; ///< First fatal detail when state is Failed.

        /** @return Whether timed inference may begin now. */
        [[nodiscard]] bool ready() const noexcept
        {
            return state == InferenceMeasurementReadinessState::Ready;
        }

        /** @return Whether the runner has reported a terminal preparation error. */
        [[nodiscard]] bool failed() const noexcept
        {
            return state == InferenceMeasurementReadinessState::Failed;
        }
    };
} // namespace llaminar2
