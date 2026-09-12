/**
 * @file MoEOverlayReplicaCacheCapacity.h
 * @brief Immutable replica-count grant selected by physical memory admission.
 *
 * The CLI cache size is an upper bound, not permission to evict mandatory
 * model weights. Admission chooses a positive bounded cache and graph setup
 * consumes that same grant. This value carries geometry, never a byte ledger.
 */
#pragma once

#include <stdexcept>

namespace llaminar2
{
    /** @brief Bind the requested replica limit to its physically admitted count. */
    class MoEOverlayReplicaCacheCapacity final
    {
    public:
        /**
         * @brief Freeze a capacity selected against the complete physical BOM.
         * @param requested Requested upper bound per layer and participant.
         * @param admitted Largest fitting count, or zero only for an off cache.
         * @throws std::invalid_argument for an out-of-range or disabled grant.
         */
        MoEOverlayReplicaCacheCapacity(int requested, int admitted)
            : requested_(requested), admitted_(admitted)
        {
            if (requested < 0 || admitted < 0 || admitted > requested ||
                (requested > 0 && admitted == 0))
                throw std::invalid_argument("Invalid ExpertOverlay replica-cache capacity grant");
        }

        /** @return Unmodified request upper bound, for identity and diagnostics. */
        [[nodiscard]] int requested() const noexcept { return requested_; }
        /** @return Exact per-layer, per-participant count owned by admission. */
        [[nodiscard]] int admitted() const noexcept { return admitted_; }

        /**
         * @brief Authenticate a consumer's policy before exposing its grant.
         * @param requested Current resolved request, including debug overrides.
         * @return Admitted count; changed policy requires a new admission.
         * @throws std::logic_error when a stale grant reaches graph construction.
         */
        [[nodiscard]] int resolve(int requested) const
        {
            if (requested != requested_)
                throw std::logic_error("ExpertOverlay replica-cache request changed after admission");
            return admitted_;
        }

        bool operator==(const MoEOverlayReplicaCacheCapacity &) const = default;

    private:
        int requested_;
        int admitted_;
    };
}
