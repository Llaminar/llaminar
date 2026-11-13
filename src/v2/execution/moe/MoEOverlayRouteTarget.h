/**
 * @file MoEOverlayRouteTarget.h
 * @brief Typed target encoded in the existing acquired placement-bank word.
 *
 * Whole participants and fixed CPU cohort tiers occupy disjoint encodings.
 * This value owns no storage or membership authority: expert membership is
 * determined only by its word in the acquired versioned placement bank.
 */
#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ROUTE_TARGET_HD __host__ __device__
#else
#define LLAMINAR_ROUTE_TARGET_HD
#endif
namespace llaminar2
{
    /** @brief Checked alternatives over one existing signed route word. */
    class MoEOverlayRouteTarget final
    {
    public:
        /** @return Unassigned target; never a synthetic whole owner. */
        LLAMINAR_ROUTE_TARGET_HD constexpr MoEOverlayRouteTarget() noexcept = default;
        /**
         * @brief Encode one canonical whole-owner destination.
         * @param participant Canonical global whole participant.
         * @return Its checked target.
         * @throws std::invalid_argument For a negative participant.
         */
        static MoEOverlayRouteTarget whole(std::int32_t participant)
        {
            if (participant < 0)
                throw std::invalid_argument("Whole route target requires a global participant");
            return fromWord(participant);
        }
        /**
         * @brief Encode an original fixed CPU cohort without selecting a scalar representative.
         * @param tier Original canonical fixed CPU cohort tier.
         * @return Its checked target.
         * @throws std::invalid_argument For a negative or unencodable tier.
         */
        static MoEOverlayRouteTarget cpuCohort(std::int32_t tier)
        {
            if (tier < 0 || tier > std::numeric_limits<std::int32_t>::max() - 1)
                throw std::invalid_argument(
                    "CPU cohort route target has an unencodable original tier");
            return fromWord(static_cast<std::int32_t>(-std::int64_t(tier) - 2));
        }
        /** @brief Interpret an existing bank word. @param word Placement encoding. @return Its
         * typed view. */
        LLAMINAR_ROUTE_TARGET_HD static constexpr MoEOverlayRouteTarget
        fromWord(std::int32_t word) noexcept
        {
            return MoEOverlayRouteTarget(word);
        }
        /** @return Exact original word, preserving existing bank storage/layout. */
        LLAMINAR_ROUTE_TARGET_HD constexpr std::int32_t word() const noexcept
        {
            return word_;
        }
        /** @return Whether this target selects one whole participant. */
        LLAMINAR_ROUTE_TARGET_HD constexpr bool isWhole() const noexcept
        {
            return word_ >= 0;
        }
        /** @return Whether this target selects a fixed complete CPU cohort. */
        LLAMINAR_ROUTE_TARGET_HD constexpr bool isCPUCohort() const noexcept
        {
            return word_ < -1;
        }
        /** @return Whether no target was assigned. */
        LLAMINAR_ROUTE_TARGET_HD constexpr bool isAbsent() const noexcept
        {
            return word_ == -1;
        }

    private:
        /** @param word Canonical signed encoding; all bit patterns have one alternative. */
        LLAMINAR_ROUTE_TARGET_HD explicit constexpr MoEOverlayRouteTarget(
            std::int32_t word) noexcept
            : word_(word)
        {
        }
        std::int32_t word_ = -1;
    };
} // namespace llaminar2
#undef LLAMINAR_ROUTE_TARGET_HD
