/**
 * @file MoEOverlayJointRouteTarget.h
 * @brief Capture-stable CPU cohort geometry without frozen expert membership.
 *
 * The acquired placement bank alone selects which experts occupy the fixed
 * cohort. This launch value retains only its original target/member identity.
 */
#pragma once
#include "MoEOverlayRouteTarget.h"
#include "DeviceMoERuntimeABI.h"
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_JOINT_TARGET_HD __host__ __device__
#else
#define LLAMINAR_JOINT_TARGET_HD
#endif
namespace llaminar2
{
    /** @brief Original fixed local member of an externally computed CPU cohort. */
    class MoEOverlayJointRouteTarget final
    {
    public:
        /** @brief Ordinary lane with no CPU cohort role. */
        constexpr MoEOverlayJointRouteTarget() noexcept = default;
        /**
         * @brief Retain original fixed cohort/member geometry for capture.
         * @param experts Original logical expert axis.
         * @param participant Exact global cohort member.
         * @param layer Original model-global layer.
         * @param tier Original canonical cohort tier, independent of incoming IDs.
         * @return Capture-stable metadata; no execution readiness is asserted.
         * @throws std::invalid_argument For missing or unencodable original geometry.
         */
        static MoEOverlayJointRouteTarget fromOriginalCohort(std::uint32_t experts,
                                                             std::int32_t participant,
                                                             std::int32_t layer, std::int32_t tier)
        {
            if (!experts || experts > moe_runtime_abi::kMaxExperts || participant < 0 || layer < 0)
                throw std::invalid_argument(
                    "Joint route target requires original cohort/member geometry");
            MoEOverlayJointRouteTarget result;
            result.experts_ = experts;
            result.participant_ = participant;
            result.layer_ = layer;
            result.cohort_ = MoEOverlayRouteTarget::cpuCohort(tier).word();
            return result;
        }
        /** @return Whether this lane has an original cohort role. */
        LLAMINAR_JOINT_TARGET_HD constexpr bool enabled() const noexcept
        {
            return experts_ != 0;
        }
        /** @return Original immutable cohort tag, or absent for an ordinary lane. */
        LLAMINAR_JOINT_TARGET_HD constexpr std::int32_t cohortTarget() const noexcept
        {
            return cohort_;
        }
        /**
         * @param experts Original logical expert axis.
         * @param participant Canonical global target member.
         * @param layer Original model-global layer.
         * @return Whether capture retained this exact geometry.
         */
        LLAMINAR_JOINT_TARGET_HD constexpr bool
        validFor(std::uint32_t experts, std::int32_t participant, std::int32_t layer) const noexcept
        {
            return !enabled() ||
                   (experts_ == experts && participant_ == participant && layer_ == layer);
        }
        /**
         * @brief Resolve live membership exclusively from the acquired bank word.
         * @param expert Original expert from the acquired bank.
         * @param word SAME acquired bank's current target encoding.
         * @param resolved Receives this lane's selected global target, or absent.
         * @return False for invalid expert or a whole/cohort policy conflict.
         */
        LLAMINAR_JOINT_TARGET_HD constexpr bool resolve(std::int32_t expert, std::int32_t word,
                                                        std::int32_t *resolved) const noexcept
        {
            if (!resolved || expert < 0 || (enabled() && std::uint32_t(expert) >= experts_))
                return false;
            const auto target = MoEOverlayRouteTarget::fromWord(word);
            if (target.isWhole())
            {
                if (enabled() && word == participant_)
                    return false;
                *resolved = word;
            }
            else
                *resolved = enabled() && word == cohort_ ? participant_ : -1;
            return true;
        }

    private:
        std::uint32_t experts_ = 0;
        std::int32_t participant_ = -1;
        std::int32_t layer_ = -1;
        std::int32_t cohort_ = -1;
    };
    static_assert(std::is_trivially_copyable_v<MoEOverlayJointRouteTarget>);
} // namespace llaminar2
#undef LLAMINAR_JOINT_TARGET_HD
