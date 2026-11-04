/**
 * @file MoEOverlayStaticJointRouteMembership.h
 * @brief Immutable source-cohort projection for captured CPU-shard dispatch.
 *
 * A whole expert has one durable device-bank target. A CPU tensor-sharded expert
 * is external to that local bank and must deliver its original routed input to
 * every actual cohort member. This small value embeds only the selected lane's
 * immutable expert membership in its existing captured launch. It is neither a
 * placement authority nor a mutable host mirror, and owns no device allocation.
 * The original placement epoch, layer and target must still match before use.
 */
#pragma once

#include "DeviceMoERuntimeABI.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_JOINT_HD __host__ __device__
#else
#define LLAMINAR_MOE_JOINT_HD
#endif

namespace llaminar2
{
    /** @brief Capture-stable membership for one original CPU cohort participant. */
    class MoEOverlayStaticJointRouteMembership final
    {
    public:
        /** @brief Empty membership retains ordinary versioned single-owner routing. */
        constexpr MoEOverlayStaticJointRouteMembership() noexcept = default;

        /**
         * @brief Freeze a cold projection of original canonical source ownership.
         * @param expert_count Original logical expert axis.
         * @param target Original global participant ID, not a dense socket index.
         * @param layer Original model-global layer.
         * @param epoch Original immutable Static placement epoch.
         * @param expert_ids Exact expert IDs whose source cohort contains target.
         * @return An immutable launch value; no native readiness is asserted.
         * @throws std::invalid_argument For incomplete or out-of-range metadata.
         */
        static MoEOverlayStaticJointRouteMembership fromProjection(
            std::uint32_t expert_count, std::int32_t target,
            std::int32_t layer, std::uint64_t epoch,
            std::span<const std::int32_t> expert_ids)
        {
            if (expert_count == 0u || expert_count > moe_runtime_abi::kMaxExperts ||
                target < 0 || layer < 0 || epoch == 0u || expert_ids.empty())
                throw std::invalid_argument("Static joint dispatch requires original complete source membership");
            MoEOverlayStaticJointRouteMembership result;
            result.expert_count_ = expert_count;
            result.target_ = target;
            result.layer_ = layer;
            result.epoch_ = epoch;
            for (const auto expert : expert_ids)
            {
                if (expert < 0 || static_cast<std::uint32_t>(expert) >= expert_count ||
                    result.contains(expert))
                    throw std::invalid_argument("Static joint dispatch has an invalid or duplicate source expert");
                result.bits_[static_cast<std::uint32_t>(expert) / 32u] |=
                    std::uint32_t{1} << (static_cast<std::uint32_t>(expert) % 32u);
            }
            return result;
        }

        /** @return Whether this lane owns a declared immutable joint projection. */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr bool enabled() const noexcept
        {
            return expert_count_ != 0u;
        }

        /** @return Whether the captured projection belongs to this exact launch. */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr bool validFor(
            std::uint32_t experts, std::int32_t target, std::int32_t layer) const noexcept
        {
            return !enabled() ||
                (experts == expert_count_ && target == target_ && layer == layer_);
        }

        /** @return Immutable source epoch for a cold original-group projection. */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr std::uint64_t initialEpoch() const noexcept { return epoch_; }

        /** @return Whether the acquired device-bank epoch is the original Static epoch. */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr bool acceptsEpoch(
            std::uint64_t epoch) const noexcept
        {
            return !enabled() || epoch == epoch_;
        }

        /** @return Whether this original expert's cohort contains the captured target. */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr bool contains(
            std::int32_t expert) const noexcept
        {
            return expert >= 0 && static_cast<std::uint32_t>(expert) < expert_count_ &&
                (bits_[static_cast<std::uint32_t>(expert) / 32u] &
                 (std::uint32_t{1} << (static_cast<std::uint32_t>(expert) % 32u))) != 0u;
        }

        /**
         * @brief Resolve only this lane's dispatch selection, never whole ownership.
         * @param expert Original logical expert ID.
         * @param durable_target Current scalar bank entry; -1 denotes external work.
         * @param resolved Receives the target used by this lane's existing compactor.
         * @return False for malformed bank data or a conflicting whole/joint owner.
         *
         * Every target uses its own original membership. Thus a joint expert can
         * reach all 2..8 members without selecting one fragment as a whole owner.
         */
        [[nodiscard]] LLAMINAR_MOE_JOINT_HD constexpr bool resolve(
            std::int32_t expert, std::int32_t durable_target,
            std::int32_t *resolved) const noexcept
        {
            if (!resolved || durable_target < -1)
                return false;
            if (contains(expert))
            {
                if (durable_target != -1)
                    return false;
                *resolved = target_;
            }
            else
                *resolved = durable_target;
            return true;
        }

    private:
        std::uint32_t expert_count_ = 0u;
        std::int32_t target_ = -1;
        std::int32_t layer_ = -1;
        std::uint64_t epoch_ = 0u;
        std::uint32_t bits_[(moe_runtime_abi::kMaxExperts + 31u) / 32u] = {};
    };

    static_assert(std::is_trivially_copyable_v<MoEOverlayStaticJointRouteMembership>);
}

#undef LLAMINAR_MOE_JOINT_HD
