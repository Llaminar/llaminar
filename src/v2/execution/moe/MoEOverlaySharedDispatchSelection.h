/**
 * @file MoEOverlaySharedDispatchSelection.h
 * @brief Stateless original-row selection for one authenticated shared channel group.
 *
 * The original router and same acquired bank select useful rows against fixed
 * original consumer identities. This
 * helper owns no union mask, placement state, counters or storage. Metadata
 * validates the complete prefix before the mapped row writer executes.
 */
#pragma once
#include "MoEOverlayActivationPacketABI.h"
#include <span>
#include <bit>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SHARED_ROW_HD __host__ __device__
#else
#define LLAMINAR_SHARED_ROW_HD
#endif
namespace llaminar2 {
    /** @brief Original cold target/member identity; no expert membership or epoch state. */
    struct MoEOverlaySharedDispatchTarget
    {
        std::int32_t participant = -1;
        MoEOverlayJointRouteTarget membership{};
    };
    /**
     * @brief Authenticate a shared consumer group before retaining its channel pointers.
     * @param experts Original source expert axis.
     * @param layer Original model-global layer.
     * @param members Complete original transport consumer identities.
     * @return Whether members are unique, fixed and belong to this captured geometry.
     */
    inline bool validateMoEOverlaySharedDispatchGroup(
        std::uint32_t experts, std::int32_t layer,
        std::span<const MoEOverlaySharedDispatchTarget> members) noexcept
    {
        if (!experts || experts > moe_runtime_abi::kMaxExperts || layer < 0 || members.empty())
            return false;
        for (std::size_t i = 0; i < members.size(); ++i)
        {
            if (members[i].participant < 0 ||
                !members[i].membership.validFor(experts, members[i].participant, layer))
                return false;
            for (std::size_t prior = 0; prior < i; ++prior)
                if (members[prior].participant == members[i].participant)
                    return false;
        }
        return true;
    }
    /**
     * @brief Test exact IEEE signed zero independently of a GPU program's FTZ arithmetic.
     * @param weight Original finite coefficient, validated separately before compaction.
     * @return True only for the two zero encodings; nonzero subnormals remain live data.
     */
    LLAMINAR_SHARED_ROW_HD inline bool moeOverlayJointRouteWeightIsZero(float weight) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return (__float_as_uint(weight) & 0x7fffffffu) == 0u;
#else
        return (std::bit_cast<std::uint32_t>(weight) & 0x7fffffffu) == 0u;
#endif
    }
    /** @brief Complete row decision; invalid inputs never authorize a write. */
    enum class MoEOverlaySharedRowDecision : std::uint8_t
    {
        Invalid,
        Unneeded,
        Needed
    };
    /**
     * @brief Resolve one original row against every original consumer, without mutation.
     * @param launch Immutable source operand and original group descriptor.
     * @param routes Acquired original placement bank, not an independent owner table.
     * @param epoch SAME acquired placement epoch; the outer grant validates selector/generation.
     * @param row Original physical row owned by one copy CTA.
     * @return Needed exactly once if any consumer selects an original route slot.
     */
    LLAMINAR_SHARED_ROW_HD inline MoEOverlaySharedRowDecision
    selectMoEOverlaySharedRow(const MoEOverlayActivationDispatchPackLaunch &launch,
                              const std::int32_t *routes, std::uint64_t epoch,
                              std::int32_t row) noexcept
    {
        if (!routes || !launch.route_expert_ids_fp32 || !launch.route_weights || row < 0 ||
            ((launch.shared_row_controls == nullptr) != (launch.shared_row_target_count == 0u)) ||
            epoch == 0)
            return MoEOverlaySharedRowDecision::Invalid;
        const auto active =
            launch.active_row_count_device ? *launch.active_row_count_device : launch.physical_rows;
        if (active < 0 || active > launch.physical_rows)
            return MoEOverlaySharedRowDecision::Invalid;
        if (row >= active)
            return MoEOverlaySharedRowDecision::Unneeded;
        bool needed = false;
        for (int slot = 0; slot < launch.packet.top_k; ++slot)
        {
            const auto coordinate = static_cast<std::size_t>(row) * launch.packet.top_k + slot;
            const float value = launch.route_expert_ids_fp32[coordinate],
                        weight = launch.route_weights[coordinate];
            if (!(value >= 0.0f && value < launch.placement.expert_count) ||
                !(weight >= -0x1.fffffep+127F && weight <= 0x1.fffffep+127F))
                return MoEOverlaySharedRowDecision::Invalid;
            const auto expert = static_cast<std::int32_t>(value);
            if (static_cast<float>(expert) != value)
                return MoEOverlaySharedRowDecision::Invalid;
            if (MoEOverlayRouteTarget::fromWord(routes[expert]).isCPUCohort() &&
                moeOverlayJointRouteWeightIsZero(weight))
                continue;
            if (launch.shared_row_target_count == 0u)
            {
                std::int32_t selected = -1;
                if (!launch.joint_route_target.validFor(launch.placement.expert_count,
                                                        launch.target_participant_id,
                                                        launch.model_layer_index) ||
                    !launch.joint_route_target.resolve(expert, routes[expert], &selected))
                    return MoEOverlaySharedRowDecision::Invalid;
                needed = needed || selected == launch.target_participant_id;
            }
            if (launch.shared_row_target_count != 0u)
            {
                const auto target = MoEOverlayRouteTarget::fromWord(routes[expert]);
                for (std::uint32_t member = 0; member < launch.shared_row_target_count; ++member)
                {
                    const auto &channel = launch.shared_row_controls[member].channel;
                    const bool joint =
                        channel.return_role !=
                        static_cast<std::uint32_t>(MoEOverlayActivationReturnRole::Ordinary);
                    if (!moeOverlayActivationReturnRoleValid(channel.return_role) ||
                        (joint && channel.cohort_target >= -1) ||
                        (!joint && channel.cohort_target != -1))
                        return MoEOverlaySharedRowDecision::Invalid;
                    if (joint && target.isWhole() &&
                        routes[expert] == channel.target_participant_id)
                        return MoEOverlaySharedRowDecision::Invalid;
                    needed =
                        needed ||
                        (target.isWhole() && routes[expert] == channel.target_participant_id) ||
                        (joint && routes[expert] == channel.cohort_target);
                }
            }
        }
        return needed ? MoEOverlaySharedRowDecision::Needed : MoEOverlaySharedRowDecision::Unneeded;
    }
    /** @brief Write one useful element after complete-prefix validation and row selection. */
    LLAMINAR_SHARED_ROW_HD inline void
    copyMoEOverlaySharedRowElement(const MoEOverlayActivationDispatchPackLaunch &launch,
                                   std::int32_t row, std::int32_t column) noexcept
    {
        const auto index = static_cast<std::size_t>(row) * launch.packet.d_model + column;
        launch.packet.hidden_rows_fp32[index] = launch.hidden_rows_fp32[index];
    }

}
#undef LLAMINAR_SHARED_ROW_HD
