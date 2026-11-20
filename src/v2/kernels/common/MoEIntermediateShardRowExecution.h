/**
 * @file MoEIntermediateShardRowExecution.h
 * @brief Shared CUDA/HIP selection of original local F fragments from an acquired bank.
 *
 * Cohort membership is the acquired bank's encoded route target. The immutable
 * binding contributes only original member/source geometry, never current expert
 * ownership. Whole-expert replica masks cannot represent a fragment cohort:
 * fragment selection requires zero replica metadata and a complete local payload.
 */
#pragma once

#include "execution/moe/DeviceMoEIntermediateShardBinding.h"
#include "DeviceMoEFloatingMatrixDesc.h"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_FRAGMENT_ROW_HD __host__ __device__
#else
#define LLAMINAR_MOE_FRAGMENT_ROW_HD
#endif

namespace llaminar2
{
    /** @brief A selected local partial, a legitimate remote route, or fatal missing authority. */
    enum class MoEIntermediateShardRowDisposition : std::uint8_t { Remote, Local, Invalid };

    /** @brief The existing local compute-assignment view; never a logical whole-owner publication. */
    struct MoEIntermediateShardRowSelection
    {
        MoEIntermediateShardRowDisposition disposition = MoEIntermediateShardRowDisposition::Invalid;
        std::int32_t participant = -1;
    };

    /**
     * @brief Authenticate one original fragment's live acquired route and membership.
     * @tparam Bank Host or ABI-identical native placement-bank view.
     * @param binding Closed original family-derived geometry retained by the captured runtime.
     * @param bank SAME acquired epoch containing route targets and prepared descriptors.
     * @param expert Original logical expert ID selected by routing.
     * @param participant Validated runtime-local member coordinate; the binding separately retains the original global ID.
     * @return Local for this member, Remote for another target, Invalid for broken ownership.
     * @details Published consumers rely on the bank producer for immutable geometry
     * and payload validity. Raw descriptor consumers also call the checked selector below.
     */
    template <typename Bank>
    [[nodiscard]] LLAMINAR_MOE_FRAGMENT_ROW_HD constexpr MoEIntermediateShardRowSelection
    selectMoEIntermediateShardMembership(const DeviceMoEIntermediateShardBinding &binding,
        const Bank &bank, std::int32_t expert, std::int32_t participant) noexcept
    {
        if (!binding.enabled() || participant < 0 || expert < 0 || static_cast<std::uint32_t>(expert) >= bank.expert_count)
            return {};
        std::int32_t resolved = -1;
        const auto word = bank.overlay_route_participant[expert];
        if (MoEOverlayRouteTarget::fromWord(word).isAbsent() ||
            !binding.target().resolve(expert, word, &resolved)) return {};
        if (word != binding.target().cohortTarget())
            return bank.local_compute_mask[expert] == 0u
                ? MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Remote, -1}
                : MoEIntermediateShardRowSelection{};
        const auto &descriptor = bank.experts[expert];
        if (resolved != binding.target().participantId() || !bank.local_compute_mask[expert] ||
            bank.resident_participant_mask[expert] != 0u || bank.replica_role[expert] != 0u ||
            descriptor.logical_expert_id != expert || descriptor.local_slot < 0 ||
            descriptor.owner_participant != -1 ||
            descriptor.projection_set != DeviceMoEProjectionSet::IntermediateShard ||
            (descriptor.weight_format != DeviceMoEWeightFormat::NativeQuantised &&
                !deviceMoEWeightFormatIsFloating(descriptor.weight_format)) ||
            !deviceMoEIntermediateShardFlagsValid(descriptor.flags))
            return {};
        return {MoEIntermediateShardRowDisposition::Local, participant};
    }

    /**
     * @brief Preserve the checked direct-descriptor API after authenticating membership.
     * @tparam Bank Host or ABI-identical native placement-bank view.
     * @param binding Original retained geometry. @param bank SAME acquired bank.
     * @param expert Actual selected expert. @param participant Runtime-local participant.
     * @return Original selection including complete physical and shape validation.
     */
    template <typename Bank>
    [[nodiscard]] LLAMINAR_MOE_FRAGMENT_ROW_HD constexpr MoEIntermediateShardRowSelection
    selectMoEIntermediateShardRow(const DeviceMoEIntermediateShardBinding &binding,
        const Bank &bank, std::int32_t expert, std::int32_t participant) noexcept
    {
        const auto membership=selectMoEIntermediateShardMembership(binding,bank,expert,participant);
        if(membership.disposition!=MoEIntermediateShardRowDisposition::Local)return membership;
        const auto& descriptor=bank.experts[expert];
        if(!deviceMoEExpertProjectionReady(descriptor,DeviceMoEProjectionSet::IntermediateShard))return {};
        const auto h = static_cast<std::int32_t>(binding.modelColumns());
        const auto f = static_cast<std::int32_t>(binding.localIntermediateColumns());
        bool geometry;
        if (deviceMoEWeightFormatIsFloating(descriptor.weight_format))
            geometry = descriptor.floating_gate.n == f && descriptor.floating_gate.k == h &&
                descriptor.floating_up.n == f && descriptor.floating_up.k == h &&
                descriptor.floating_down.n == h && descriptor.floating_down.k == f;
        else
            // These scalar fields belong to the same already-bounded, ready
            // candidate row. Group independent comparisons after the original
            // ownership/readiness and divisibility gates; never chase planes.
            geometry = h % 32 == 0 && f % 32 == 0 &&
                ((descriptor.gate.n == f) & (descriptor.gate.k == h) &
                 (descriptor.up.n == f) & (descriptor.up.k == h) &
                 (descriptor.down.n == h) & (descriptor.down.k == f) &
                 (descriptor.gate.blocks_per_row == static_cast<std::uint32_t>(h / 32)) &
                 (descriptor.up.blocks_per_row == static_cast<std::uint32_t>(h / 32)) &
                 (descriptor.down.blocks_per_row == static_cast<std::uint32_t>(f / 32)));
        return geometry ? MoEIntermediateShardRowSelection{MoEIntermediateShardRowDisposition::Local, participant}
                        : MoEIntermediateShardRowSelection{};
    }
} // namespace llaminar2
#undef LLAMINAR_MOE_FRAGMENT_ROW_HD
