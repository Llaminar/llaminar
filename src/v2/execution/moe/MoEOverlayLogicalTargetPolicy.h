/**
 * @file MoEOverlayLogicalTargetPolicy.h
 * @brief Shared logical-target snapshot folding for device policy and its replay oracle.
 *
 * Immutable native membership determines the receipt set. A cohort contributes
 * one resident target only after every original member agrees; no member is a
 * representative owner. Global selected-route demand is independently read
 * from its original producer, including experts absent from that GPU.
 */
#pragma once
#include "DeviceMoERebalancePolicyShared.h"
#include "DeviceMoERuntimeABI.h"
#include "MoEOverlayDeviceControllerFabricABI.h"
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_LOGICAL_POLICY_HD __host__ __device__
#else
#define LLAMINAR_LOGICAL_POLICY_HD
#endif
namespace llaminar2
{
    /** @brief Complete original-member authority decision for one logical target. */
    struct MoEOverlayLogicalTargetSnapshot
    {
        bool valid = false;
        bool authoritative = false;
    };

    /**
     * @brief Fold exact original members without multiplying a logical resident.
     * @param cohort Whether the original target is a cohort, never inferred from a payload.
     * @param members Exact original admitted member extent.
     * @param read_word Returns each member's release-published residency plane word.
     * @return Invalid for partial authority, missing physical state or malformed membership.
     */
    template <class ReadWord>
    LLAMINAR_LOGICAL_POLICY_HD inline MoEOverlayLogicalTargetSnapshot moeOverlayFoldLogicalTargetSnapshot(
        bool cohort, std::uint32_t members, ReadWord read_word) noexcept
    {
        // GPU TP1 remains a true complete-F cohort. Whole targets still
        // require exactly one owner; metadata never infers kind from width.
        if (!members || (!cohort && members != 1u)) return {};
        bool selected = false;
        for (std::uint32_t member = 0u; member < members; ++member)
        {
            const auto word = read_word(member);
            const bool owner = moe_rebalance_policy::collectedStateAuthoritativeOwner(word);
            const bool physical = moe_rebalance_policy::collectedStatePhysicallyResident(word);
            if ((owner && !physical) || (cohort && owner != physical) || (member && owner != selected))
                return {};
            selected = owner;
        }
        return {.valid = true, .authoritative = selected};
    }

    /**
     * @brief Borrow the original selected-route counter independently of local execution.
     * @param bank Same runtime-owned recurring histogram bank selected by its device writer state.
     * @param phase Original decode/prefill/accepted-verifier source. @param expert Global logical expert.
     * @return Selected counter address, or null for a foreign source coordinate.
     * @note The caller reads this counter atomically on GPU. No local histogram
     *       or resident descriptor is an input to this source selection.
     */
    LLAMINAR_LOGICAL_POLICY_HD inline const std::uint64_t *moeOverlayGlobalSelectedCounter(
        const moe_runtime_abi::DeviceMoERuntimeHistogramBank &bank,
        std::uint32_t phase, std::uint32_t expert) noexcept
    {
        return phase < moe_runtime_abi::kHistogramSourceCount && expert < moe_runtime_abi::kHistogramMaxExperts
            ? bank.selected[phase] + expert : nullptr;
    }

    /** @brief One cumulative selected-counter observation and its reset-safe consumed delta. */
    struct MoEOverlaySelectedDemandDelta
    {
        std::uint64_t delta = 0u;
        std::uint64_t next_baseline = 0u;
    };

    /**
     * @brief Consume the original global selected counter independently of residency.
     * @param selected Global selected-route count for exactly one recurring phase/expert.
     * @param previous Same producer's device-owned prior cumulative count.
     * @return Exact delta; a lower cumulative value starts a fresh counter generation.
     */
    LLAMINAR_LOGICAL_POLICY_HD constexpr MoEOverlaySelectedDemandDelta moeOverlaySelectedDemandDelta(
        std::uint64_t selected, std::uint64_t previous) noexcept
    {
        return {selected >= previous ? selected - previous : selected, selected};
    }
}
#undef LLAMINAR_LOGICAL_POLICY_HD
