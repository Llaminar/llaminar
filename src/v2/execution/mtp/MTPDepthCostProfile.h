/**
 * @file MTPDepthCostProfile.h
 * @brief Immutable original-TRAIN transaction costs for one measured MTP scope.
 *
 * Costs are ceil(median(native decode_us / physical verifier runs)) over all
 * original five TRAIN prompts and three repetitions at every fixed depth. The
 * evidence lives in qfn-mtp-original-train-transaction-cost-profile-r1-private,
 * profile SHA256 131804c1b3537a0c899e725f31730800dcfe12920927a50fd38032aae3b53976.
 * These are finite n256/C4096 estimates, not clocks or an identical-text utility
 * guarantee. Scalar switches avoid local device arrays and unadmitted storage.
 */
#pragma once
#include "MTPDepthLearnedPolicy.h"
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_COST_HD __host__ __device__ __forceinline__
#else
#define LLAMINAR_MTP_COST_HD inline
#endif
namespace llaminar2
{
/** @brief A complete measured law, or absence of a measured cost authority. */
enum class MTPDepthCostProfileKind : uint32_t { None, Greedy, Stochastic };

/** @brief A borrowed immutable profile selected before consuming any utility. */
struct MTPDepthCostProfile
{
    MTPDepthCostProfileKind kind = MTPDepthCostProfileKind::None;
    /** @return Whether this exact measured scope owns positive transaction costs. */
    LLAMINAR_MTP_COST_HD constexpr bool active() const noexcept
    { return kind != MTPDepthCostProfileKind::None; }
    /** @param depth Positive speculative depth. @return Integer microseconds, zero if unavailable. */
    LLAMINAR_MTP_COST_HD constexpr uint32_t costAt(uint32_t depth) const noexcept
    {
        if (kind == MTPDepthCostProfileKind::Greedy)
        {
            switch (depth)
            {
            case 1:return 33853; case 2:return 38657; case 3:return 44485;
            case 4:return 50678; case 5:return 54652; case 6:return 59379;
            case 7:return 64089; case 8:return 72849; case 9:return 76018;
            case 10:return 81652; case 11:return 85536; case 12:return 90843;
            case 13:return 95966; case 14:return 101074; case 15:return 104359;
            default:return 0;
            }
        }
        if (kind == MTPDepthCostProfileKind::Stochastic)
        {
            switch (depth)
            {
            case 1:return 34022; case 2:return 38758; case 3:return 44734;
            case 4:return 50902; case 5:return 54971; case 6:return 59622;
            case 7:return 64427; case 8:return 73224; case 9:return 76427;
            case 10:return 82064; case 11:return 86158; case 12:return 91538;
            case 13:return 96474; case 14:return 101630; case 15:return 104715;
            default:return 0;
            }
        }
        return 0;
    }
};

/**
 * @brief Select only the complete original ROCm/QFN measured context and law.
 * @param context Canonical immutable backend/model/scope/law plus enabled intent.
 * @return A scalar immutable profile; no generic or partially matching profile.
 * Bounds remain an admission responsibility: an active profile never substitutes
 * a legacy policy to hide an unavailable depth-zero cost.
 */
LLAMINAR_MTP_COST_HD constexpr MTPDepthCostProfile resolveMTPDepthCostProfile(
    const MTPLearnedDepthPolicyContext &context) noexcept
{
    const auto &s=context.scope;
    if (!context.enabled || context.backend!=MTPDepthPolicyBackend::ROCm ||
        context.model_class!=MTPDepthPolicyModelClass::MoE ||
        s.verifier_width_policy!=MTPVerifierPhysicalWidthPolicy::ExactLogicalRows ||
        s.tp_width!=4 || s.pp_stages!=1 || s.retained_depth!=15 ||
        s.hidden_dim!=2560 || s.main_layers!=48 || s.expert_count!=512 ||
        s.experts_per_token!=10 || s.compute_capability_major!=90 || s.compute_capability_minor!=6)
        return {};
    if (context.verify_mode==MTPVerifyMode::Greedy)
        return {MTPDepthCostProfileKind::Greedy};
    if (context.verify_mode==MTPVerifyMode::SpeculativeSampling)
        return {MTPDepthCostProfileKind::Stochastic};
    return {};
}
}
#undef LLAMINAR_MTP_COST_HD
