/**
 * @file MTPParityComparisonPlacement.h
 * @brief Dispatch one serial/verifier comparison through its admitted placement owner.
 *
 * The scope covers only the existing comparison callback. It neither observes
 * device placement nor suppresses maintenance between independent comparisons.
 * Runner-owned support checks remain authoritative: an unavailable scope fails
 * instead of retrying the callback without placement protection.
 */
#pragma once

#include "execution/moe/MoEExpertOwnerMap.h"
#include "config/OrchestrationConfig.h"

#include <functional>
#include <stdexcept>

namespace llaminar2::test::parity
{
    /** Original authority required by one comparison; this does not grant native support. */
    enum class MTPParityComparisonPlacementOwner { Unscoped, CPUJoint, Device };

    /**
     * @brief Resolve comparison ownership from the same frozen runtime plan.
     * @param runtime Admitted configuration, never sampled live placement.
     * @return Original owner; unchanged non-moving/CPU-apportioned paths are unscoped.
     * @throws std::logic_error For missing or contradictory frozen authority.
     */
    inline MTPParityComparisonPlacementOwner mtpParityComparisonPlacementOwner(
        const OrchestrationConfig &runtime)
    {
        const auto &plan = runtime.moe_routed_expert_plan;
        if (runtime.moe_rebalance.mode != MoERebalanceRuntimeMode::Dynamic ||
            !plan || !plan->usesExpertOverlayAuthority())
            return MTPParityComparisonPlacementOwner::Unscoped;
        const auto authority = resolveMoEOverlayAuthorityExecutionKind(*plan);
        if (authority == MoEOverlayAuthorityExecutionKind::Unresolved ||
            authority != plan->authority_execution)
            throw std::logic_error("MTP comparison requires the original frozen placement authority");
        if (MoEExpertOwnerMap::usesCPUJointFragments(*plan))
            return MTPParityComparisonPlacementOwner::CPUJoint;
        if (authority == MoEOverlayAuthorityExecutionKind::DeviceResident)
            return MTPParityComparisonPlacementOwner::Device;
        return MTPParityComparisonPlacementOwner::Unscoped;
    }

    /**
     * @brief Execute an unchanged comparison under the original placement authority.
     * @tparam Runner Production orchestration runner or device-free dispatch fixture.
     * @param runner Original admitted runner; its configuration supplies authority identity.
     * @param body Existing reset, serial request and verifier comparison, invoked once.
     * @throws std::exception For inconsistent admission, unsupported native scope,
     *         or the original callback failure. No alternate execution is attempted.
     */
    template<class Runner>
    void withMTPParityComparisonPlacement(Runner &runner, const std::function<void()> &body)
    {
        if (!body)
            throw std::invalid_argument("MTP comparison requires its original callback");
        switch (mtpParityComparisonPlacementOwner(runner.config()))
        {
        case MTPParityComparisonPlacementOwner::CPUJoint:
            runner.withMoEOverlayInferencePlacementScope(body);
            return;
        case MTPParityComparisonPlacementOwner::Device:
            runner.withMoEOverlayDevicePlacementScope(body);
            return;
        case MTPParityComparisonPlacementOwner::Unscoped:
            body();
            return;
        }
        throw std::logic_error("MTP comparison has an unknown placement owner");
    }
} // namespace llaminar2::test::parity
