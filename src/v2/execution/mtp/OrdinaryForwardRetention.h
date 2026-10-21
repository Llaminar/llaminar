/**
 * @file OrdinaryForwardRetention.h
 * @brief Explicit ownership of one complete retained ordinary forward.
 *
 * A host-serviced ExpertOverlay body is a retained transaction, not one native
 * child graph. The caller chooses the admitted contract before materialization;
 * failure of a native export can never select hosted execution.
 */
#pragma once
#include "execution/local_execution/orchestrators/IInferenceRunner.h"
#include <stdexcept>
namespace llaminar2
{
/** @brief Immutable execution domain, authenticated by the graph's overlay/epoch owner. */
enum class OrdinaryForwardDomain : uint8_t { ParticipantLocal, HostedExpertOverlay };
/** @brief The exact export a consumer is capable of retaining. */
enum class OrdinaryForwardRetentionKind : uint8_t { NativeComposable, HostedExpertOverlay };
/**
 * @brief Resolve export ownership from the existing serving policy and domain.
 * @param execution Canonical complete generation execution policy.
 * @param domain Admitted local or explicitly host-serviced ExpertOverlay domain.
 * @return Hosted only for an explicitly admitted hosted ExpertOverlay body.
 * @throws std::invalid_argument For unsupported or contradictory policies.
 */
[[nodiscard]] inline OrdinaryForwardRetentionKind resolveOrdinaryForwardRetention(
    DeviceGenerationExecutionPolicy execution, OrdinaryForwardDomain domain)
{
    switch (domain)
    {
    case OrdinaryForwardDomain::ParticipantLocal:
        if (execution == DeviceGenerationExecutionPolicy::NativeConditionalGraph ||
            execution == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
            return OrdinaryForwardRetentionKind::NativeComposable;
        break;
    case OrdinaryForwardDomain::HostedExpertOverlay:
        if (execution == DeviceGenerationExecutionPolicy::HostScheduledCapturedTransactions)
            return OrdinaryForwardRetentionKind::HostedExpertOverlay;
        break;
    default:
        throw std::invalid_argument("Unknown ordinary forward execution domain");
    }
    throw std::invalid_argument("Ordinary forward domain contradicts its complete generation execution policy");
}
}
