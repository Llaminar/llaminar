/**
 * @file MoECountedTransferPolicy.h
 * @brief Resolve model-authored immutable phase into counted-transfer scheduling.
 *
 * Ordinary decode retains the measured endpoint grouping. Prefill and grouped
 * verification retain original endpoint scheduling; grouped verification is an
 * explicit conservative, currently untuned policy, not a decode-size inference.
 * Call only for declared counted transport. Native/P2P paths need no resolution.
 */
#pragma once
#include "DeviceMoEOverlayServiceTelemetry.h"
#include "transfer/CapturedTransferEndpointScheduling.h"

namespace llaminar2
{
    /** @return Exact graph-phase scheduling; no row/count/DebugEnv inference.
     * @param phase Model-authored semantic phase frozen before graph construction.
     * @throws std::invalid_argument For Auto or an undeclared phase. */
    [[nodiscard]] constexpr CapturedTransferEndpointScheduling moeCountedTransferScheduling(
        MoEOverlayServicePhaseHint phase)
    {
        switch (phase)
        {
        case MoEOverlayServicePhaseHint::Decode:
            return CapturedTransferEndpointScheduling::Batched;
        case MoEOverlayServicePhaseHint::Prefill:
        case MoEOverlayServicePhaseHint::GroupedVerifier:
            return CapturedTransferEndpointScheduling::Individual;
        case MoEOverlayServicePhaseHint::Auto: break;
        }
        throw std::invalid_argument("Counted MoE transfer requires an explicit immutable graph phase");
    }
}
