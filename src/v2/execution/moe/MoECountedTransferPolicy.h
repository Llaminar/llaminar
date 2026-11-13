/**
 * @file MoECountedTransferPolicy.h
 * @brief Authenticate the graph-authored phase of the original native counted collective.
 *
 * Prefill, decode and grouped verification all use the first-class original
 * NCCL/RCCL composition. Their phase remains part of semantic capture identity;
 * no legacy endpoint batching mode is selected or advertised here.
 */
#pragma once
#include "DeviceMoEOverlayServiceTelemetry.h"
#include <stdexcept>

namespace llaminar2
{
    /** @return Accept an explicit graph phase without selecting another transport or launch mode.
     * @param phase Model-authored semantic phase frozen before graph construction.
     * @throws std::invalid_argument For Auto or an undeclared phase. */
    constexpr void requireMoENativeCountedPhase(
        MoEOverlayServicePhaseHint phase)
    {
        switch (phase)
        {
        case MoEOverlayServicePhaseHint::Decode:
        case MoEOverlayServicePhaseHint::Prefill:
        case MoEOverlayServicePhaseHint::GroupedVerifier:
            return;
        case MoEOverlayServicePhaseHint::Auto: break;
        }
        throw std::invalid_argument("Counted MoE transfer requires an explicit immutable graph phase");
    }
}
