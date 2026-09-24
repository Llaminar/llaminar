/**
 * @file MoEOverlayServiceTelemetryPublication.h
 * @brief Coherent host acquisition of finite GPU service publications.
 *
 * CUDA and ROCm publish cumulative routed-expert service totals into the same
 * backend-neutral mapped-page ABI. This helper is the sole host decoder for
 * that ABI, shared by both the all-GPU device-controller fabric and the
 * host-authority observation service. It reads no device allocation directly
 * and never performs a transfer or synchronization.
 */

#pragma once

#include "DeviceMoEOverlayServiceTelemetry.h"
#include "MoEOverlayParticipantResidency.h"

#include <cstdint>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Acquire one unchanged, complete mapped telemetry generation.
     *
     * The producer writes an odd generation before changing cells and an even
     * generation after a system-release fence. This reader accepts the data
     * only when the same nonzero even generation brackets the complete copy.
     * A false return is ordinary publication overlap or malformed geometry;
     * callers may retry without delaying inference.
     *
     * @param publication Stable host alias of the mapped publication record.
     * @param expected_participant_id Immutable global participant identity.
     * @param expected_layer_count Exact model layer count.
     * @param output Receives one layer-ordered cumulative row per layer.
     * @param generation Optional accepted even publication generation.
     * @return True only for one coherent, identity-matching generation.
     */
    [[nodiscard]] bool trySnapshotMoEOverlayServiceTelemetryPublication(
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication,
        int expected_participant_id,
        std::uint32_t expected_layer_count,
        std::vector<MoEOverlayParticipantLayerServiceTotals> *output,
        std::uint64_t *generation = nullptr) noexcept;
} // namespace llaminar2
