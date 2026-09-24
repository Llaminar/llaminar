/**
 * @file MoEOverlayServiceTelemetryPublication.cpp
 * @brief Seqlock acquisition for mapped GPU service telemetry.
 */

#include "MoEOverlayServiceTelemetryPublication.h"

#include <atomic>
#include <cstddef>

namespace llaminar2
{
    namespace
    {
        /** @brief Acquire one trivially-copyable mapped scalar. */
        template <typename T>
        T loadAcquire(T &value) noexcept
        {
            return std::atomic_ref<T>(value).load(
                std::memory_order_acquire);
        }
    } // namespace

    bool trySnapshotMoEOverlayServiceTelemetryPublication(
        MoEOverlayDeviceServiceTelemetryPublicationHeader *publication,
        int expected_participant_id,
        std::uint32_t expected_layer_count,
        std::vector<MoEOverlayParticipantLayerServiceTotals> *output,
        std::uint64_t *generation) noexcept
    {
        static_assert(
            kExpertHistogramProductionSourceCount ==
                kDeviceMoEOverlayServicePhaseCount,
            "host economy phases must match the device telemetry ABI");

        if (generation)
            *generation = 0u;
        if (!output)
            return false;
        output->clear();

        try
        {
            if (!publication || expected_participant_id < 0 ||
                expected_layer_count == 0u)
            {
                return false;
            }

            const std::uint64_t before = loadAcquire(
                publication->generation);
            if (before == 0u || (before & 1u) != 0u ||
                publication->magic !=
                    kDeviceMoEOverlayServiceTelemetryMagic ||
                publication->version !=
                    kDeviceMoEOverlayServiceTelemetryVersion ||
                publication->participant_id != expected_participant_id ||
                publication->layer_count != expected_layer_count ||
                publication->phase_count !=
                    kDeviceMoEOverlayServicePhaseCount ||
                publication->valid_sample_count > expected_layer_count ||
                publication->armed_sample_count > expected_layer_count ||
                publication->begun_sample_count > expected_layer_count)
            {
                return false;
            }

            const auto *const cells =
                deviceMoEOverlayServiceTelemetryCells(publication);
            output->resize(expected_layer_count);
            for (std::uint32_t layer = 0u;
                 layer < expected_layer_count;
                 ++layer)
            {
                auto &row = (*output)[layer];
                row.participant_id = expected_participant_id;
                row.layer = static_cast<int>(layer);
                for (std::size_t phase = 0u;
                     phase < kDeviceMoEOverlayServicePhaseCount;
                     ++phase)
                {
                    const auto &cell = cells[
                        static_cast<std::size_t>(layer) *
                            kDeviceMoEOverlayServicePhaseCount +
                        phase];
                    row.total_nanoseconds[phase] =
                        cell.total_nanoseconds;
                    row.activation_count[phase] = cell.activation_count;
                    row.sample_count[phase] = cell.sample_count;
                    row.overflowed[phase] = cell.overflowed != 0u;
                }
            }

            const std::uint64_t after = loadAcquire(
                publication->generation);
            if (after != before || (after & 1u) != 0u)
            {
                output->clear();
                return false;
            }
            if (generation)
                *generation = after;
            return true;
        }
        catch (...)
        {
            output->clear();
            if (generation)
                *generation = 0u;
            return false;
        }
    }
} // namespace llaminar2
