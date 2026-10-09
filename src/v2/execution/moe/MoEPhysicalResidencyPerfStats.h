/**
 * @file MoEPhysicalResidencyPerfStats.h
 * @brief Physical-residency lifecycle evidence with topology-bounded storage.
 *
 * The fabric publishes these observations only after its authoritative slot
 * transition succeeds. Transaction and epoch numbers are evidence, never new
 * statistics keys. No prepared weights, cache payloads or device state are read.
 */
#pragma once

#include "utils/PerfStatsCollector.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    /** @brief Successful physical transitions with distinct counter semantics. */
    enum class MoEPhysicalWaveEvent
    {
        DestinationsStaged,
        EpochPublished,
        SourcesRetired,
        WaveAborted,
    };

    /**
     * @brief Record one completed physical wave transition.
     * @param event Exact transition that the physical slot authority completed.
     * @param transaction Authenticated command identity.
     * @param base_epoch Previous durable physical epoch.
     * @param candidate_epoch Epoch named by the same command.
     * @param slots Exact staged migration or retired local-slot count; ignored
     *        for publication and abort, whose counter unit is one transition.
     * @param device Stable owning participant/topology label.
     * @throws std::invalid_argument for an unknown event discriminator.
     */
    inline void recordMoEPhysicalWave(
        MoEPhysicalWaveEvent event, std::uint64_t transaction,
        std::uint64_t base_epoch, std::uint64_t candidate_epoch,
        std::uint64_t slots, const std::string &device)
    {
        const char *name = nullptr;
        switch (event)
        {
        case MoEPhysicalWaveEvent::DestinationsStaged:
            name = "device_physical_destinations_staged";
            break;
        case MoEPhysicalWaveEvent::EpochPublished:
            name = "device_physical_epoch_published";
            slots = 1;
            break;
        case MoEPhysicalWaveEvent::SourcesRetired:
            name = "device_physical_sources_retired";
            break;
        case MoEPhysicalWaveEvent::WaveAborted:
            name = "device_physical_wave_aborted";
            slots = 1;
            break;
        default:
            throw std::invalid_argument("Unknown physical residency event");
        }
        // Preserve exact large identities and work extents as constant-size
        // numeric evidence. The collector owns no per-transaction map nodes.
        PerfStatsCollector::addCounterWithSequence("moe_overlay_residency",
            name, static_cast<double>(slots), {transaction, base_epoch, candidate_epoch, slots},
            "maintenance", device);
    }

    /**
     * @brief Record loader-owned slots becoming recyclable after reader retirement.
     * @param epoch Retired physical epoch.
     * @param slots Exact number of initial allocations returned to their arena.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEBootstrapSlotsRecycled(
        std::uint64_t epoch, std::uint64_t slots, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_residency",
            "bootstrap_live_slots_recycled", static_cast<double>(slots), {epoch, slots},
            "maintenance", device);
    }

    /**
     * @brief Record a completed physical seal retained by a reusable model context.
     * @param epoch Exact quiescent physical epoch.
     * @param canonical_experts Retained canonical prepared experts.
     * @param compacted_experts Shadow experts compacted by the successful seal.
     * @param device Stable owning participant/topology label.
     */
    inline void recordMoEReusableContextSeal(
        std::uint64_t epoch, std::uint64_t canonical_experts,
        std::uint64_t compacted_experts, const std::string &device)
    {
        PerfStatsCollector::addCounterWithSequence("moe_overlay_residency",
            "reusable_context_physical_seals", 1.0, {epoch, canonical_experts, compacted_experts},
            "model_teardown", device);
    }
}
