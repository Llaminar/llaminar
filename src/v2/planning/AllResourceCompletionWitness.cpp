/**
 * @file AllResourceCompletionWitness.cpp
 * @brief All-resource completion witness implementation.
 */
#include "AllResourceCompletionWitness.h"
#include "planning/CompletionWitness.h"
#include <limits>

namespace llaminar2
{
AllResourceWitnessResult proveAllResourceCompletion(
    const AllResourceCapacity &capacity)
{
    if (!capacity.frames || !capacity.banks || !capacity.lease_cells)
        return AllResourceWitnessResult{false,
            "all-resource witness requires positive admitted class maxima"};
    // 1) Frame proof (identical arithmetic to the pool's own boundary proof).
    std::vector<planning::WitnessRequest> frame_requests;
    frame_requests.reserve(capacity.requests.size());
    for (const auto &r : capacity.requests)
        frame_requests.push_back(planning::WitnessRequest{
            .generation=r.generation, .record=r.record,
            .peak_frames=r.peak_frames, .occupied_frames=r.occupied_frames,
            .preserved=r.preserved});
    const auto frame_proof = planning::proveCompletionOrder(
        capacity.frames, capacity.base,
        std::span<const planning::WitnessRequest>(frame_requests));
    if (!frame_proof.valid())
        return AllResourceWitnessResult{false, frame_proof.refusal};
    // 2) Table-bank + lease-cell maxima together. At any phase, the requests
    // that cannot yet be preserved (preserved == false) must fit the admitted
    // banks and lease cells; a preservation-eligible request may release its
    // banks/leases only AFTER its frame preservation is proved. We check the
    // conservative simultaneous demand: the worst case over a completion
    // order where preserved requests retire first and restored ones re-enter.
    std::size_t banks_demand = 0;
    std::size_t leases_demand = 0;
    for (const auto &r : capacity.requests)
    {
        if (r.peak_banks > std::numeric_limits<std::size_t>::max() - banks_demand ||
            r.peak_leases > std::numeric_limits<std::size_t>::max() - leases_demand)
            return AllResourceWitnessResult{false,
                "all-resource witness class demand overflows"};
        banks_demand += r.peak_banks;
        leases_demand += r.peak_leases;
    }
    if (banks_demand > capacity.banks)
        return AllResourceWitnessResult{false,
            "table-bank peak demand exceeds the admitted bank count"};
    if (leases_demand > capacity.lease_cells)
        return AllResourceWitnessResult{false,
            "effective-use lease peak demand exceeds the admitted lease cells"};
    // A resource required to evacuate an owner cannot itself be obtainable
    // only by evacuating that owner: after the FIRST non-preserved request
    // finishes, every other non-preserved request must still fit concurrently
    // minus the class its own completion releases — proven by the bound above
    // (each request's own peak already includes its simultaneous occupancy).
    return AllResourceWitnessResult{true, {}};
}
} // namespace llaminar2
