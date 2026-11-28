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
    const AllResourceCapacity &capacity,
    std::span<planning::WitnessRequest> frame_workspace)
{
    if (!capacity.frames || !capacity.banks || !capacity.lease_cells)
        return AllResourceWitnessResult{false,
            "all-resource witness requires positive admitted class maxima"};
    if (frame_workspace.size() < capacity.requests.size())
        return AllResourceWitnessResult{false,
            "all-resource witness frame workspace is smaller than the request membership"};
    // 1) Frame proof through the caller's bounded workspace (ZERO per-call heap
    //    allocation; the composed witness follows the supplied-workspace rule
    //    of the frame proof exactly).
    for (std::size_t i = 0; i < capacity.requests.size(); ++i)
    {
        const auto &r = capacity.requests[i];
        frame_workspace[i] = planning::WitnessRequest{
            .generation=r.generation, .record=r.record,
            .peak_frames=r.peak_frames, .occupied_frames=r.occupied_frames,
            .preserved=r.preserved};
    }
    const auto frame_proof = planning::proveCompletionOrder(
        capacity.frames, capacity.base,
        frame_workspace.first(capacity.requests.size()));
    if (!frame_proof.valid())
        return AllResourceWitnessResult{false, frame_proof.refusal};
    // 2) Table-bank + lease-cell maxima together (B4): observed live demand
    // plus the proposed request's DECLARED peaks must fit the admitted class
    // maxima. The observation comes from the authority itself (liveClassDemand),
    // never a per-request assumption; the existing live requests already hold
    // their own banks/leases and release them through retirement.
    std::size_t banks_demand = capacity.live_banks;
    std::size_t leases_demand = capacity.live_leases;
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
