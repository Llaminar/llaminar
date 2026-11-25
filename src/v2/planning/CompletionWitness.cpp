/**
 * @file CompletionWitness.cpp
 * @brief Bounded deterministic completion-order decision for page frames.
 *
 * The algorithm is deliberately small and total: it tries one completion
 * order (declared request order) and checks phase arithmetic exactly.
 * A request may advance to its resident peak only from frames that are free
 * now or become free when earlier requests in that order finish. A preserved
 * request keeps its current extent and never frees frames in this witness.
 * Anything unprovable is a typed refusal, never a guessed progress step.
 * This is the page-frame subset of the complete all-resource/topology witness
 * (design §5.3, plan P0/P5); callers must compose the remaining classes.
 */
#include "CompletionWitness.h"
#include <algorithm>
#include <limits>

namespace llaminar2::planning
{
bool phaseFits(std::size_t frames, std::size_t occupied, std::size_t needed)
{
    if (needed > frames || occupied > frames) return false;
    // Checked subtraction form: no addition overflow, no underflow.
    return needed <= frames - occupied;
}

CompletionWitness proveCompletionOrder(std::size_t frames, std::span<const WitnessRequest> requests)
{
    CompletionWitness witness;
    if (!frames)
    {
        witness.refusal = "zero materialized frames";
        return witness;
    }
    if (requests.empty())
    {
        witness.steps.push_back({WitnessStep::Kind::FinalRetire, 0, 0, 0});
        return witness;
    }
    std::size_t total_occupied = 0;
    for (const auto &request : requests)
    {
        if (!request.peak_frames || request.peak_frames > frames)
        {
            witness.refusal = "declared request cannot fit any single resident phase";
            return witness;
        }
        if (request.occupied_frames > request.peak_frames)
        {
            witness.refusal = "occupied frames exceed the declared request peak";
            return witness;
        }
        if (request.occupied_frames > std::numeric_limits<std::size_t>::max() - total_occupied)
        {
            witness.refusal = "occupied frame total overflows";
            return witness;
        }
        total_occupied += request.occupied_frames;
    }
    if (total_occupied > frames)
    {
        witness.refusal = "currently occupied frames exceed the materialized pool";
        return witness;
    }
    std::size_t free_now = frames - total_occupied; // Frames not occupied by any live request.
    std::size_t freed = 0;                           // Frames released by completed requests.
    for (std::size_t index = 0; index < requests.size(); ++index)
    {
        const auto &request = requests[index];
        if (request.preserved)
        {
            // Preserved requests continue at their current extent; the witness
            // does not require them to reach their peak now, and they do not
            // release frames. Their occupancy remains charged.
            witness.steps.push_back({WitnessStep::Kind::Preserve, request.generation,
                request.occupied_frames, 0});
            continue;
        }
        const std::size_t demand = request.peak_frames - request.occupied_frames;
        // Capacity available now plus frames released by earlier completions.
        // Checked form: no overflow in the sum, no underflow in the difference.
        const bool fits = demand <= free_now || demand - free_now <= freed;
        if (!fits)
        {
            witness.refusal = "no completion order: request peak exceeds capacity available after prior completions";
            witness.steps.clear(); // Fail closed: no usable partial plan after refusal.
            return witness;
        }
        witness.steps.push_back({WitnessStep::Kind::Advance, request.generation,
            demand, 0});
        // Completing the request releases its occupied frames for later ones.
        freed += request.occupied_frames;
        if (freed > frames) freed = frames; // Guard against arithmetic error.
        witness.steps.push_back({WitnessStep::Kind::FinalRetire, request.generation,
            0, request.occupied_frames});
    }
    return witness;
}
} // namespace llaminar2::planning
