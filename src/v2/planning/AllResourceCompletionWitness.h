/**
 * @file AllResourceCompletionWitness.h
 * @brief Composed completion witness over frame + table-bank + lease classes.
 *
 * Plan §11/P1: a bounded optimistic admission must prove a finite completion
 * order across EVERY resource class at once — protected page frames, table
 * banks, and effective-use lease cells — before any class is acquired. This
 * composition runs the per-frame completion proof AND checks the other classes'
 * exact maxima together; a refusal leaves every class untouched (the caller
 * acquires nothing). It is a PURE planner input: PMA remains the sole physical
 * authority and TransferEngine owns movement.
 *
 * The composer is deliberately NOT a second ledger: it holds no live counter
 * and never allocates in the hot path; the caller reuses the pool's own
 * boundary proof and supplies the admitted class maxima.
 */
#pragma once
#include "planning/CompletionWitness.h"
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
/** @brief One request's admitted demand across all classes. */
struct AllResourceRequest
{
    std::uint64_t generation = 0;
    std::size_t record = 0;        ///< Unique pool request identity.
    std::size_t peak_frames = 0;   ///< Complete resident working set at peak.
    std::size_t occupied_frames = 0; ///< Live private frames + outstanding grants.
    std::size_t peak_banks = 0;    ///< Table banks needed at peak.
    std::size_t peak_leases = 0;   ///< Effective-use lease cells needed at peak.
    bool preserved = false;        ///< Lower-tier preservation available.
    bool operator==(const AllResourceRequest &) const = default;
};

/** @brief Composed admitted class maxima for one instance. */
struct AllResourceCapacity
{
    std::size_t frames = 0;             ///< Admitted pool frames.
    std::size_t banks = 0;              ///< Admitted table banks.
    std::size_t lease_cells = 0;        ///< Admitted effective-use lease cells.
    planning::WitnessBaseOccupancy base; ///< Physical occupancy outside requests.
    std::vector<AllResourceRequest> requests; ///< Live + proposed (bounded membership).
};

/** @brief Composition result: valid on success with a precise refusal otherwise. */
struct AllResourceWitnessResult
{
    bool valid = false;
    std::string refusal; ///< Nonempty exactly when invalid.
};

/**
 * @brief Prove a finite completion order across ALL classes.
 *
 * The frame proof is executed first (identical arithmetic to the pool's own
 * proof). The table-bank and lease-cell maxima are then verified together:
 * at every preserved/restored phase, the number of requests needing banks or
 * leases may never exceed the admitted class capacity (a class that is
 * required to evacuate an owner must not itself be obtainable only by
 * evacuating that owner).
 *
 * @return A valid result when every class has a bounded completion order;
 * otherwise a valid=false with the first deterministic refusal diagnostic.
 */
AllResourceWitnessResult proveAllResourceCompletion(
    const AllResourceCapacity &capacity);
} // namespace llaminar2
