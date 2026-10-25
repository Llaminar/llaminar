/**
 * @file ServingSnapshotArenaInventory.h
 * @brief Typed diagnostic arena populations for the canonical serving family.
 *
 * Snapshot sharing follows forward execution role, not native executable count:
 * prefill/verifier alternatives share a lane, MTP conditions share another,
 * while history-bearing and resident-input ordinary forwards are dedicated.
 * Sidecar arenas never own main-model state. This is an immutable BOM input;
 * PhysicalMemoryAuthority remains the only allocation and lifetime ledger.
 */
#pragma once
#include <cstddef>
#include "kernels/common/SparseQueryExecutionProgram.h"
#include <limits>
#include <stdexcept>

namespace llaminar2
{
/** @brief Overflow-checked setup geometry used by snapshot owner projections. */
inline std::size_t snapshotPopulationProduct(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::overflow_error("Serving snapshot population product overflows");
    return a * b;
}
/** @brief Overflow-checked sum of physically distinct retained snapshot owners. */
inline std::size_t snapshotPopulationSum(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        throw std::overflow_error("Serving snapshot population sum overflows");
    return a + b;
}
/** @brief Sealed serving arena roles, independent of observation file count. */
class ServingSnapshotArenaInventory final
{
public:
    /**
     * @brief Resolve the ordinary setup family and its proven sharing classes.
     * @param has_prefill Whether an admitted prefill ladder exists; its alternatives share one wide arena.
     * @param retains_mtp Whether condition/verifier forwards are retained, independent of active request depth.
     * @param sidecars Canonical MTPGraphOwnerPlan sidecar slots, separate from main state.
     * @param condition_requests Exact retained request capacity for MTP conditions; zero when absent.
     * @param query_programs Main-query program count; scalar snapshots remain dedicated per actual forward.
     * @throws std::invalid_argument for an inconsistent retained sidecar family.
     * @throws std::overflow_error for unrepresentable populations.
     */
    static ServingSnapshotArenaInventory ordinary(bool has_prefill, bool retains_mtp,
        std::size_t sidecars, std::size_t condition_requests, std::size_t query_programs = 1u)
    {
        if (query_programs == 0 || query_programs > kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Serving snapshot inventory has an unknown query-program family");
        if ((!retains_mtp && (sidecars || condition_requests)) || (retains_mtp && !condition_requests))
            throw std::invalid_argument("Serving snapshot inventory has inconsistent retained family geometry");
        return ServingSnapshotArenaInventory(
            has_prefill || retains_mtp ? 1u : 0u, snapshotPopulationProduct(2u, query_programs),
            retains_mtp ? 1u : 0u, sidecars, condition_requests);
    }
    /** @return Shared prefill/grouped-verifier lanes; bucket count is not an owner multiplier. */
    std::size_t wideMainOwners() const noexcept { return wide_; }
    /** @return Dedicated history-bearing and resident-input ordinary decode lanes. */
    std::size_t scalarMainOwners() const noexcept { return scalar_; }
    /** @return Shared MTP-condition lanes, including restored-prefix condition alternatives. */
    std::size_t conditionMainOwners() const noexcept { return conditions_; }
    /** @return Exact retained scalar condition rows, authored by MTPGraphOwnerPlan. */
    std::size_t conditionRows() const noexcept { return condition_rows_; }
    /** @return Predictor-only owners, excluded from all main state pricing. */
    std::size_t predictorOwners() const noexcept { return sidecars_; }
    /** @return All actual main-model arena owners. */
    std::size_t mainOwners() const { return snapshotPopulationSum(snapshotPopulationSum(wide_, scalar_), conditions_); }
    /** @return Main plus separate predictor arena owners. */
    std::size_t allOwners() const { return snapshotPopulationSum(mainOwners(), sidecars_); }
    friend bool operator==(const ServingSnapshotArenaInventory &, const ServingSnapshotArenaInventory &) = default;
private:
    /** @brief Only the canonical role resolver mints a population. */
    ServingSnapshotArenaInventory(std::size_t wide, std::size_t scalar, std::size_t conditions,
        std::size_t sidecars, std::size_t condition_rows)
        : wide_(wide), scalar_(scalar), conditions_(conditions), sidecars_(sidecars), condition_rows_(condition_rows) { (void)allOwners(); }
    std::size_t wide_, scalar_, conditions_, sidecars_, condition_rows_;
};
/**
 * @brief Authenticated ordinary reference phase bounds before serving-owner projection.
 *
 * Multiple scalar observations describe alternatives of the same retained
 * producer surface. Per-source maxima price each dedicated scalar owner; their
 * count cannot create or remove an arena. Wide phases are already widened by
 * the reference reader to the admitted physical row envelope. Unrelated
 * predictor evidence retains its separate pre-existing declaration.
 */
struct MainSnapshotPhasePayload
{
    std::size_t wide_main_bytes = 0u;
    std::size_t scalar_main_bytes = 0u;
    std::size_t separate_reference_bytes = 0u;
    /** @brief Project ordinary payloads over real roles without repricing native state banks. */
    std::size_t bytesFor(const ServingSnapshotArenaInventory &inventory) const
    {
        const auto wide = snapshotPopulationProduct(wide_main_bytes, inventory.wideMainOwners());
        const auto scalar_rows = snapshotPopulationSum(inventory.scalarMainOwners(),
            snapshotPopulationProduct(inventory.conditionMainOwners(), inventory.conditionRows()));
        const auto scalar = snapshotPopulationProduct(scalar_main_bytes, scalar_rows);
        return snapshotPopulationSum(snapshotPopulationSum(wide, scalar), separate_reference_bytes);
    }
    friend bool operator==(const MainSnapshotPhasePayload &, const MainSnapshotPhasePayload &) = default;
};
}
