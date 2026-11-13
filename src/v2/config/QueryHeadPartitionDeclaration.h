/**
 * @file QueryHeadPartitionDeclaration.h
 * @brief Immutable original-consumer geometry for canonical aligned query splitting.
 *
 * This declaration is cold source metadata, not a second assignment or a live
 * state ledger. The existing TensorParallelConfig distributor owns the split.
 * Every retained GDN consumer contributes an original-head divisibility demand;
 * no GDN consumer means the existing one-query-head granularity. Production
 * adoption still requires the same declaration at every construction owner.
 */
#pragma once
#include "GDNHeadAssignment.h"
#include <numeric>
#include <utility>

namespace llaminar2
{
/** @brief Original geometry of one retained GDN consumer sharing the Q partition space. */
struct OriginalGDNHeadGeometry
{
    int key_heads = 0;
    int value_heads = 0;
    bool operator==(const OriginalGDNHeadGeometry &) const = default;
};
/** @brief Copyable sealed source declaration; callers cannot author an arbitrary alignment scalar. */
class QueryHeadPartitionDeclaration
{
public:
    /**
     * @brief Resolve the smallest query grain satisfying every actual original consumer.
     * @param original_query_heads Original common query partition space.
     * @param consumers Complete original retained GDN geometries; empty means non-GDN alignment1.
     * @return Immutable original geometry plus its minimal common divisor grain.
     * @throws std::invalid_argument For missing source geometry or invalid GDN modular linkage.
     */
    [[nodiscard]] static QueryHeadPartitionDeclaration fromOriginalConsumers(
        int original_query_heads, std::vector<OriginalGDNHeadGeometry> consumers)
    {
        if (original_query_heads <= 0)
            throw std::invalid_argument("Query partition requires positive original head geometry");
        int grain = 1;
        for (const auto &consumer : consumers)
        {
            const int required = GDNHeadAssignment::partitionGranularity(
                consumer.key_heads, consumer.value_heads, original_query_heads);
            // Every required grain divides original_query_heads, so their LCM
            // also divides it and cannot overflow the original positive int.
            grain = std::lcm(grain, required);
        }
        return QueryHeadPartitionDeclaration(original_query_heads, grain, std::move(consumers));
    }
    /** @return Authenticated original partition width; never a participant-derived sum. */
    [[nodiscard]] int originalQueryHeads() const noexcept { return original_query_heads_; }
    /** @return Minimal positive alignment consumed by the existing canonical distributor. */
    [[nodiscard]] int granularity() const noexcept { return granularity_; }
    /** @return Complete retained original GDN demands; no model/device names are stored. */
    [[nodiscard]] const std::vector<OriginalGDNHeadGeometry> &consumers() const noexcept { return consumers_; }
private:
    /** @brief Only the checked source factory can mint a concrete granularity. */
    QueryHeadPartitionDeclaration(int query_heads, int grain, std::vector<OriginalGDNHeadGeometry> consumers)
        : original_query_heads_(query_heads), granularity_(grain), consumers_(std::move(consumers)) {}
    int original_query_heads_;
    int granularity_;
    std::vector<OriginalGDNHeadGeometry> consumers_;
};
} // namespace llaminar2
