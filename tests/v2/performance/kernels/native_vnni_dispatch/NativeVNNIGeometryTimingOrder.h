/**
 * @file NativeVNNIGeometryTimingOrder.h
 * @brief Typed balanced sampling orders for dense and routed MoE diagnostics.
 *
 * The immutable shared .def inventories own actual measurement permutations.
 * Retained graphs are submitted serially in this order; no host timing order
 * is substituted for a different native candidate or a repeated graph body.
 */
#pragma once
#include <array>
#include <cstddef>
namespace llaminar2::test::native_vnni_dispatch
{
/** @brief Exact number of physically distinct diagnostic contenders. */
inline constexpr std::size_t kGeometryTimingCandidates = 5;
/** @brief Complete ten-round rotation/reversal period from the common inventory. */
inline constexpr std::array<std::array<std::size_t, kGeometryTimingCandidates>, 10>
    kGeometryTimingOrders{{
#define LLAMINAR_GEOMETRY_TIMING_ORDER(a,b,c,d,e) {{a,b,c,d,e}},
#include "NativeVNNIGeometryTimingOrder.def"
#undef LLAMINAR_GEOMETRY_TIMING_ORDER
}};
/** @return Whether all permutations, positions and pair directions are balanced. */
constexpr bool geometryTimingOrdersValid() noexcept
{
    std::array<std::array<std::size_t, 5>, 5> exposures{}, before{};
    for (const auto &order : kGeometryTimingOrders)
    {
        std::array<bool, 5> seen{};
        for (std::size_t position = 0; position < 5; ++position)
        {
            const auto candidate = order[position];
            if (candidate >= 5 || seen[candidate]) return false;
            seen[candidate] = true;
            ++exposures[candidate][position];
            for (std::size_t later = position + 1; later < 5; ++later)
            {
                if (order[later] >= 5) return false;
                ++before[candidate][order[later]];
            }
        }
    }
    for (std::size_t a = 0; a < 5; ++a)
        for (std::size_t b = 0; b < 5; ++b)
            if (exposures[a][b] != 2 || (a != b && before[a][b] != 5)) return false;
    return true;
}
static_assert(geometryTimingOrdersValid(), "diagnostic order inventory must be balanced");
/**
 * @brief Read the actual immutable order for one warmup or measured round.
 * @param round Zero-based completed round, independent of latency sorting.
 * @return All five candidate indices exactly once, repeating after ten rounds.
 */
[[nodiscard]] constexpr const auto &geometryTimingOrder(std::size_t round) noexcept
{
    return kGeometryTimingOrders[round % kGeometryTimingOrders.size()];
}
/** @brief Three actual M1 gate/up widths; the five dense modes remain unchanged. */
inline constexpr std::array<std::array<std::size_t, 3>, 6> kMoERouteGeometryTimingOrders{{
#define LLAMINAR_MOE_ROUTE_TIMING_ORDER(a,b,c) {{a,b,c}},
#include "NativeVNNIMoERouteTimingOrder.def"
#undef LLAMINAR_MOE_ROUTE_TIMING_ORDER
}};
/**
 * @brief Read one complete three-way rotation/reversal round.
 * @param round Zero-based chronological round; sorting latency never changes it.
 * @return Each actual gate/up contender once, balanced over six rounds.
 */
[[nodiscard]] constexpr const auto &moeRouteGeometryTimingOrder(std::size_t round) noexcept
{
    return kMoERouteGeometryTimingOrders[round % kMoERouteGeometryTimingOrders.size()];
}

}
