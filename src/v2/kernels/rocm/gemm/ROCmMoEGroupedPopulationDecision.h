/**
 * @file ROCmMoEGroupedPopulationDecision.h
 * @brief Pure population classification and exclusive fused row ownership.
 *
 * Only the population compilation shard and device-free policy tests include
 * these decisions. Retained bindings stay stable while measured thresholds
 * change, avoiding recompilation of unrelated ordered and native kernels.
 */
#pragma once
#include "ROCmMoEGroupedPopulationProjection.h"
#if defined(__HIPCC__)
#define LLAMINAR_POPULATION_HD __host__ __device__
#else
#define LLAMINAR_POPULATION_HD
#endif
namespace llaminar2::rocm
{
/** @brief Integer reduction identity; empty experts never lower the positive minimum. */
struct MoEGroupedPopulation
{
    int active = 0;
    int minimum = INT_MAX;
    int maximum = 0;
};

/** @param domain Sealed operation family. @param population Reduced positive expert counts.
 * @return Exclusive projection writer, or Invalid for malformed/unsupported state.
 * @note RouteOwned is resolved first by the existing route authority. The narrow
 * cold count12 condition means one complete TM12 tile; adjacent partial tiles
 * deliberately do not inherit that measured choice. */
[[nodiscard]] LLAMINAR_POPULATION_HD constexpr MoEGroupedPopulationFamily groupedPopulationFamily(
    MoEGroupedPopulationDomain domain, MoEGroupedPopulation population) noexcept
{
    using enum MoEGroupedPopulationFamily;
    if (domain != MoEGroupedPopulationDomain::Qwen && domain != MoEGroupedPopulationDomain::Compact)
        return Invalid;
    const int experts = domain == MoEGroupedPopulationDomain::Qwen ? 256 : 8;
    if (population.active == 0)
        return population.minimum == INT_MAX && population.maximum == 0 ? Dot4 : Invalid;
    if (population.active < 0 || population.active > experts || population.minimum <= 0 ||
        population.maximum < population.minimum || population.maximum > 4096)
        return Invalid;
    if (domain == MoEGroupedPopulationDomain::Compact)
        return population.active == 8 && population.minimum == population.maximum && population.minimum >= 12
            ? N32 : Dot4;
    if (population.minimum >= 16 || (population.active <= 32 && population.minimum >= 7)) return N32;
    if (population.maximum >= 16 || population.maximum >= 5 * population.minimum ||
        (population.minimum == 12 && population.maximum == 12)) return WideFloor16;
    return Dot4;
}

/** @param family Published global policy. @param count Whole-expert count.
 * @return Exclusive arithmetic producer, or Invalid for malformed state. */
[[nodiscard]] LLAMINAR_POPULATION_HD constexpr MoEGroupedDotProducer populationProjectionProducer(
    MoEGroupedPopulationFamily family,int count) noexcept
{
    if (count <= 0) return MoEGroupedDotProducer::Invalid;
    switch (family)
    {
    case MoEGroupedPopulationFamily::Dot4: return MoEGroupedDotProducer::OrderedDot4;
    case MoEGroupedPopulationFamily::N32: return MoEGroupedDotProducer::Wmma;
    case MoEGroupedPopulationFamily::WideFloor16:
        return count >= 16 ? MoEGroupedDotProducer::Wmma : MoEGroupedDotProducer::OrderedDot4;
    default: return MoEGroupedDotProducer::Invalid;
    }
}
/** @param role Projection role. @param family The same publication read by its consumer.
 * @param count Whole-expert count. @return Exact directory row step, or zero for invalid/empty state. */
[[nodiscard]] LLAMINAR_POPULATION_HD constexpr int populationProjectionRows(
    MoEPrefillProjectionRole role,MoEGroupedPopulationFamily family,int count) noexcept
{
    if (role != MoEPrefillProjectionRole::GateUp && role != MoEPrefillProjectionRole::Down) return 0;
    const auto producer = populationProjectionProducer(family,count);
    if (producer == MoEGroupedDotProducer::Invalid) return 0;
    return role == MoEPrefillProjectionRole::Down || producer == MoEGroupedDotProducer::Wmma ? 16 : 12;
}
}
#undef LLAMINAR_POPULATION_HD
