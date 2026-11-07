/**
 * @file CPUCacheObservation.h
 * @brief Genuine deterministic-cache observations without convenience capacities.
 *
 * This cold, allocation-free decoder accepts CPUID records, not guessed cache
 * sizes. A cache is private only when its reported logical sharers fit a proven
 * SMT-core topology. Missing topology, unsupported leaves and malformed/wrapped
 * sizes remain unknown zeros. Tests supply records to this decoder; they never
 * replace the process's hardware capabilities or immutable native observation.
 */
#pragma once
#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace llaminar2::cpu
{
/** @brief Cache fields proven for the observing logical CPU, not a package sum. */
struct CPUCacheObservation
{
    std::uint64_t private_l1d_bytes = 0;
    std::uint64_t private_l2_bytes = 0;
    std::uint64_t shared_l3_bytes = 0;
    std::uint32_t cache_line_bytes = 0;
    std::uint32_t private_l2_ways = 0;
    std::uint32_t shared_l3_ways = 0;
};

/** @brief One decoded deterministic cache record; all-zero denotes invalid/absent. */
struct DeterministicCacheObservation
{
    std::uint64_t bytes = 0;
    std::uint32_t type = 0, level = 0, logical_sharers = 0, line_bytes = 0, ways = 0;
};

/** @brief Decode one architectural leaf4/8000001D record with checked size arithmetic.
 * @param registers EAX/EBX/ECX/EDX from the selected cache subleaf.
 * @return Exact data/unified-cache metadata, or unknown for an absent/invalid record.
 */
[[nodiscard]] constexpr DeterministicCacheObservation decodeDeterministicCache(
    const std::array<std::uint32_t, 4> &registers) noexcept
{
    const auto type = registers[0] & 31u;
    const auto level = (registers[0] >> 5) & 7u;
    if ((type != 1 && type != 3) || level < 1 || level > 3) return {};
    const auto ways = ((registers[1] >> 22) & 1023u) + 1;
    const auto line = (registers[1] & 4095u) + 1;
    const std::array<std::uint64_t, 4> factors{
        ways, ((registers[1] >> 12) & 1023u) + 1, line, std::uint64_t(registers[2]) + 1};
    std::uint64_t bytes = 1;
    for (auto factor : factors)
    {
        if (bytes > std::numeric_limits<std::uint64_t>::max() / factor) return {};
        bytes *= factor;
    }
    return {.bytes = bytes, .type = type, .level = level,
        .logical_sharers = ((registers[0] >> 14) & 4095u) + 1,
        .line_bytes = line, .ways = ways};
}

/** @brief Observe cache records using only supported leaves and proven SMT geometry.
 * @tparam Reader Callable returning four uint32 registers for (leaf, subleaf).
 * @param read Hardware reader in production; explicit record fixture in decoder tests.
 * @return Genuine private/shared fields; no unknown value becomes a default capacity.
 */
template<class Reader>
[[nodiscard]] CPUCacheObservation observeX86CacheGeometry(Reader &&read)
{
    const auto identification = read(0u, 0u);
    const auto basic = identification[0];
    const auto extended = read(0x80000000u, 0u)[0];
    std::uint32_t smt_threads = 0;
    for (auto leaf : {0x1fu, 0xbu})
    {
        if (basic < leaf) continue;
        for (std::uint32_t sub = 0; sub < 32; ++sub)
        {
            const auto record = read(leaf, sub);
            const auto kind = (record[2] >> 8) & 255u;
            if (kind == 0 || (record[1] & 65535u) == 0) break;
            if (kind == 1)
            {
                // Cache sharing names addressable APIC IDs, not only enabled
                // siblings. Preserve the SMT ID width when SMT is disabled.
                const auto shift = record[0] & 31u;
                if (shift < 16 && (record[1] & 65535u) <= (1u << shift))
                    smt_threads = 1u << shift;
                break;
            }
        }
        if (smt_threads) break;
    }
    // AMD Family15h reports cores per compute unit here, not SMT threads.
    // Only documented Zen-family semantics can establish a private-core domain
    // (also reflected by Linux topology_amd.c::parse_8000_001e).
    const bool amd = identification[1] == 0x68747541u &&
        identification[3] == 0x69746e65u && identification[2] == 0x444d4163u;
    if (!smt_threads && amd && basic >= 1 && extended >= 0x8000001eu &&
        (read(0x80000001u, 0u)[2] & (1u << 22)))
    {
        const auto signature = read(1u, 0u)[0];
        const auto base_family = (signature >> 8) & 15u;
        const auto family = base_family == 15 ? base_family + ((signature >> 20) & 255u) : base_family;
        if (family >= 0x17)
            smt_threads = ((read(0x8000001eu, 0u)[1] >> 8) & 255u) + 1;
    }

    CPUCacheObservation result;
    for (auto leaf : {4u, 0x8000001du})
    {
        if ((leaf == 4 && basic < leaf) || (leaf != 4 && extended < leaf)) continue;
        for (std::uint32_t sub = 0; sub < 16; ++sub)
        {
            const auto registers = read(leaf, sub);
            if ((registers[0] & 31u) == 0) break;
            const auto cache = decodeDeterministicCache(registers);
            if (!cache.bytes) continue;
            const bool private_core = smt_threads && cache.logical_sharers <= smt_threads;
            if (cache.level == 1 && cache.type == 1 && private_core && !result.private_l1d_bytes)
            {
                result.private_l1d_bytes = cache.bytes;
                result.cache_line_bytes = cache.line_bytes;
            }
            if (cache.level == 2 && private_core && !result.private_l2_bytes)
            {
                result.private_l2_bytes = cache.bytes;
                result.private_l2_ways = cache.ways;
                if (!result.cache_line_bytes) result.cache_line_bytes = cache.line_bytes;
            }
            if (cache.level == 3 && !result.shared_l3_bytes)
            {
                result.shared_l3_bytes = cache.bytes;
                result.shared_l3_ways = cache.ways;
            }
        }
    }
    return result;
}
} // namespace llaminar2::cpu
