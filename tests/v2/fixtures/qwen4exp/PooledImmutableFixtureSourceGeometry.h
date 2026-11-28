/**
 * @file PooledImmutableFixtureSourceGeometry.h
 * @brief Single authority for the pooled read-graph fixture's native source geometry.
 *
 * The pooled fixture (PooledImmutableReadGraphFixture.h) and the direct
 * eviction-counter regression that mirrors it
 * (ImmutableRowTable.PooledFixtureGeometryForcesWindowEviction) must describe
 * the same original source, or the counter proof would silently stop
 * describing the fixture. Both include these constants; neither may restate
 * the values. The MemoryMapped slot count is the physical window-capacity
 * knob: it deliberately differs from historical two-slot values so window
 * eviction pressure is exercised on every host page size, while the logical
 * row count and codec row bytes stay fixed.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include "loaders/ImmutableRowTableExtent.h"

namespace llaminar2::test
{
    struct PooledImmutableFixtureSourceGeometry
    {
        static constexpr std::size_t kPrefixBytes = 37; ///< Unaligned original-file prefix.
        static constexpr std::size_t kRows = 1000; ///< Logical row count.
        static constexpr std::size_t kRowBytes = 90; ///< Codec bytes per row (IQ4_NL x 160 elements).
        static constexpr std::uint32_t kRowElements = 160; ///< Elements per row.
        static constexpr auto kFormat = GGUFTensorType::IQ4_NL; ///< Original codec.
        static constexpr std::size_t kWindowSlots = 1; ///< MemoryMapped LRU slot count.
    };
} // namespace llaminar2::test
