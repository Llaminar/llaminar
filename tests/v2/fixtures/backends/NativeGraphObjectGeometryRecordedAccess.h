/**
 * @file NativeGraphObjectGeometryRecordedAccess.h
 * @brief Recorded owner evidence for synthetic planning fixtures.
 *
 * Planning prices each GPU's graph adapter with its owner's published size.
 * Synthetic fixtures plan hypothetical ranks that publish nothing; this named
 * access records one owner value for every GPU, independent of the backends
 * compiled into the test binary. Production code cannot construct it.
 */
#pragma once
#include "backends/NativeGraphObjectGeometry.h"

namespace llaminar2::test {

struct NativeGraphObjectGeometryRecordedAccess
{
    /** Recorded owner adapter size used by planning fixtures. */
    static constexpr std::size_t kRecordedBytes = 1024;

    /** @return Geometry answering kRecordedBytes for every GPU owner. */
    static NativeGraphObjectGeometry uniform(std::size_t bytes = kRecordedBytes)
    {
        NativeGraphObjectGeometry geometry;
        geometry.recorded_ = bytes;
        return geometry;
    }
};

} // namespace llaminar2::test
