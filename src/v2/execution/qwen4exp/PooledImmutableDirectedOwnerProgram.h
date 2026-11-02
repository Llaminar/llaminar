/**
 * @file PooledImmutableDirectedOwnerProgram.h
 * @brief Source-authored flat remote owner program declared before admission and checked against actual stages.
 *
 * This descriptor describes only the original directed owner graph. It never
 * certifies a complete source/model/MTP graph or grants native ownership.
 */
#pragma once
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/PooledRemoteGraphProgram.h"
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
namespace llaminar2 {class ComputeGraph;}
namespace llaminar2::qwen4exp {
/** @brief Sole author of the portable launch recipe and exact original owner geometry. */
class PooledImmutableDirectedOwnerProgram final {
public:
    /**
     * @brief Declare the actual owner recipe using original identity-free native geometry.
     * @param physical Checked ordered storage owners and independent coordinator.
     * @param coordinator Matching original coalescing/request geometry.
     * @param coordinate Addressed opposite-vendor owner coordinate.
     * @return Closed source descriptor; absent installed host certification still rejects materialization.
     */
    static PooledRemoteGraphProgram declare(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate);
    /** @brief Reconstruct exactly the same descriptor from actual source-issued native plans after admission.
     * @param owner Actual source-issued addressed native plan, not a fabricated geometry identity.
     * @param coordinator Actual matching policy/coalescing plan.
     * @return The same source descriptor that planning declared before materialization. */
    static PooledRemoteGraphProgram fromOriginalPlans(const PooledImmutableOwnerNativePlan& owner,
        const PooledImmutableCoordinatorNativePlan& coordinator);
    /**
     * @brief Compare actual typed local stages/edges/dispatch inputs before native recording.
     * @param graph Exclusive actual directed owner graph, without unrelated model stages.
     * @param owner Actual original prepared owner plan, retaining its original source generation.
     * @param coordinator Actual original policy plan.
     * @param program Cold declaration from the same pre-admission geometry.
     * @throws std::exception Before capture on changed role, ABI, geometry, ordering or unsupported stage.
     */
    static void requireActual(const ComputeGraph& graph,const PooledImmutableOwnerNativePlan& owner,
        const PooledImmutableCoordinatorNativePlan& coordinator,const PooledRemoteGraphProgram& program);
};
}
