/**
 * @file PooledImmutableDirectedOwnerProgram.h
 * @brief Shared exact owner semantics with separate mapped-flat and native-General contracts.
 *
 * The mapped descriptor retains its original portable kernel seal. Native
 * graphs authenticate the same actual semantic recipe without minting a flat
 * seal or resource allowance. Neither arm certifies a complete source/model/MTP
 * graph or grants native transport ownership.
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
    /** @brief Declare the exact mapped-flat or native-General storage contract before admission.
     * @param physical Original physical roster. @param coordinator Same source policy. @param coordinate Actual owner coordinate.
     * @return Native General for homogeneous original endpoints; mapped certificate for cross-vendor endpoints.
     */
    static PooledStorageOwnerProgram declareStorage(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate);
    /** @brief Reconstruct the same typed contract from actual service-issued original plans.
     * @param owner Original physical owner plan. @param coordinator Same source/policy plan.
     * @return Exact mapped-flat or native-General declaration; no resource authority is created.
     * @throws std::exception For foreign source/generation, geometry or transport ownership.
     */
    static PooledStorageOwnerProgram fromOriginalStoragePlans(const PooledImmutableOwnerNativePlan& owner,
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
    /**
     * @brief Authenticate the actual native-library owner recipe without a flat kernel certificate.
     * @param graph Actual participant-local owner graph before native recording.
     * @param owner Original prepared physical owner; its retained endpoints own native authority.
     * @param coordinator Same original source/policy plan, including table and generation.
     * @throws std::exception For cross-vendor plans or changed transport, phases, extent, slice or DAG.
     * @note This proves semantic graph composition only. The original General
     * executable/source admission separately owns native-library resource bounds.
     */
    static void requireNativeActual(const ComputeGraph& graph,const PooledImmutableOwnerNativePlan& owner,
        const PooledImmutableCoordinatorNativePlan& coordinator);
};
}
