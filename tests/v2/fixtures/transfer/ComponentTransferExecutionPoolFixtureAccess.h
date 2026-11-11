/**
 * @file ComponentTransferExecutionPoolFixtureAccess.h
 * @brief Explicit native stream infrastructure for low-level component tests.
 *
 * These pools reuse the production cold kernel preparation and exact background
 * scheduling body. They do not certify original rank, NUMA, source, or PMA
 * admission. Original model/native fixtures must use the public permit API.
 * Component lanes cannot supply originalContext() or enter native peer events.
 */
#pragma once
#include "transfer/TransferEngine.h"

namespace llaminar2::test {
/** @brief The sole test-side friend allowed to create component-only pools. */
struct ComponentTransferExecutionPoolFixtureAccess final {
    /**
     * @brief Materialize real context-owned component streams outside capture.
     * @param engine Existing public movement authority used by this component.
     * @param count Actual concurrent stream count.
     * @param device Actual physical endpoint, without fabricated rank provenance.
     * @param name Stable infrastructure identity isolated from admitted pools.
     * @return Strong worker-retaining lanes, each rejecting originalContext().
     */
    [[nodiscard]] static std::vector<PersistentTransferExecutionLane> allocate(
        const TransferEngine &engine,size_t count,DeviceId device,const std::string &name)
    {return engine.allocateComponentTransferExecutionLanes(count,device,name);}

    /**
     * @brief Record only a host-modeled component handle for authority negatives.
     * @param lifetime Device-free owner retained through all copies of the handle.
     * @param stream Host sentinel representing an exact non-default stream.
     * @param device Symbolic GPU; no backend is loaded or queried.
     * @return Component-only scope, with no way to obtain an original permit.
     */
    [[nodiscard]] static PersistentTransferExecutionLane recordedComponentForReplay(
        std::shared_ptr<void> lifetime,void *stream,DeviceId device)
    {return PersistentTransferExecutionLane(PersistentTransferExecutionLane::ComponentFixtureTag{},
        std::move(lifetime),stream,device,0);}
};
}
