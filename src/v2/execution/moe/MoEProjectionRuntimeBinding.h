/**
 * @file MoEProjectionRuntimeBinding.h
 * @brief Setup-only bridge from frozen prepared ownership to the existing runtime epoch.
 *
 * The projection mode uses exactly the same runtime placement table and device
 * controller as whole experts. This bridge publishes its movable gate/up
 * descriptor family; immutable down columns are retained separately by that
 * table. It never chooses or mirrors a request-time owner.
 */
#pragma once
#include "execution/moe/MoEOverlayPreparedExpertPayload.h"
#include <vector>

namespace llaminar2
{
    class DeviceMoERuntimeTable;
    class MoEExpertOwnerMap;

    /**
     * @brief Idempotently publish one frozen, prepared projection layer at setup.
     * @param runtime Existing parent or child runtime with its fixed down bank bound.
     * @param owners Frozen initial global gate/up owner map.
     * @param layer Authenticated model layer, including a retained MTP layer.
     * @param payloads Local gate/up owners in global expert order; all others empty.
     * @param stream Exact setup publication stream, never null/default.
     * @throws std::invalid_argument For missing/mismatched ownership or payload family.
     * @throws std::runtime_error If the existing runtime cannot publish the baseline.
     *
     * A child publishes its canonical parent first. Once the runtime declares
     * its setup baseline installed, graph construction never overwrites the
     * live device placement, including after Dynamic movement or prefix restore.
     */
    void publishInitialMoEProjectionRuntimeLayer(DeviceMoERuntimeTable &runtime,
        const MoEExpertOwnerMap &owners, int layer,
        const std::vector<MoEOverlayPreparedExpertPayload> &payloads, void *stream);
}
