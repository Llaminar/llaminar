/**
 * @file MoEIntermediateShardArenaGeometry.h
 * @brief Original-source activation geometry for participant-local F-sharded FFNs.
 *
 * This metadata contribution is shared by graph schema, resolver and PMA BOM.
 * It owns no allocation or live state. Routed F follows the canonical native
 * source partition, while shared F follows the original dense/shared TP
 * assignment. Local router slots never acquire transport or rank-bank padding.
 */
#pragma once
#include "execution/moe/GPUExpertIntermediateTiling.h"
#include "config/TensorParallelConfig.h"
#include <cstddef>
#include <optional>
namespace llaminar2
{
struct ModelMemoryProfile;
struct GraphSchema;
struct GraphResolverConfig;
/** @brief Complete immutable local activation envelope across main and predictor sources. */
class MoEIntermediateShardArenaGeometry final
{
public:
    /**
     * @brief Resolve actual native routed cuts and independent original shared assignment.
     * @param source Original metadata inventory including every NextN source layer.
     * @param participants Declared original homogeneous GPU cohort size, including one.
     * @param participant Original cohort coordinate, never a physical GPU ordinal.
     * @param shared_assignment Original dense/shared TP interval, absent only for a singleton.
     * @param first_layer First original source layer in this retained stage manifest.
     * @param last_layer Last retained source layer, inclusive; -1 selects the original full inventory.
     * @param tiling Immutable partition preference consumed by the same original weight owner.
     * @return Exact maxima reused by serial main/predictor graph families.
     * @throws std::invalid_argument For incomplete source, incompatible/native-split geometry or foreign ownership.
     */
    static MoEIntermediateShardArenaGeometry resolve(const ModelMemoryProfile &source, int participants,
        int participant, const std::optional<DeviceShardingAssignment> &shared_assignment = {},
        int first_layer = 0, int last_layer = -1,
        GPUExpertIntermediateTiling tiling = GPUExpertIntermediateTiling::Coarse128);
    /** @brief Bind the same exact physical row widths to the graph resolver. */
    void bindResolver(GraphResolverConfig &resolver) const;
    /** @brief Give simultaneous local partials distinct existing arena lifetime groups. */
    void bindSchema(GraphSchema &schema) const;
    /** @return Maximum actual local routed F, not the source's complete F. */
    int routedIntermediate() const noexcept { return routed_; }
    /** @return Original local shared F, independent of routed ownership. */
    int sharedIntermediate() const noexcept { return shared_; }
    /** @return Existing gate/up scratch capacity shared by nonconcurrent routed/shared arithmetic. */
    int scratchIntermediate() const noexcept;
    /** @return Router-slot count only; there is no pre-fold rank-bank plane. */
    int routeSlots() const noexcept { return slots_; }
private:
    /** @brief Seal complete source-derived geometry; partial declarations cannot escape. */
    MoEIntermediateShardArenaGeometry(int routed, int shared, int slots):routed_(routed),shared_(shared),slots_(slots){}
    int routed_,shared_,slots_;
};
}
