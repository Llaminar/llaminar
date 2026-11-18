/**
 * @file MoEIntermediateShardRuntimeBinding.h
 * @brief Cold descriptor publication from the original prepared fragment family.
 *
 * This boundary borrows the original construction bank only while exporting
 * immutable physical descriptors. Runtime banks and their existing initial
 * publication lifecycle remain the sole placement/reset authority; no engine,
 * initial epoch reader or host execution mirror is retained by the graph.
 */
#pragma once
namespace llaminar2
{
class DeviceMoERuntimeTable;
class MoEExpertOwnerMap;
struct MoEOverlayParticipantLayerBank;
/**
 * @brief Publish the original local-F baseline into the existing runtime/parent banks.
 * @param runtime Participant runtime retaining the exact original family for this global layer.
 * @param original Same original owner map retained by source preparation and residency.
 * @param source Actual installed original prepared fragment bank, borrowed for this call only.
 * @param layer Original global main or predictor source layer.
 * @param stream Exact non-null cold setup publication stream.
 * @throws std::invalid_argument For a foreign family/source/member or a call inside capture.
 * @throws std::runtime_error For failed native descriptor export or bank publication.
 */
void publishInitialMoEIntermediateShardRuntimeLayer(DeviceMoERuntimeTable &runtime,
    const MoEExpertOwnerMap &original, const MoEOverlayParticipantLayerBank &source, int layer, void *stream);
}
