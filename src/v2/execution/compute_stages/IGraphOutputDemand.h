/**
 * @file IGraphOutputDemand.h
 * @brief Optional cold publication-demand boundary for closed fused graph regions.
 *
 * Only demand-aware fused stages implement this interface. Existing stage and
 * snapshot ownership remain unchanged: the executor asks its original semantic
 * filter about cold descriptors, and the stage unions selected observations
 * with canonical outside reads in one immutable native publication contract.
 * No mutable graph pointer, payload mirror, allocator or binding cache is owned
 * here. A separate opt-in interface preserves unrelated compute-stage ABIs.
 */
#pragma once
#include "IComputeStage.h"

namespace llaminar2 {
class ComputeGraph;
class BufferArena;
class DeviceMoERuntimeTable;

/** @brief Exact ordered read of the producer's selected gate/up/down descriptor rows. */
struct MoESelectedDescriptorRead
{
    std::string_view producer; ///< Exact participant-local producer node.
    const DeviceMoERuntimeTable* runtime; ///< Original descriptor-table binding owner.
    int model_layer; ///< Layer within that original owner.
};

/** @brief Declare device reads of selected descriptor rows before graph preparation. */
class IMoESelectedDescriptorConsumer {
public:
    /** @brief Retire the declaration with the stage that consumes the rows. */
    virtual ~IMoESelectedDescriptorConsumer()=default;
    /** @return Immutable publication reads used by this stage for its lifetime. */
    virtual std::span<const MoESelectedDescriptorRead> selectedDescriptorReads() const noexcept=0;
};
/** @brief Stage-owned immutable output demand, sealed only during cold preparation. */
class IGraphOutputDemand {
public:
    /** @brief Retire this optional view with its owning compute stage. */
    virtual ~IGraphOutputDemand()=default;
    /**
     * @brief Declare possible semantic snapshot outputs before native preparation.
     * @return Cold immutable descriptors, or no declaration for this stage variant.
     * @details Must not inspect payloads, materialize storage, execute a stage or
     * depend on prepared device views. Names, types, geometry and row metadata
     * describe the same possible outputs as the ordinary snapshot interface;
     * data/tensor pointers may be absent until preparation. The executor's
     * existing descriptor-aware filter remains the sole selection authority.
     */
    virtual std::optional<StageDumpInfo> coldSnapshotPublicationDeclaration() const=0;
    /**
     * @brief Seal actual outside reads and selected observations as one demand.
     * @param graph Complete contracted participant-local graph, not retained.
     * @param arena Canonical tensor/alias authority, never a copied binding cache.
     * @param selected_observations Names selected by the existing snapshot filter.
     * @param snapshot_epoch Existing complete callback/filter configuration identity.
     * @details The owning stage must reject changed prepared bindings or policy
     * rather than update a live captured program. No graph pointer or mutable
     * execution observation is retained. Ordinary stages do not implement this
     * optional interface and keep their original preparation-dependent dumps.
     */
    virtual void sealGraphOutputDemand(const ComputeGraph& graph,const BufferArena& arena,
        std::span<const std::string> selected_observations,std::uint64_t snapshot_epoch)=0;
};
} // namespace llaminar2
