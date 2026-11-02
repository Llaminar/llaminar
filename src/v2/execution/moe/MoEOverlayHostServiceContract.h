/** @file MoEOverlayHostServiceContract.h
 * @brief One topology-only host-boundary decision shared by pre-quota planning and graph lowering.
 *
 * Every declared endpoint participates even with an initially empty expert mask.
 * Mutable routed ownership/load therefore changes neither thread demand nor
 * captured endpoint topology. Mapped node-local GPU transport needs no CPU ticket worker.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include <span>
#include <functional>
namespace llaminar2 {
class IMPIContext;
struct ClusterInventory;
/** @brief Exact authored endpoint classification; no ownership rows or quota arithmetic. */
class MoEOverlayHostServiceContract final {
public:
    /** @brief Classify existing physical rank transport from original tier descriptors. */
    static MoEOverlayHostServiceContract resolve(const MoERoutedExpertPlacementPlan&,std::span<const MoEExpertOwnerParticipant>,const IMPIContext&);
    /** @brief Resolve from original immutable physical topology without constructing an MPI context. */
    static MoEOverlayHostServiceContract resolve(const MoERoutedExpertPlacementPlan&,std::span<const MoEExpertOwnerParticipant>,const ClusterInventory&);
    /** @return Same-tier-excluded local endpoints using actual host-serviced boundaries. */
    std::span<const int> localParticipants()const noexcept{return local_;}
    /** @return Remote endpoints in canonical tier/participant order. */
    std::span<const int> remoteParticipants()const noexcept{return remote_;}
    /** @return Whether any topology-declared endpoint requires host ticket progress. */
    bool hasHostBoundary()const noexcept{return !local_.empty()||(!remote_.empty()&&!allRemoteNodeLocal_);}
    /** @return Whether every remote endpoint is mapped node-local and at least one exists. */
    bool mappedActivationTopology()const noexcept{return !remote_.empty()&&allRemoteNodeLocal_;}
    /** @return Whether this exact continuation graph owns the one required executor service. */
    bool requiresExecutorService(int rank,DeviceId consumer,int localTPDegree,int localTPIndex)const;
    /** @return Same continuation tier excluded by actual graph lowering. */
    int continuationTier()const noexcept{return continuationTier_;}
private:
    static MoEOverlayHostServiceContract resolveImpl(const MoERoutedExpertPlacementPlan&,std::span<const MoEExpertOwnerParticipant>,int,const std::function<bool(int,int)>&);
    struct ContinuationGeometry {ExecutionDomainScope scope;std::size_t participants;RoutedExpertComputePolicy compute;};
    std::optional<ContinuationGeometry> continuation_; ///< Immutable copied declaration scalars; no borrowed candidate pointer escapes.
    std::vector<MoEExpertOwnerParticipant> continuationParticipants_;
    int continuationTier_=-1,rootParticipant_=-1;
    std::vector<int> local_,remote_;
    bool allRemoteNodeLocal_=true,distributed_=false;
};
}
