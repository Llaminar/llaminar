/** @file MoEOverlayHostServiceContract.h
 * @brief One topology-only host-boundary decision shared by pre-quota planning and graph lowering.
 *
 * Pre-quota planning retains every declared endpoint, including initially empty
 * expert masks. Admitted execution may omit only tiers whose complete frozen
 * quota is zero and whose original owner map confirms no live contribution.
 * Mutable masks never remove an endpoint. Mapped node-local GPU transport needs
 * no CPU ticket worker.
 */
#pragma once
#include "MoEExpertOwnerMap.h"
#include <span>
#include <functional>
#include <cstdint>
namespace llaminar2 {
class IMPIContext;
enum class GraphNativeCaptureEnvelope : uint8_t;
struct ClusterInventory;
/** @brief Authored service demand and its immutable admitted execution projection. */
class MoEOverlayHostServiceContract final {
public:
    /**
     * @brief Retain all endpoints with any possible admitted expert contribution.
     * @param plan Original frozen plan; unresolved quotas retain their endpoints.
     * @param owners Original complete owner map, never a mutable routing mask.
     * @return Original participant descriptors without permanently empty external tiers.
     * @throws std::logic_error For inconsistent tier identities or zero-quota ownership.
     */
    static std::vector<MoEExpertOwnerParticipant> admittedExpertParticipants(
        const MoERoutedExpertPlacementPlan& plan, const MoEExpertOwnerMap& owners);
    /**
     * @brief Project immutable admitted membership without inventing expert ownership.
     * @param plan Original frozen logical placements and quotas.
     * @param participants Complete ordered descriptors from declaredParticipants.
     * @return Exact original descriptors excluding only proven permanently empty external tiers.
     * @throws std::logic_error For incomplete/foreign membership or zero-tier contradictions.
     * This metadata projection cannot authorize whole-expert or fragment dispatch.
     */
    static std::vector<MoEExpertOwnerParticipant> admittedExpertParticipants(
        const MoERoutedExpertPlacementPlan& plan,std::span<const MoEExpertOwnerParticipant> participants);
    /** @brief Classify existing physical rank transport from original tier descriptors. */
    static MoEOverlayHostServiceContract resolve(const MoERoutedExpertPlacementPlan&,std::span<const MoEExpertOwnerParticipant>,const IMPIContext&);
    /** @brief Resolve from original immutable physical topology without constructing an MPI context. */
    static MoEOverlayHostServiceContract resolve(const MoERoutedExpertPlacementPlan&,std::span<const MoEExpertOwnerParticipant>,const ClusterInventory&);
    /**
     * @brief Resolve execution after original admission has frozen every tier quota.
     * @param plan Original admitted topology; no placement or capacity is changed.
     * @param owners Complete immutable original ownership with stable participant IDs.
     * @param world Original rank topology used by the pre-quota classification.
     * @return Service endpoints excluding only proven permanently empty external tiers.
     * @throws std::logic_error If a frozen zero quota contradicts actual ownership.
     */
    static MoEOverlayHostServiceContract resolveAdmitted(const MoERoutedExpertPlacementPlan& plan,
        const MoEExpertOwnerMap& owners, const IMPIContext& world);
    /**
     * @brief Project the same admitted endpoints from the original physical inventory.
     * @param plan Original admitted topology and frozen tier quotas.
     * @param owners Complete original ownership, preserving participant identities.
     * @param inventory Original physical rank topology; no runtime probing occurs.
     * @return The identical admitted execution contract as the MPI overload.
     * @throws std::logic_error If a frozen zero quota contradicts actual ownership.
     */
    static MoEOverlayHostServiceContract resolveAdmitted(const MoERoutedExpertPlacementPlan& plan,
        const MoEExpertOwnerMap& owners, const ClusterInventory& inventory);
    /**
     * @brief Lower this immutable admitted service contract to its native parent envelope.
     * @param dispatchAuthority Whether this original continuation participant owns host dispatch.
     * @return One complete device timeline without host work; otherwise the original ticket role.
     *
     * Only the frozen admitted projection may remove an empty external tier. The
     * pre-quota resolve result remains conservative and selects a ticket whenever
     * a host-serviced endpoint is still declared.
     */
    [[nodiscard]] GraphNativeCaptureEnvelope captureEnvelope(bool dispatchAuthority) const noexcept;
    /** @return Same-tier-excluded local endpoints using actual host-serviced boundaries. */
    std::span<const int> localParticipants()const noexcept{return local_;}
    /** @return External retained endpoint graphs in original tier/participant order; real ranks are unchanged. */
    std::span<const int> externalParticipants()const noexcept{return external_;}
    /**
     * @return Whether an admitted same-rank cross-vendor expert endpoint requires
     * its original finite-transaction publisher, independently of CPU service work.
     */
    bool requiresRankLocalInferencePublisher() const noexcept { return rankLocalCrossVendor_; }
    /** @return Whether any topology-declared endpoint requires host ticket progress. */
    bool hasHostBoundary()const noexcept{return !local_.empty()||(!external_.empty()&&!allExternalNodeLocal_);}
    /** @return Whether every external endpoint is mapped node-local and at least one exists. */
    bool mappedActivationTopology()const noexcept{return !external_.empty()&&allExternalNodeLocal_;}
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
    std::vector<int> local_,external_;
    bool allExternalNodeLocal_=true,distributed_=false;
    bool rankLocalCrossVendor_=false; ///< Immutable explicit heterogeneous GPU boundary; no CPU service is invented.
};
}
