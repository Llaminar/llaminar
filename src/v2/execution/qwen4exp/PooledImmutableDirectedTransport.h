/**
 * @file PooledImmutableDirectedTransport.h
 * @brief Original same-PMA directed channel ownership for declared remote immutable cache participants.
 *
 * One remote owner reuses exactly two original channels across Pin, Free,
 * Victim, Source, Publication, Read and retirement. Local owners create none.
 * Participant graphs and recurring hosted transaction submission remain explicit
 * consumers; this owner launches no graph and holds no model or residency state.
 */
#pragma once
#include "PooledImmutableDirectedChannelGeometry.h"
#include "PooledImmutableParticipantRetirementGraph.h"
#include "ImmutableLookupExecutionBoundary.h"
namespace llaminar2::qwen4exp {
/** @brief Complete immutable original chain for one addressed remote coordinate. */
struct PooledImmutableRemoteTransport final {
    std::uint32_t coordinate;
    std::shared_ptr<const PooledImmutableCapturedPinEdge> pin;
    std::shared_ptr<const PooledImmutableCapturedPinReturn> returned;
    std::shared_ptr<const PooledImmutableCapturedFreeWave> free;
    std::shared_ptr<const PooledImmutableCapturedVictimWave> victim;
    std::shared_ptr<const PooledImmutableCapturedSourceWave> source;
    std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication;
    std::shared_ptr<const PooledImmutableCapturedReadWave> read;
    std::shared_ptr<const PooledImmutableCapturedRetirementWave> retirement;
};
/** @brief Pure BOM projection and actual original TransferEngine materialization, never another admission ledger. */
class PooledImmutableDirectedTransport final {
public:
    /** @brief Exact two-channel protocol floor at zero optional owner payload capacity. */
    static PhysicalMemoryPlan mandatoryBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources);
    /** @brief Exact full two-channel contribution for this candidate's actual owner capacities. */
    static PhysicalMemoryPlan completeBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources);
    /** @brief Source-owned exact growth above the same ordered protocol floor; add before candidate fits. */
    static PhysicalMemoryPlan slotDependentBOM(const PooledImmutablePhysicalOwnerGeometry&,const PooledImmutableCoordinatorGeometry&,
        PhysicalMemoryResource host,std::span<const PhysicalMemoryResource> originalGPUResources);
    /** @brief Bind original source permits/streams and existing directed waves once outside capture.
     * @param boundary Actual service/source owner retaining the original prepared participant cohort.
     * @return Strong two-channel chain owner; no peer graph/worker/stream is created here.
     */
    static std::shared_ptr<const PooledImmutableDirectedTransport> create(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary);
    /** @return Actual remotes in original coordinate order; co-located owners are absent. */
    std::span<const PooledImmutableRemoteTransport> remotes()const noexcept{return remotes_;}
    /** @return Actual original addressed remote chain; a local/foreign coordinate rejects. */
    const PooledImmutableRemoteTransport& remote(std::uint32_t coordinate)const;
    /** @brief Reject source/worker/generation/channel/root drift before graph composition. */
    void requireCurrent()const;
private:
    /** @brief Construct only after all actual source permits have authenticated. */
    explicit PooledImmutableDirectedTransport(std::shared_ptr<ImmutableLookupExecutionBoundary> boundary):boundary_(std::move(boundary)){}
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary_;
    std::vector<PooledImmutableRemoteTransport> remotes_;
};
}
