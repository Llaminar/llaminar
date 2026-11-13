/**
 * @file PooledImmutableSourceKernels.h
 * @brief Real retained RAM/SSD provider bridge for accepted native pooled grants.
 *
 * This code bundle retains the existing service-issued source channel. Source
 * storage and owner/coordinator roots have their original PMA owners; canonical
 * request control and output remain graph-owned. Every source gate is a fixed
 * captured operation, with no host cache decisions or alternate delivery path.
 */
#pragma once
#include "PooledImmutableSourceGeometry.h"
#include "PooledImmutableReadOps.h"
#include <memory>
namespace llaminar2 {class DeviceGenerationControlBinding;}
namespace llaminar2::qwen4exp {
class PooledImmutableSourcePreparedStorage;
/** @brief Distinct source producers/readers with explicit owner/coordinator graph edges between them. */
enum class PooledSourceAction:std::uint32_t {Plan,AcquireSeal,Publish,AcquireResponse,PackOwner,RetireResponse,Read,ReadersRetired,ObserveRetired,BeforeReady,
    ImportPublication=10,ValidatePublication=11,ImportPublicationAcquired=12,ValidatePublicationAcquired=13,ImportReady=14,ValidateReady=15,PlanReads=16,ImportReadRows=17,SealReads=18,PlanReleases=24,ImportReleases=25,ReconcileReleases=26,ImportRetired=27,FinishRetired=28,PackReaderRetirement=29,SnapshotReleased=30};
/** @return Nominal fixed owner import; full-frame consensus operations have no addressed owner. */
constexpr bool isPooledPublicationImport(PooledSourceAction action){return action==PooledSourceAction::ImportPublication || action==PooledSourceAction::ImportPublicationAcquired || action==PooledSourceAction::ImportReady;}
/** @return Original source publication/acquisition/Ready consensus; these operations retire no private-row reader. */
constexpr bool isPooledPublicationConsensus(PooledSourceAction action){return action==PooledSourceAction::ValidatePublication || action==PooledSourceAction::ValidatePublicationAcquired || action==PooledSourceAction::ValidateReady;}
/** @return Fixed original terminal protocol operations; no full-factory capability is advertised. */
constexpr bool isPooledRetirementOperation(PooledSourceAction action){return action==PooledSourceAction::PlanReleases || action==PooledSourceAction::ImportReleases || action==PooledSourceAction::ReconcileReleases || action==PooledSourceAction::ImportRetired || action==PooledSourceAction::FinishRetired || action==PooledSourceAction::PackReaderRetirement || action==PooledSourceAction::SnapshotReleased;}
/** @return Exactly the original addressed terminal metadata/receipt imports. */
constexpr bool isPooledRetirementAddressed(PooledSourceAction action){return action==PooledSourceAction::ImportReleases || action==PooledSourceAction::ImportRetired || action==PooledSourceAction::PackReaderRetirement;}
/** @return Addressing belongs only to actual fixed captured owner operations. */
constexpr bool isPooledSourceAddressed(PooledSourceAction action){return isPooledRetirementAddressed(action) || action==PooledSourceAction::AcquireSeal || action==PooledSourceAction::PackOwner || isPooledPublicationImport(action) || action==PooledSourceAction::ImportReadRows;}
/** @brief Native output already retained/sealed by its actual reusable graph stage. */
struct PooledSourceQuery {
    float* output=nullptr; ///< Original canonical embedding output, without aliasing native authority.
    std::size_t elements=0; ///< This graph's exact physical FP32 envelope, no larger than the shared source declaration.
    std::uint32_t owner=adaptive_lookup::None; ///< Immutable coordinate for the exact owner-addressed source operation.
};
/** @brief Checked participant-local source code; arbitrary directed owner edges remain explicit transport contracts. */
class PooledImmutableRetirementAuthority;
class PooledImmutableLocalHotNativePack;
class IPooledImmutableSourceKernels {
public:
    /** @brief Release only after graph terminal and final source readers; retained source bindings own mapped backing. */
    virtual ~IPooledImmutableSourceKernels()=default;
    /** @return Complete original source/frame/owner/coordinator declarations. */
    virtual const PooledImmutableSourceGeometry& geometry()const noexcept=0;
    /** @return Existing service-issued provider/channel binding, retained unchanged. */
    virtual const ImmutableSubordinateLookupBinding& source()const noexcept=0;
    /** @brief Initialize source protocol metadata once, retaining its exact setup event even on partial failure. */
    void initialize(PooledImmutableSourcePreparedStorage& storage,void* exact_stream)const;
    /**
     * @brief Publish only immutable I/O admission and its actual retained-worker ticket outside capture.
     * @param storage Same initialized original source roots.
     * @param generation Exact admitted service generation.
     * @param invocation Exact retained-worker dispatch ticket for that generation.
     * @param exact_stream Producer stream ordered before the complete captured parent.
     * @note Successful request reset preserves sequence history, warmed slots and graph topology.
     */
    void admit(PooledImmutableSourcePreparedStorage& storage,std::uint64_t generation,std::uint64_t invocation,void* exact_stream)const;
    /**
     * @brief Enqueue one fixed source operation over actual accepted grants and canonical controller.
     * @param action Explicit captured source producer/reader operation.
     * @param storage Original source-issued root owner.
     * @param control Existing canonical request authority, never a host-owned health mirror.
     * @param query Exact original FP32 output span for Read, absent for other operations.
     * @param exact_stream Actual non-default participant stream with graph producer edges already acquired.
     */
    void enqueue(PooledSourceAction action,PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,PooledSourceQuery query,void* exact_stream)const;
    /** @brief Copy exact compact native rows through original readable-row/pin authority; no FP32 output is created. */
    void enqueueCompactNativePack(PooledImmutableSourcePreparedStorage&,const DeviceGenerationControlBinding&,
        const PooledImmutableLocalHotNativePack&,void* exact_stream)const;
    /** @brief Close only at the complete parent's canonical terminal, after source reader retirement. */
    void closeWhenTerminal(PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,void* exact_stream)const;
    /** @brief Explicit event-ordered request reset closes after actual source readers, independent of the generation terminal predicate. */
    void closeForReset(PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,void* exact_stream)const;
    /**
     * @brief Close explicit reset/cancel only under the actual complete retained cohort.
     * @param storage Original source root owner authenticated by the complete descriptor.
     * @param control Original canonical controller. @param exact_stream Exact source worker stream.
     * @param cohort Retained actual Read/output/Retirement source and remote programs.
     * @note The same native close still validates reader completion, original
     * coordinator Idle and exact reported ordinal; this does not fabricate drain.
     */
    void closeForReset(PooledImmutableSourcePreparedStorage& storage,
        const DeviceGenerationControlBinding& control,void* exact_stream,const PooledImmutableRetirementAuthority& cohort)const;
    /** @return Complete source/code/root/runtime identity; stages additionally bind exact controller/output roles. */
    std::uint64_t captureIdentity(const PooledImmutableSourcePreparedStorage& storage)const;
protected:
    /** @return All immutable codebook addresses, wall-clock deadline, format and launch policies. */
    virtual std::uint64_t codeIdentity()const noexcept=0;
    /** @brief Receive only checked original addresses; perform no allocation or host observation. */
    virtual void initializeNative(const pooled_source::DeviceArgs&,void*)const=0;
    /** @brief Preserve device I/O sequence history while publishing the original admission/ticket. */
    virtual void admitNative(const pooled_source::DeviceArgs&,std::uint64_t,std::uint64_t,void*)const=0;
    /** @brief Submit the fixed exact-live-byte source operation on the supplied exact stream. */
    virtual void launch(PooledSourceAction,const pooled_source::DeviceArgs&,PooledSourceQuery,void*)const=0;
    /** @brief Complete-parent terminal close uses only original canonical device predicates. */
    virtual void closeNative(const pooled_source::DeviceArgs&,const int*,void*)const=0;
private:
    /** @brief Reject foreign service/root associations before any native operation. */
    void requireStorage(const PooledImmutableSourcePreparedStorage&)const;
    /** @return Original sealed native coordinates, never live host-materialized counts. */
    pooled_source::DeviceArgs args(const PooledImmutableSourcePreparedStorage&,const DeviceGenerationControlBinding*)const;
};
/** @return CUDA source bridge with actual original codebooks and closed service binding. */
std::unique_ptr<IPooledImmutableSourceKernels> createCUDAPooledImmutableSourceKernels(PooledImmutableSourceGeometry,
    ImmutableSubordinateLookupBinding);
/** @return HIP source bridge under the same original service/protocol/byte/lifetime contract. */
std::unique_ptr<IPooledImmutableSourceKernels> createROCmPooledImmutableSourceKernels(PooledImmutableSourceGeometry,
    ImmutableSubordinateLookupBinding);
}
