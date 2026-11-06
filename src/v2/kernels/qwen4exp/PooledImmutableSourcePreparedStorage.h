/**
 * @file PooledImmutableSourcePreparedStorage.h
 * @brief Existing-service-issued original PMA source/accepted-owner packet roots.
 *
 * Only ImmutableLookupService creates/reuses this registration. It owns no
 * provider or service handle (avoiding a source/storage lifetime cycle), no host
 * payload mirror and no live model counters. Source kernels and the existing
 * retained graph boundary keep the actual service binding alive independently.
 */
#pragma once
#include "PooledImmutableSourceGeometry.h"
#include "PooledImmutableNativeStorage.h"
#include "backends/NativeContextPreparation.h"
#include <optional>
namespace llaminar2 {class ModelContext;}
namespace llaminar2::test {struct PooledImmutableFactoryReplayAccess;}
namespace llaminar2 {class RetainedParticipantGraphExecutable;class RetainedParticipantGraphSourceSlots;class PooledRemoteGraphProgram;}
namespace llaminar2::qwen4exp {
class ImmutableLookupService;
/** @brief Immutable original-worker ownership consumed by actual source-root materialization. */
class PreparedImmutableSourceWorkers final {
public:
    /** @return Same original source preparation, never an independently rebound authority. */
    const std::shared_ptr<NativeContextPreparation>& source()const noexcept{return source_;}
    /** @return Actual original coordinator worker used to initialize source/policy roots. */
    const PreparedGPUContextUsePermit& coordinator()const noexcept{return coordinator_;}
    /** @return Actual original owner workers in physical owner-coordinate order. */
    std::span<const PreparedGPUContextUsePermit> owners()const noexcept{return owners_;}
    /** @return Actual immutable native source/owner root identity at successful materialization. */
    std::uint64_t rootIdentity()const noexcept{return root_identity_;}
private:
    friend class PooledImmutableSourcePreparedStorage;
    PreparedImmutableSourceWorkers(std::shared_ptr<NativeContextPreparation> source,PreparedGPUContextUsePermit coordinator,
        std::vector<PreparedGPUContextUsePermit> owners,std::uint64_t identity)
        :source_(std::move(source)),coordinator_(std::move(coordinator)),owners_(std::move(owners)),root_identity_(identity){}
    std::shared_ptr<NativeContextPreparation> source_;
    PreparedGPUContextUsePermit coordinator_;
    std::vector<PreparedGPUContextUsePermit> owners_;
    std::uint64_t root_identity_;
};
/** @brief One service channel's original native source storage and exact owner/coordinator associations. */
class PooledImmutableSourcePreparedStorage final {
public:
    /** @return Immutable exact original source packet geometry. */
    const PooledImmutableSourceGeometry& geometry()const noexcept{return geometry_;}
    /** @return The same original physical owner, retaining its actual slots and accepted grant banks. */
    const std::shared_ptr<PooledImmutableOwnerPreparedStorage>& owner()const{
        const auto local=geometry_.parameters().local_owner;
        if(local==adaptive_lookup::None)throw std::logic_error("Source has no participant-local cache reader");
        return owners_.at(local);
    }
    /** @return Original ordered owners, retained for cold identity/lifetime; GPU source work never dereferences remote roots. */
    std::span<const std::shared_ptr<PooledImmutableOwnerPreparedStorage>> originalOwners()const noexcept{return owners_;}
    /** @return The same independent original policy coordinator. */
    const std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>& coordinator()const noexcept{return coordinator_;}
    /** @return Consumed original worker ownership; absence means the prepared-source path did not materialize this root. */
    const std::optional<PreparedImmutableSourceWorkers>& retainedPreparedSourceWorkers()const noexcept{return prepared_workers_;}
    /** @return Original canonical source frame/native root aliases; optional cache is always absent. */
    PooledImmutableNativeRoots roots()const;
    /** @brief Authenticate exact source-owned native geometry and original PMA before reuse.
     * @param physical Original admitted physical layout. @param coordinator Matching policy geometry.
     * @param source Original source metadata. @param expected_compute_readers Expected declared compute-reader count.
     * @param memory Same original model-owned PMA.
     * @throws std::exception For foreign policy/source/authority or changed native roots. */
    void requireFor(const PooledImmutablePhysicalOwnerGeometry& physical,const PooledImmutableCoordinatorGeometry& coordinator,
        const ImmutableRowTableSource& source,std::uint32_t expected_compute_readers,
        const std::shared_ptr<const PhysicalMemoryAuthority>& memory)const;
    /** @return Actual original local owner, policy and source root claims, counted exactly once. */
    ImmutableLookupNativeOwnerObservation retainedOwnerObservation()const;
    /** @return Complete actual PMA/native-runtime/address/geometry/owner identity; reject any drift. */
    std::uint64_t captureIdentity()const;
    /** @brief Join only original native setup producers on the exact future consumer stream. */
    void acquireSetup(void* exact_stream)const;
private:
    friend class PooledImmutableRetainedParticipants;
    /** @brief Borrow the source's already-admitted serial remote definition, never another model's slot. */
    std::shared_ptr<RetainedParticipantGraphExecutable> acquireOriginalRemoteDefinition(const PooledRemoteGraphProgram&,void*);
    /** @brief Retire actual graph handles before the original runner arenas and native cohort disappear. */
    void releaseOriginalRemoteDefinition(const PooledRemoteGraphProgram&,
        const std::shared_ptr<RetainedParticipantGraphExecutable>&);
    /** @brief Withdraw failed native handles before local construction releases its original operands. */
    void abandonOriginalRemoteDefinition(const PooledRemoteGraphProgram&,
        const std::shared_ptr<RetainedParticipantGraphExecutable>&,std::exception_ptr);
    friend class ImmutableLookupService;
    friend class ::llaminar2::ModelContext;
    friend struct ::llaminar2::test::PooledImmutableFactoryReplayAccess;
    /** @brief Retain only workers actually consumed by the ModelContext materialization body, once. */
    void bindPreparedSourceWorkers(std::shared_ptr<NativeContextPreparation>,PreparedGPUContextUsePermit,
        std::vector<PreparedGPUContextUsePermit>);
    friend class IPooledImmutableSourceKernels;
    /**
     * @brief Materialize exactly admitted source staging through the existing canonical root materializer.
     * @param geometry Actual source-issued checked geometry.
     * @param owners Same original ordered owner roots retained by the source declaration.
     * @param coordinator Same original global directory/policy roots.
     * @param memory The service's original rank PMA; the caller already authenticated equality.
     * @param stream Exact ordinary setup stream; no metadata kernel or source request runs here.
     * @param group Closed original source claim, or empty for installed source selection.
     * @return Complete strong root owner or an exception with no published service registration.
     * @throws std::exception For foreign roots/claims or failed physical materialization.
     */
    static std::shared_ptr<PooledImmutableSourcePreparedStorage> materialize(PooledImmutableSourceGeometry geometry,
        std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* stream,std::unique_ptr<NativeOrderingEventGroup> group);

    /** @brief Only successful original-service materialization publishes these retained associations. */
    PooledImmutableSourcePreparedStorage(PooledImmutableSourceGeometry geometry,
        std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners,std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<detail::PooledNativeRootStorage> roots)
        :geometry_(std::move(geometry)),owners_(std::move(owners)),coordinator_(std::move(coordinator)),roots_(std::move(roots)){}
    std::optional<PreparedImmutableSourceWorkers> prepared_workers_;
    PooledImmutableSourceGeometry geometry_;
    std::vector<std::shared_ptr<PooledImmutableOwnerPreparedStorage>> owners_;
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::shared_ptr<detail::PooledNativeRootStorage> roots_;
    std::shared_ptr<RetainedParticipantGraphSourceSlots> remote_definition_slots_; ///< Original fixed owner roster; no source/cohort back-reference.
};
}
