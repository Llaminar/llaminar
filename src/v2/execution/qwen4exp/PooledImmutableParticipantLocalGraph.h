/**
 * @file PooledImmutableParticipantLocalGraph.h
 * @brief Original same-allocator owner bindings for mixed pooled pre-source waves.
 *
 * A local owner consumes original producers directly and publishes fixed typed
 * metadata into the existing coordinator frontier. It never creates a transfer
 * endpoint, temporary packet, count mirror or allocation authority.
 */
#pragma once
#include "PooledImmutableParticipantVictimGraph.h"
#include <variant>

namespace llaminar2::qwen4exp {
/** @brief Checked original local relation, retaining both prepared owners and their sole PMA. */
class PooledImmutableLocalOwnerBinding final {
public:
    /** @brief Authenticate original same-allocator membership and PMA before constructing any local graph node. */
    static std::shared_ptr<const PooledImmutableLocalOwnerBinding> bind(
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,
        std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Reject changed original physical roots, runtime generations, membership or authority. */
    void requireCurrent()const;
    /** @return Original addressed coordinate, including its C0 identity. */
    std::uint32_t coordinate()const noexcept{return owner_->plan().parameters().owner.coordinate();}
    /** @return Exact original positive reply capacity; zero omits only payload work. */
    bool hasPayload()const noexcept{return owner_->plan().parameters().native.reply_capacity!=0;}
    /** @return Original local coordinator owner. */
    const std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>& coordinatorStorage()const noexcept{return coordinator_;}
    /** @return Original physical owner, never an alias or transported surrogate. */
    const std::shared_ptr<PooledImmutableOwnerPreparedStorage>& ownerStorage()const noexcept{return owner_;}
    /** @return Complete original input spans for one actual owner phase. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Checked original direct/directed owner operation, retaining actual source roots. */
    PooledImmutableOwnerOperands ownerOperands(pooled_native::OwnerAction action)const;
    /** @return Actual local metadata producer or original Grant prefix, with exact addressed bounds. */
    PooledImmutableCoordinatorOperands coordinatorOperands(pooled_native::CoordinatorAction action)const;
    /** @return Complete immutable original relation, independent of live counts/offsets. */
    std::uint64_t captureIdentity()const;
private:
    /** @brief Construct only from fully authenticated original owners. */
    PooledImmutableLocalOwnerBinding(std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PhysicalMemoryAuthority> memory);
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::uint64_t coordinator_identity_,owner_identity_;
};

/** @brief Nominal local or actual remote owner; local work never has a nullable transport endpoint. */
using PooledImmutablePinParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,
    std::shared_ptr<const PooledImmutableCapturedPinReturn>>;
/** @brief Original local relation or actual remote Free bindings over the original Pin channels. */
using PooledImmutableFreeParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,
    std::shared_ptr<const PooledImmutableCapturedFreeWave>>;
/** @brief Original local relation or actual remote Victim bindings; no owner frontier is invented. */
using PooledImmutableVictimParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,
    std::shared_ptr<const PooledImmutableCapturedVictimWave>>;

/** @brief Immutable original full-cohort order shared by mixed Pin/Free/Victim assembly. */
class PooledImmutableMixedCohort final {
public:
    /** @brief Require exactly the declared P original members and exactly one same-allocator local owner. */
    static PooledImmutableMixedCohort fromOriginal(std::vector<PooledImmutablePinParticipant> owners);
    /** @brief Reauthenticate every retained original relation before graph mutation or capture. */
    void requireCurrent()const;
    /** @return Original ordered members, without coordinate remapping. */
    const std::vector<PooledImmutablePinParticipant>& owners()const noexcept{return owners_;}
    /** @return Actual original local owner coordinate. */
    std::uint32_t localCoordinate()const noexcept{return local_;}
    /** @return Original coordinator shared by all original members. */
    const std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>& coordinatorStorage()const noexcept{return coordinator_;}
private:
    /** @brief Keep only validated complete membership. */
    PooledImmutableMixedCohort(std::vector<PooledImmutablePinParticipant> owners,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,std::uint32_t local)
        :owners_(std::move(owners)),coordinator_(std::move(coordinator)),local_(local){}
    std::vector<PooledImmutablePinParticipant> owners_;
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::uint32_t local_;
};
}

namespace llaminar2 {
/** @brief Require exactly the original cold phase predecessors; extra/backward edges cannot masquerade as a valid completion. */
void requirePooledOriginalDependencies(const ComputeGraph& graph,const std::string& consumer,
    std::vector<std::string> expected);
/** @brief Actual coordinator/local-owner Pin boundaries retained for subsequent mixed free insertion. */
struct PooledImmutableMixedPinCompletion {
    std::shared_ptr<const qwen4exp::PooledImmutableMixedCohort> cohort;
    PooledPinParticipantNodes coordinator;
    std::string localBegin,localPin,localSeal,localAcquire,commandsProduced;
    std::string beginProducer,seal;
    std::vector<PooledPinReplyOwnerNodes> replies;
    std::vector<std::string> frontiers;
};
/** @brief Authenticate actual original local/remote Pin producers and every overwrite/read frontier before Free mutation. */
void requirePooledMixedPinCompletion(const ComputeGraph& graph,const PooledImmutableMixedPinCompletion& original);
/** @brief Reusable original coordinator Pin assembly over actual remote or direct-local member operands.
 * @param beginProducer Original coordinator Begin, required exactly when a local member is present.
 * @param commandSent Exact remote command/Invocation send nodes; the local entry is empty and becomes its real Pin reader.
 * @note Caller binds both original local owner/coordinator roots to the same participant arena before construction. */
PooledImmutableMixedPinCompletion appendPooledPinCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const qwen4exp::PooledImmutablePinParticipant> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,
    std::span<const std::string> commandSent,const std::string& beginProducer,
    const std::string& commandsProduced,const std::string& prefix);
/** @brief Actual complete mixed Free boundary, with original local producer names for later insertion authentication. */
struct PooledImmutableMixedFreeCompletion {
    std::optional<PooledImmutableMixedPinCompletion> previous;
    PooledPinParticipantNodes coordinator;
    std::vector<qwen4exp::PooledImmutableFreeParticipant> owners;
    std::vector<std::string> availability,frontiers;
    std::vector<PooledPinReplyOwnerNodes> replies;
    std::string planned,completed,localAvailability,localAccept,localExecute,localReceipt,localAcquire;
};
/** @brief Shared original coordinator Free assembly for actual remote or direct-local participants. */
PooledImmutableMixedFreeCompletion appendPooledFreeCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const qwen4exp::PooledImmutableFreeParticipant> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,
    const std::string& pinFrontiersComplete,const std::string& prefix,
    const PooledImmutableMixedPinCompletion* localPin=nullptr);
/** @brief Authenticate actual mixed Free producer/read/send edges before any Victim insertion. */
void requirePooledMixedFreeCompletion(const ComputeGraph& graph,const PooledImmutableMixedFreeCompletion& original);
/** @brief Actual complete mixed Victim frontier; source successors must authenticate this original receipt chain. */
struct PooledImmutableMixedVictimCompletion {
    std::optional<PooledImmutableMixedFreeCompletion> previous;
    PooledPinParticipantNodes coordinator;
    std::vector<qwen4exp::PooledImmutableVictimParticipant> owners;
    std::vector<PooledPinReplyOwnerNodes> replies;
    std::string planned,completed,localAccept,localExecute,localReceipt;
};
/** @brief Shared original Victim assembly, with no unused owner frontier traffic. */
PooledImmutableMixedVictimCompletion appendPooledVictimCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const qwen4exp::PooledImmutableVictimParticipant> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,
    const std::string& sentFree,const std::string& prefix,
    const PooledImmutableMixedFreeCompletion* localFree=nullptr);
/** @brief Authenticate actual original mixed Victim receipt frontier before a source successor mutates the graph. */
void requirePooledMixedVictimCompletion(const ComputeGraph& graph,const PooledImmutableMixedVictimCompletion& original);
}
