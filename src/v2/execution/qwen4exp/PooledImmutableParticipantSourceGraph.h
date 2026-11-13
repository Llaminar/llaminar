/**
 * @file PooledImmutableParticipantSourceGraph.h
 * @brief Original service-issued source packets and explicit participant-local fill endpoints.
 *
 * This component retains the original r5 victim/coordinator/owner authorities,
 * the sole real source service and original PMA. It ends at actual fill outcomes;
 * publication, remote readers and complete native factories remain guarded.
 */
#pragma once
#include "PooledImmutableDemandFrontier.h"
#include "PooledImmutableParticipantVictimGraph.h"
#include "PooledImmutableParticipantLocalGraph.h"
#include "kernels/qwen4exp/PooledImmutableSourcePreparedStorage.h"
namespace llaminar2::qwen4exp {
/** @brief Distinct protocol producers; allocation padding is never a wire extent. */
enum class PooledSourceWireRole {InitialFrontier,SealReceipt,Fills,Payload,TerminalFrontier};
/** @brief Actual original source/owner binding, with no new physical root, service or ledger. */
class PooledImmutableCapturedSourceWave final {
public:
    /**
     * @brief Bind exact original source packet/count and remote owner ingress subranges.
     * @param previous Actual original completed victim-wave holder.
     * @param boundary Sole actual service-issued source/storage/code owner.
     * @param memory Same original PMA that admitted every native root and channel.
     * @return Complete fixed/counted endpoints; native/rank and mixed-local gaps reject.
     */
    static std::shared_ptr<const PooledImmutableCapturedSourceWave> bind(
        std::shared_ptr<const PooledImmutableCapturedVictimWave> previous,
        std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Authenticate original source, storage, cohort, allocation and runtime generations. */
    void requireCurrent()const;
    /** @return Original victim/free/pin chain, retaining its exact two channels. */
    const std::shared_ptr<const PooledImmutableCapturedVictimWave>& previous()const noexcept{return previous_;}
    /** @return Same real source boundary, never a substituted provider. */
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary()const noexcept{return boundary_;}
    /** @return Original addressed owner coordinate. */
    uint32_t coordinate()const noexcept{return previous_->declaration().coordinate();}
    /** @return Whether this owner has admitted Fill/payload storage; live zero remains counted. */
    bool hasPayload()const noexcept{return previous_->declaration().hasPayload();}
    /** @return Exact original producer/receiver root subrange, not its dynamic wire count. */
    PooledPinWireSpan source(PooledSourceWireRole role)const;
    PooledPinWireSpan destination(PooledSourceWireRole role)const;
    /** @return Original fixed/counted binding; C0 data roles reject. */
    const PooledImmutableTransferBinding& binding(PooledSourceWireRole role,CapturedTransferEndpoint endpoint)const;
    /** @return Exact original inputs for Seal/Apply/Validate; terminal outcome is distinct from reusable header. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Complete source/edge/binding identity excluding mutable source protocol values. */
    uint64_t captureIdentity()const;
private:
    using Pair=std::pair<PooledImmutableTransferBinding,PooledImmutableTransferBinding>;
    using Bindings=std::array<std::optional<Pair>,5>;
    /** @brief Only complete original-authority validation publishes a source endpoint holder. */
    PooledImmutableCapturedSourceWave(std::shared_ptr<const PooledImmutableCapturedVictimWave> previous,
        std::shared_ptr<ImmutableLookupExecutionBoundary> boundary,std::shared_ptr<PhysicalMemoryAuthority> memory,Bindings bindings)
        :previous_(std::move(previous)),boundary_(std::move(boundary)),memory_(std::move(memory)),bindings_(std::move(bindings)){}
    std::shared_ptr<const PooledImmutableCapturedVictimWave> previous_;
    std::shared_ptr<ImmutableLookupExecutionBoundary> boundary_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    Bindings bindings_;
};
/** @brief Original direct-local relation or real captured source endpoint; no nullable channel denotes local work. */
using PooledImmutableSourceParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,std::shared_ptr<const PooledImmutableCapturedSourceWave>>;
/** @return Exact original physical participant; source membership/PMA is reauthenticated before use. */
std::shared_ptr<PooledImmutableOwnerPreparedStorage> pooledSourceOwner(const PooledImmutableSourceParticipant& member);
/** @brief Reauthenticate the actual original source/coordinator/ordered owner association and local-versus-remote role. */
void requirePooledSourceParticipant(const PooledImmutableSourceParticipant& member,const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary);
/** @return Exact original source/fill operands, borrowing source packet only for the one actual local owner. */
PooledNativeOperandSpans pooledSourceOwnerInputs(const PooledImmutableSourceParticipant& member,
    const std::shared_ptr<ImmutableLookupExecutionBoundary>& boundary,pooled_native::OwnerAction action);
/** @return Immutable action/endpoint identity; final multiply preserves Producer0 versus Consumer1. */
inline constexpr uint64_t pooledSourceEndpointCaptureIdentity(uint64_t binding,PooledSourceWireRole role,CapturedTransferEndpoint endpoint)noexcept
{return ((((binding^(48+static_cast<unsigned>(role)))*1099511628211ULL)^static_cast<unsigned>(endpoint))*1099511628211ULL)|1;}
}
namespace llaminar2 {
/** @brief Concrete original source producer boundaries; this descriptor grants no storage/lifetime authority. */
struct PooledImmutableSourceFillCompletion {
    std::vector<qwen4exp::PooledImmutableSourceParticipant> owners;
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary;
    std::optional<PooledImmutableMixedVictimCompletion> previous;
    std::optional<DeviceGenerationControlBinding> control;
    PooledPinParticipantNodes coordinator;
    std::string predecessor,planned,sealed,published,acquired,retired;
    std::string localSeal,localDrain,localApply,localValidate,localOutcome;
    std::vector<std::string> initial,sealReceived,sealAcquired,packed,fillSent,payloadSent,terminalSent;
};
/** @brief Append original mixed source phases, retaining one local-last packet until its actual Apply/copy completion. */
PooledImmutableSourceFillCompletion appendPooledSourceCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const qwen4exp::PooledImmutableSourceParticipant> owners,
    std::shared_ptr<qwen4exp::ImmutableLookupExecutionBoundary> boundary,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,DeviceGenerationControlBinding control,
    const std::string& completeVictims,const std::string& prefix,const PooledImmutableMixedVictimCompletion* mixed=nullptr);
/** @brief Authenticate every actual original source producer, packet reader and prior victim chain before a successor mutates graph. */
void requirePooledSourceCoordinatorCompletion(const ComputeGraph& graph,const PooledImmutableSourceFillCompletion& original);
/** @brief Authenticate actual original owner source receive/seal/fill/copy/validation and prior victim chain before publication. */
void requirePooledSourceOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedSourceWave>& original,const std::string& fillOutcome);
/** @brief Append actual source-seal/receipt, exact Fill/payload receive and owner-local Apply/copy/Validate.
 * @param sentVictims Exact original owner VictimReceipt send, authenticated before graph mutation.
 * @return Participant-local span ending with guaranteed typed success/failure PublishHeader56. */
PooledPinParticipantNodes appendPooledSourceOwnerFill(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedSourceWave> edge,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& sentVictims,const std::string& prefix, PooledImmutableDemandFrontier* demand=nullptr);
/** @brief Append the sole original request, all-owner seal acquisition, compact packets and truthful source retirement.
 * @param completeVictims Exact original ALL-owner CompleteVictims; no local completion may be fabricated.
 * @param control Original coordinator generation controller, retained by the source boundary.
 * @return Participant-local span ending after exact terminal frontier fanout; no publication/read/Ready support. */
PooledPinParticipantNodes appendPooledSourceCoordinatorFill(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedSourceWave>> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,DeviceGenerationControlBinding control,
    const std::string& completeVictims,const std::string& prefix);
}

namespace llaminar2 {
/**
 * @brief Seal the actual inline owner's first response wait after ordinary trunk readiness.
 * @param graph Original graph with the complete addressed owner protocol.
 * @param frontier Original seal-send and first response-receive descriptor.
 * @param trunk Actual same-participant trunk producers, never the demand prefix.
 * Only this consumed join may supplement the otherwise sole protocol predecessor.
 */
void bindPooledInlineDemandConsumption(ComputeGraph& graph,const PooledImmutableDemandFrontier& frontier,
    const std::vector<std::string>& trunk);
}
