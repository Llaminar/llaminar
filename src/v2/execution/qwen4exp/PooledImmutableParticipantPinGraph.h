/**
 * @file PooledImmutableParticipantPinGraph.h
 * @brief Original-cohort directed pin operands and participant-local graph endpoints.
 *
 * This is a bounded initial-wave component, not a complete lookup materializer.
 * Declarations derive from the existing source-owned plans. Actual binding
 * retains their original prepared roots/PMA and uses the explicitly declared
 * mapped or producer-owned peer channel. Original inventory validates same-vendor
 * edges before materialization; no failed-peer fallback or host count readback exists.
 */
#pragma once
#include "PooledImmutableLocalLookupGraph.h"
#include "transfer/CapturedTransferChannel.h"
#include "PooledImmutableTransferBinding.h"
namespace llaminar2::qwen4exp
{
/** @brief Nominal wire roles; storage alignment can never be selected as their live extent. */
enum class PooledPinWireRole {Invocation,Commands,Frontier,Replies,Header};
/**
 * @brief Combine one immutable retained binding identity, nominal wire action and exact endpoint.
 * @param bindingIdentity Full original capture identity (roots, generations, operands and channel).
 * @param role Actual invocation/command/frontier/reply/header action, never mutable protocol state.
 * @param endpoint Exact producer or consumer represented by this participant-local stage.
 * @return Stable nonzero variant; final mixing preserves the endpoint before nonzero normalization.
 *
 * Both directed stage classes use this single policy. OR-ing the raw endpoint
 * into the final low bit erases Producer=0 versus Consumer=1. Multiplication
 * after endpoint mixing avoids that collision; no live offset/count is hashed.
 */
inline constexpr uint64_t pooledPinEndpointCaptureIdentity(uint64_t bindingIdentity,
    PooledPinWireRole role,CapturedTransferEndpoint endpoint) noexcept
{
    constexpr uint64_t prime=1099511628211ULL;
    const auto action=(bindingIdentity^static_cast<unsigned>(role))*prime;
    return ((action^static_cast<unsigned>(endpoint))*prime)|1;
}

/** @brief Exact original root subrange, without an allocation or copied live value. */
struct PooledPinWireSpan
{
    BufferId buffer;
    size_t offset,bytes;
    bool operator==(const PooledPinWireSpan&)const=default;
};
/** @brief Immutable declaration for one original coordinator-to-owner initial wave. */
class PooledImmutablePinEdge final
{
public:
    /** @brief Authenticate the complete original owner/source/coordinator relation before declaring an edge. */
    static PooledImmutablePinEdge fromPlans(PooledImmutableCoordinatorNativePlan coordinator,PooledImmutableOwnerNativePlan owner);
    /** @return Exact original independent source and receiver allocators. */
    PhysicalMemoryAllocatorIdentity sender()const noexcept{return coordinator_.allocator();}
    /** @return Exact original addressed physical owner, never a TP compute coordinate. */
    PhysicalMemoryAllocatorIdentity receiver()const noexcept{return owner_.allocator();}
    /** @return Whether original allocators differ; this never claims heterogeneity or transport permission. */
    bool remote()const noexcept{return sender()!=receiver();}
    /** @return True only for a rank-local GPU edge with its explicit mapped transport policy. */
    bool capturedChannelTransport()const noexcept;
    /** @return Original addressed owner coordinate, including C0. */
    uint32_t coordinate()const noexcept{return owner_.parameters().owner.coordinate();}
    /** @return Whether this owner has a positive admitted command bank; runtime zero remains a counted message. */
    bool hasCommands()const noexcept{return maximumCommandBytes()!=0;}
    /** @return Exact admitted command maximum, never allocation capacity on the wire. */
    size_t maximumCommandBytes()const noexcept;
    /** @return The original source's whole compact command region; it can exceed this edge's wire maximum. */
    PooledPinWireSpan commandSource()const noexcept;
    /** @return Original16-byte descriptor produced by CoordinatorAction::PinCommands. */
    PooledPinWireSpan commandSlice()const noexcept;
    /** @return Original owner frame command input; no extra packet bank is allocated. */
    PooledPinWireSpan commandDestination()const noexcept;
    /** @return Immutable48-byte invocation control, never padded to its64-byte storage alignment. */
    PooledPinWireSpan invocationSource()const noexcept;
    /** @return Original remote-owner ingress invocation span; local edges keep the direct source role. */
    PooledPinWireSpan invocationDestination()const noexcept;
    /** @return Distinct incoming uint64 byte publication, absent exactly for C0. */
    std::optional<PooledPinWireSpan> incomingCount()const noexcept;
    /** @return Exact existing channel geometry to contribute before optional placement; no admission is performed. */
    CapturedTransferChannelMemory channelMemory()const;
    /** @return Exact original current-wave Grant prefix and original uint64 byte producer. */
    PooledPinWireSpan replySource()const noexcept;
    /** @return Original owner progress byte scalar, never a repacked row count. */
    PooledPinWireSpan replyCountSource()const noexcept;
    /** @return Reusable original pin_commands prefix; every command send must finish before receiving here. */
    PooledPinWireSpan replyDestination()const noexcept;
    /** @return Separate serial receive scalar, absent exactly when this owner is C0. */
    std::optional<PooledPinWireSpan> replyCountDestination()const noexcept;
    /** @return Original typed owner PinHeader32, not its64-byte slot capacity. */
    PooledPinWireSpan pinHeaderSource()const noexcept;
    /** @return Exact original ordered owner coordinate in the coordinator metadata ingress. */
    PooledPinWireSpan pinHeaderDestination()const noexcept;
    /** @return Complete actual P×PinHeader32 frontier required by each owner, without allocation padding. */
    PooledPinWireSpan frontierSource()const noexcept;
    /** @return Existing owner frame pin_headers bank, never another frontier allocation. */
    PooledPinWireSpan frontierDestination()const noexcept;
    /** @return Original reverse channel geometry; native/rank gaps remain hard errors. */
    CapturedTransferChannelMemory returnChannelMemory()const;
    /** @return Checked original coordinator plan; no new source authority is minted. */
    const PooledImmutableCoordinatorNativePlan& coordinator()const noexcept{return coordinator_;}
    /** @return Checked original addressed owner plan. */
    const PooledImmutableOwnerNativePlan& owner()const noexcept{return owner_;}
    /** @return Complete immutable source/geometry/coordinate identity, excluding mutable counts. */
    uint64_t identity()const noexcept{return identity_;}
    /** @return Stable distinct nominal message identity for the original edge/wire role. */
    uint64_t messageKey(PooledPinWireRole role)const;
    /**
     * @brief Reuse exact current single-participant original producer spans, without replacing its supported path.
     * @param storage Actual original prepared coordinator owner retained by ModelContext.
     * @return Complete existing Pin schema with original tensor descriptor live_bytes.
     * @throws std::invalid_argument For a remote or multi-owner direct borrow requiring actual directed-offset machinery.
     */
    PooledNativeOperandSpans localPinInputs(const PooledImmutableCoordinatorPreparedStorage& storage)const;
private:
    /** @brief Only source/cohort authentication constructs an edge. */
    PooledImmutablePinEdge(PooledImmutableCoordinatorNativePlan coordinator,PooledImmutableOwnerNativePlan owner);
    PooledImmutableCoordinatorNativePlan coordinator_;
    PooledImmutableOwnerNativePlan owner_;
    uint64_t identity_=0;
};
/** @brief Complete original mapped endpoint/operand owners for a real cross-vendor initial-wave edge. */
class PooledImmutableCapturedPinEdge final
{
public:
    /**
     * @brief Bind admitted original roots and materialize exactly one already-priced channel through TransferEngine, outside capture.
     * @param coordinator Actual original prepared coordinator owner.
     * @param owner Actual addressed prepared physical owner.
     * @param memory Same original PMA that owns both local participant allocations and the channel.
     * @param senderSetupStream Exact original ordinary source setup stream.
     * @param receiverSetupStream Exact original ordinary receiver setup stream.
     * @param nativeCompletion Original admitted native TP completion bank; mandatory for homogeneous endpoints, absent only for cross-vendor mapped transport.
     * @return Retained original endpoint bindings; no native communicator or runtime eligibility is issued.
     * @throws std::exception For wrong cohort/generation/PMA/roots/endpoints, capture-time binding or a missing native transport.
     */
    static std::shared_ptr<const PooledImmutableCapturedPinEdge> bind(
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PhysicalMemoryAuthority> memory,
        void* senderSetupStream,void* receiverSetupStream,
        std::shared_ptr<const PooledImmutableNativeCompletionStorage> nativeCompletion = {});
    /** @brief Cold failure and final graph-owner destruction release the exact original invocation binding. */
    ~PooledImmutableCapturedPinEdge() noexcept;
    /** @return Original immutable edge declaration. */
    const PooledImmutablePinEdge& declaration()const noexcept{return declaration_;}
    /** @return Retained actual original coordinator owner. */
    const std::shared_ptr<PooledImmutableCoordinatorPreparedStorage>& coordinatorStorage()const noexcept{return coordinator_;}
    /** @return Retained actual original physical owner. */
    const std::shared_ptr<PooledImmutableOwnerPreparedStorage>& ownerStorage()const noexcept{return owner_;}
    /** @brief Reject any original physical/PMA/cohort drift, without reading device values. */
    void requireCurrent()const;
    /** @return Exact original invocation and command receive inputs for actual existing Owner Begin/Pin stages. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Original captured endpoint for one nominal phase; C0 commands have no binding. */
    const PooledImmutableTransferBinding& binding(PooledPinWireRole role,CapturedTransferEndpoint endpoint)const;
    /** @return Complete immutable endpoint/root/source-slice identity for graph caching. */
    uint64_t captureIdentity()const;
private:
    friend class PooledImmutableDirectedTransport;
    friend class PooledImmutableRetainedParticipants;
    /** @brief Original source owner invokes this only after every captured parent/borrower retired. */
    void retireOriginalInvocation() const;
    mutable bool owns_invocation_ = false; ///< Cold RAII binding ownership, never GPU progress.

    friend class PooledImmutableCapturedPinReturn;
    friend class PooledImmutableCapturedFreeWave;
    friend class PooledImmutableCapturedVictimWave;
    friend class PooledImmutableCapturedSourceWave;
    friend class PooledImmutableCapturedPublicationWave;
    friend class PooledImmutableCapturedReadWave;
    friend class PooledImmutableCapturedRetirementWave;
    /** @brief Construct only after full original endpoint/authority validation. */
    PooledImmutableCapturedPinEdge(PooledImmutablePinEdge declaration,
        std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator,
        std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner,std::shared_ptr<PhysicalMemoryAuthority> memory,
        PooledImmutableTransferChannel channel,PooledImmutableTransferBinding invocationSend,PooledImmutableTransferBinding invocationReceive,
        PooledImmutableTransferBinding frontierSend,PooledImmutableTransferBinding frontierReceive,
        std::optional<PooledImmutableTransferBinding> commandSend,std::optional<PooledImmutableTransferBinding> commandReceive);
    PooledImmutablePinEdge declaration_;
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> coordinator_;
    std::shared_ptr<PooledImmutableOwnerPreparedStorage> owner_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    uint64_t coordinator_identity_,owner_identity_;
    PooledImmutableTransferChannel channel_; // Same priced source channel, phase-reused by the free wave.
    PooledImmutableTransferBinding invocation_send_,invocation_receive_,frontier_send_,frontier_receive_;
    std::optional<PooledImmutableTransferBinding> command_send_,command_receive_;
};
/** @brief Original reverse Pin reply/header bindings, retaining the same forward/cohort/PMA owner. */
class PooledImmutableCapturedPinReturn final
{
public:
    /** @brief Bind original owner reply/count/header and coordinator serial receive banks outside capture.
     * @param forward Original admitted forward binding, also owns the eventual full frontier send.
     * @param memory Same original PMA as both actual prepared participants.
     * @param ownerStream Exact ordinary owner setup stream. @param coordinatorStream Exact source setup stream.
     * @return Actual original reverse channel/operands; no native/rank capability is issued. */
    static std::shared_ptr<const PooledImmutableCapturedPinReturn> bind(std::shared_ptr<const PooledImmutableCapturedPinEdge> forward,
        std::shared_ptr<PhysicalMemoryAuthority> memory,void* ownerStream,void* coordinatorStream);
    /** @return Same original complete directed edge. */
    const std::shared_ptr<const PooledImmutableCapturedPinEdge>& forward()const noexcept{return forward_;}
    /** @brief Reject any original PMA/root/cohort or captured storage drift. */
    void requireCurrent()const;
    /** @return Exact returned Grant or PinHeader endpoint; C0 has no Grant message. */
    const PooledImmutableTransferBinding& binding(PooledPinWireRole role,CapturedTransferEndpoint endpoint)const;
    /** @return Existing actual AcquirePins/SealPins schema over the serial bank/full ordered frontier. */
    PooledNativeOperandSpans coordinatorInputs(pooled_native::CoordinatorAction action)const;
    /** @return Original forward plus every actual reverse binding/capture identity. */
    uint64_t captureIdentity()const;
private:
    friend class PooledImmutableDirectedTransport;
    friend class PooledImmutableCapturedFreeWave;
    friend class PooledImmutableCapturedVictimWave;
    friend class PooledImmutableCapturedSourceWave;
    friend class PooledImmutableCapturedPublicationWave;
    friend class PooledImmutableCapturedReadWave;
    friend class PooledImmutableCapturedRetirementWave;
    /** @brief Construct only after original roots and public transfer bindings are admitted/retained. */
    PooledImmutableCapturedPinReturn(std::shared_ptr<const PooledImmutableCapturedPinEdge> forward,
        std::shared_ptr<PhysicalMemoryAuthority> memory,PooledImmutableTransferChannel channel,
        PooledImmutableTransferBinding headerSend,PooledImmutableTransferBinding headerReceive,
        std::optional<PooledImmutableTransferBinding> replySend,std::optional<PooledImmutableTransferBinding> replyReceive);
    std::shared_ptr<const PooledImmutableCapturedPinEdge> forward_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    PooledImmutableTransferChannel channel_; // Same reverse channel; no free-wave payload allocation.
    PooledImmutableTransferBinding header_send_,header_receive_;
    std::optional<PooledImmutableTransferBinding> reply_send_,reply_receive_;
};
}
namespace llaminar2
{
/** @brief Exact two participant-local graph spans; there is never a nested multi-device graph. */
struct PooledPinParticipantNodes {std::string first,last;};
/** @brief One original owner's exact graph names for the shared receive-bank phase lifetime. */
struct PooledPinReplyOwnerNodes
{
    uint32_t coordinate;
    std::string commandSent,replyReceived,replyAcquired,headerReceived;
    /** @brief A real direct local consumer replaces only transport, never a protocol completion. */
    enum class Path {Remote,DirectLocal} path=Path::Remote;
};
/**
 * @brief Install the actual shared-bank read→write and serial receive→consume lifetime edges.
 * @param graph Original coordinator-local graph with all actual nodes already declared.
 * @param original Original full ordered coordinator cohort; no capacity choice is made.
 * @param owners Every original coordinate exactly once, with no Grant nodes for C0.
 * @note All command sends precede every receiver; each receive→AcquirePins→header receive
 * precedes the next owner. Thus an early response can never overwrite a delayed peer's
 * command source, nor can a smaller replay consume a previous owner's incoming count.
 */
void wirePooledPinReplyBankLifetime(ComputeGraph& graph,const qwen4exp::PooledImmutableCoordinatorNativePlan& original,
    std::span<const PooledPinReplyOwnerNodes> owners,const std::string& commandsProduced);
/** @brief Authenticate the actual original owner Pin/send/read retirement chain before free-wave insertion.
 * @param graph Original owner-local graph. @param original Exact retained Pin return/channel owner.
 * @param acquiredPins Actual owner AcquirePins node to authenticate, never just a same-root phase name.
 * @throws std::invalid_argument Before graph mutation for missing/foreign/out-of-order phases, roots or endpoints. */
void requirePooledPinOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinReturn>& original,const std::string& acquiredPins);
/** @brief Authenticate the actual full ordered Pin-frontier send chain before its source ingress can be overwritten. */
void requirePooledPinFrontierCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinReturn>> owners,const std::string& completion);
/** @brief Append actual owner Pin→Grant send→SealPins→Header send→frontier receive→AcquirePins. */
PooledPinParticipantNodes appendPooledPinOwnerReturn(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinReturn> edge,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& originalPin,const std::string& prefix);
/**
 * @brief Append the complete original ordered reply gather, SealPins and exact full-frontier fanout.
 * @param graph Original coordinator graph. @param arena Same original prepared root arena.
 * @param owners Actual original return owners in declared coordinate order, including C0.
 * @param code Original coordinator backend bundle. @param commandSent Actual forward last node for each owner.
 * @param prefix Unique graph namespace. @return First receive and last frontier send for later Availability phase wiring.
 */
PooledPinParticipantNodes appendPooledPinCoordinatorReturn(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinReturn>> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,std::span<const std::string> commandSent,
    const std::string& commandsProduced,const std::string& prefix);
/**
 * @brief Append explicit send nodes after their actual original Begin/PinCommands producers.
 * @param graph Original coordinator-local graph. @param arena Its original registered root arena.
 * @param edge Exact prepared cross-vendor binding. @param beginProducer Actual original Coordinator Begin node.
 * @param commandProducer Actual original Coordinator PinCommands node. @param prefix Unique node namespace.
 * @return Invocation/command send endpoints with explicit serialized producer dependencies.
 */
PooledPinParticipantNodes appendPooledPinSender(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinEdge> edge,const std::string& beginProducer,
    const std::string& commandProducer,const std::string& prefix);
/**
 * @brief Append receive Invocation -> actual Owner Begin -> receive commands -> actual Owner Pin.
 * @param graph Original owner-local graph. @param arena Original owner root arena.
 * @param edge Same prepared original directed binding. @param code Original native owner backend bundle.
 * @param prefix Unique node namespace. @return Complete initial-wave first/last nodes for later SealPins dependencies.
 */
PooledPinParticipantNodes appendPooledPinReceiver(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedPinEdge> edge,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& prefix);
}
