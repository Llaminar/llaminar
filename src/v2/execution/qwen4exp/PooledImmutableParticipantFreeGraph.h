/**
 * @file PooledImmutableParticipantFreeGraph.h
 * @brief Original Availability/free-plan operands and participant-local free-allocation endpoints.
 *
 * This wave retains the admitted Pin cohort, source-owned banks, original
 * UInt64 publications and same PMA. It adds no resolver, storage ledger or
 * native factory. Only already-supported cross-vendor captured channels can
 * bind; the complete distributed transaction remains explicitly uninstalled.
 */
#pragma once
#include "PooledImmutableParticipantPinGraph.h"
namespace llaminar2::qwen4exp {
/** @brief Nominal free-wave wire actions; their keys do not alias initial Pin actions. */
enum class PooledFreeWireRole {Availability,Plan,Decisions,Grants,Receipt,Frontier};
/** @brief Immutable exact free-wave geometry projected from the original authenticated directed Pin edge. */
class PooledImmutableFreeEdge final {
public:
    /** @brief Retain the complete original remote source/cohort/owner plan; never invent a new physical owner.
     * @param original Authenticated distinct original allocators; native/rank metadata does not enable transport.
     * @return Exact remote declaration. @throws std::invalid_argument For a co-located edge, whose existing local P1 graph remains authoritative. */
    static PooledImmutableFreeEdge fromOriginal(PooledImmutablePinEdge original);
    /** @return Same original edge/cohort/PMA layout declarations. */
    const PooledImmutablePinEdge& original()const noexcept{return original_;}
    /** @return True for a positive admitted owner; runtime zero still exchanges counted metadata. */
    bool hasPayload()const noexcept{return original_.hasCommands();}
    /** @return Exact named original source span for this nominal wire role. */
    PooledPinWireSpan source(PooledFreeWireRole role)const;
    /** @return Exact named original receiver span, capacity separate from each device-owned live extent. */
    PooledPinWireSpan destination(PooledFreeWireRole role)const;
    /** @return Original absolute frame.decisions DirectedSlice16 produced by actual PlanFree. */
    PooledPinWireSpan decisionSlice()const noexcept;
    /** @return Original owner ingress UInt64 count for decision reception, absent exactly for C0. */
    std::optional<PooledPinWireSpan> decisionCount()const noexcept{return original_.incomingCount();}
    /** @return Original owner reply_bytes publication, never a converted row count. */
    PooledPinWireSpan grantCount()const noexcept{return original_.replyCountSource();}
    /** @return Original serial coordinator receive scalar, absent exactly for C0. */
    std::optional<PooledPinWireSpan> incomingGrantCount()const noexcept{return original_.replyCountDestination();}
    /** @return This addressed owner's exact admitted decision maximum, not the global source-bank capacity. */
    size_t maximumDecisionBytes()const noexcept;
    /** @return Maximum simultaneous message over the original forward channel across implemented Pin/free phases. */
    size_t forwardChannelMaximum()const noexcept;
    /** @return Maximum message over the original reverse channel, including C0 metadata. */
    size_t reverseChannelMaximum()const noexcept;
    /** @return Distinct immutable nominal message key within the same original channel. */
    uint64_t messageKey(PooledFreeWireRole role)const;
    /** @return Original directed owner coordinate. */
    uint32_t coordinate()const noexcept{return original_.coordinate();}
private:
    /** @brief Source/cohort identity is retained by value only after the original edge authenticated it. */
    explicit PooledImmutableFreeEdge(PooledImmutablePinEdge original):original_(std::move(original)){}
    PooledImmutablePinEdge original_;
};
/** @brief Combine the complete retained free binding identity with its nominal action and endpoint.
 * @param binding Original roots/channel/geometry/generation identity, never a mutable device value.
 * @param role Exact free-wave action. @param endpoint Actual local producer or consumer.
 * @return Stable nonzero variant; endpoint mixing precedes nonzero normalization. */
inline constexpr uint64_t pooledFreeEndpointCaptureIdentity(uint64_t binding,PooledFreeWireRole role,CapturedTransferEndpoint endpoint)noexcept
{return ((((binding^(16+static_cast<unsigned>(role)))*1099511628211ULL)^static_cast<unsigned>(endpoint))*1099511628211ULL)|1;}
/** @brief Exact original free-wave bindings over the retained Pin channels; no extra payload/channel allocation. */
class PooledImmutableCapturedFreeWave final {
public:
    /** @brief Bind original operands outside capture, reusing the same admitted forward/reverse channel owners.
     * @param pin Original complete Pin reply edge retaining both prepared participants and both channels.
     * @param memory Same original PMA; a foreign or replaced authority fails before any binding.
     * @return Retained participant-local free-wave operands; native/factory eligibility remains guarded. */
    static std::shared_ptr<const PooledImmutableCapturedFreeWave> bind(
        std::shared_ptr<const PooledImmutableCapturedPinReturn> pin,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Authenticate original prepared roots, generations, ordered membership and PMA without a device read. */
    void requireCurrent()const;
    /** @return Original Pin owner, also authenticates the preceding phase's actual graph endpoints. */
    const std::shared_ptr<const PooledImmutableCapturedPinReturn>& pin()const noexcept{return pin_;}
    /** @return Immutable exact original free geometry. */
    const PooledImmutableFreeEdge& declaration()const noexcept{return declaration_;}
    /** @return Actual fixed/sliced/counted endpoint; C0 never has a decision/Grant transaction. */
    const PooledImmutableTransferBinding& binding(PooledFreeWireRole role,CapturedTransferEndpoint endpoint)const;
    /** @return Original owner Invocation plus exact acquired free-plan/frontier operands. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Original coordinator's exact Availability/Grant/completion inputs. */
    PooledNativeOperandSpans coordinatorInputs(pooled_native::CoordinatorAction action)const;
    /** @return Full original Pin plus every retained free binding identity; live values stay excluded. */
    uint64_t captureIdentity()const;
private:
    using Pair=std::pair<PooledImmutableTransferBinding,PooledImmutableTransferBinding>;
    using Bindings=std::array<std::optional<Pair>,6>;
    /** @brief Construct only after actual original roots/PMA and all exact endpoint bindings are checked. */
    PooledImmutableCapturedFreeWave(std::shared_ptr<const PooledImmutableCapturedPinReturn> pin,
        std::shared_ptr<PhysicalMemoryAuthority> memory,PooledImmutableFreeEdge declaration,Bindings bindings)
        :pin_(std::move(pin)),memory_(std::move(memory)),declaration_(std::move(declaration)),bindings_(std::move(bindings)){}
    std::shared_ptr<const PooledImmutableCapturedPinReturn> pin_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    PooledImmutableFreeEdge declaration_;
    Bindings bindings_;
};
}
namespace llaminar2 {
/** @brief Authenticate original concrete owner free send/read chain before victim header/count-bank reuse. */
void requirePooledFreeOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedFreeWave>& original,const std::string& acquiredFree);
/** @brief Authenticate complete original coordinator free frontier/send/read chain before victim planning or receipt overwrite. */
void requirePooledFreeCoordinatorCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedFreeWave>> original,const std::string& sentFree);
/** @brief Append actual Availability→plan receive→AcceptFree/ExecuteFree→Grant/receipt sends→frontier acquire. */
PooledPinParticipantNodes appendPooledFreeOwnerWave(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedFreeWave> edge,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& acquiredPins,const std::string& prefix);
/** @brief Append actual complete Availability gather, PlanFree, compact plan fanout, serial Grant acquire,
 * complete receipt frontier and fanout. Each receive bank is reused only after its original readers/senders retire. */
PooledPinParticipantNodes appendPooledFreeCoordinatorWave(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedFreeWave>> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& pinFrontiersComplete,const std::string& prefix);
/** @brief Install the actual ordered Availability40P acquisition only after the complete preceding Pin send frontier.
 * @param graph Original coordinator-local graph. @param original Complete original ordered cohort.
 * @param receives Exactly one availability receiver for every original owner, including C0.
 * @param pinFrontiersComplete Authenticated last Pin-frontier send. @param planFree Actual PlanFree consumer.
 * @note No source ingress byte can be overwritten while any owner still reads the preceding Pin frontier. */
void wirePooledFreeAvailabilityLifetime(ComputeGraph& graph,const qwen4exp::PooledImmutableCoordinatorNativePlan& original,
    std::span<const std::string> receives,const std::string& pinFrontiersComplete,const std::string& planFree);
}
