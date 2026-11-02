/**
 * @file PooledImmutableParticipantVictimGraph.h
 * @brief Exact original second allocation-wave declarations and participant-local endpoints.
 *
 * The original free prefix, physical owner and policy remain authoritative.
 * Victim payloads use the actual original suffix descriptor and current-wave
 * Grant publication. No source fill, owner frontier fanout or native factory
 * is manufactured by this bounded uninstalled component.
 */
#pragma once
#include "PooledImmutableParticipantFreeGraph.h"
namespace llaminar2::qwen4exp {
/** @brief Only actual victim-wave consumers determine these messages; no unused owner receipt fanout. */
enum class PooledVictimWireRole {Plan,Decisions,Grants,Receipt};
/** @brief Immutable original owner/coordinator victim suffix geometry, without another plan or ledger. */
class PooledImmutableVictimEdge final {
public:
    /** @brief Retain the original remote free-wave declaration; no capacity/membership choice is made. */
    static PooledImmutableVictimEdge fromFree(PooledImmutableFreeEdge original){return PooledImmutableVictimEdge(std::move(original));}
    /** @return Same original free/pin declarations and allocators. */
    const PooledImmutableFreeEdge& previous()const noexcept{return previous_;}
    /** @return Positive admitted payload bank; runtime empty still uses counted metadata. */
    bool hasPayload()const noexcept{return previous_.hasPayload();}
    /** @return Original addressed coordinate including C0. */
    uint32_t coordinate()const noexcept{return previous_.coordinate();}
    /** @return Exact original source bank or fixed header for this nominal role. */
    PooledPinWireSpan source(PooledVictimWireRole role)const;
    /** @return Exact original receive bank, whose capacity is never its live wire count. */
    PooledPinWireSpan destination(PooledVictimWireRole role)const;
    /** @return Original victim_slices[i]16, absolute in frame.decisions after the free prefix. */
    PooledPinWireSpan decisionSlice()const noexcept;
    /** @return Original mutable count publication, absent exactly for C0. */
    std::optional<PooledPinWireSpan> decisionCount()const noexcept{return previous_.decisionCount();}
    /** @return Original current-wave owner reply byte count. */
    PooledPinWireSpan grantCount()const noexcept{return previous_.grantCount();}
    /** @return Original serial coordinator incoming count. */
    std::optional<PooledPinWireSpan> incomingGrantCount()const noexcept{return previous_.incomingGrantCount();}
    /** @return Exact addressed decision maximum, independent of the source bank's admitted span. */
    size_t maximumDecisionBytes()const noexcept{return previous_.maximumDecisionBytes();}
    /** @return Original nominal message identity distinct from Pin and free actions. */
    uint64_t messageKey(PooledVictimWireRole role)const;
private:
    /** @brief Construct only from the original authenticated remote free declaration. */
    explicit PooledImmutableVictimEdge(PooledImmutableFreeEdge original):previous_(std::move(original)){}
    PooledImmutableFreeEdge previous_;
};
/** @brief Complete immutable original binding/action/endpoint identity; live counts/offsets are excluded. */
inline constexpr uint64_t pooledVictimEndpointCaptureIdentity(uint64_t binding,PooledVictimWireRole role,CapturedTransferEndpoint endpoint)noexcept
{return ((((binding^(32+static_cast<unsigned>(role)))*1099511628211ULL)^static_cast<unsigned>(endpoint))*1099511628211ULL)|1;}
/** @brief Actual original victim operands over the same retained two channels and PMA. */
class PooledImmutableCapturedVictimWave final {
public:
    /** @brief Bind original suffix/header/count owners outside capture, adding no packet/channel/root allocation.
     * @param previous Original complete free-wave binding. @param memory Same original PMA.
     * @return Retained actual endpoint bindings; native/rank/factory eligibility stays guarded. */
    static std::shared_ptr<const PooledImmutableCapturedVictimWave> bind(
        std::shared_ptr<const PooledImmutableCapturedFreeWave> previous,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Reject original root/authority/cohort/generation drift without observing device data. */
    void requireCurrent()const;
    /** @return Original prior free binding, also owns its actual Pin channels. */
    const std::shared_ptr<const PooledImmutableCapturedFreeWave>& previous()const noexcept{return previous_;}
    /** @return Immutable original suffix geometry. */
    const PooledImmutableVictimEdge& declaration()const noexcept{return declaration_;}
    /** @return Exact fixed/sliced/counted local endpoint; C0 never has decision/Grant bindings. */
    const CapturedTransferBinding& binding(PooledVictimWireRole role,CapturedTransferEndpoint endpoint)const;
    /** @return Exact original Invocation plus acquired victim header/count prefix; no dummy C0 inputs. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Exact original PlanVictims/Grant/complete-receipt inputs. */
    PooledNativeOperandSpans coordinatorInputs(pooled_native::CoordinatorAction action)const;
    /** @return Complete original free plus all actual victim bindings. */
    uint64_t captureIdentity()const;
private:
    using Pair=std::pair<CapturedTransferBinding,CapturedTransferBinding>;
    using Bindings=std::array<std::optional<Pair>,4>;
    /** @brief Construct only after all original bindings and same-PMA retention succeed. */
    PooledImmutableCapturedVictimWave(std::shared_ptr<const PooledImmutableCapturedFreeWave> previous,
        std::shared_ptr<PhysicalMemoryAuthority> memory,PooledImmutableVictimEdge declaration,Bindings bindings)
        :previous_(std::move(previous)),memory_(std::move(memory)),declaration_(std::move(declaration)),bindings_(std::move(bindings)){}
    std::shared_ptr<const PooledImmutableCapturedFreeWave> previous_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    PooledImmutableVictimEdge declaration_;
    Bindings bindings_;
};
}
namespace llaminar2 {
/** @brief Require the actual original owner victim receipt-send chain; failure precedes source graph mutation. */
void requirePooledVictimOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedVictimWave>& original,const std::string& sentVictims);
/** @brief Require original ALL-owner victim receipt/read/send chain before source request publication. */
void requirePooledVictimCoordinatorCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedVictimWave>> owners,const std::string& completeVictims);
/** @brief Append actual victim header/count receive→Accept/ExecuteVictim→current-wave Grant/receipt send.
 * @param acquiredFree Actual original owner free completion, authenticated before graph mutation. */
PooledPinParticipantNodes appendPooledVictimOwnerWave(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedVictimWave> edge,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& acquiredFree,const std::string& prefix);
/** @brief Append actual PlanVictims, exact suffix fanout, serial Grant acquisition and ALL-owner CompleteVictims.
 * @param sentFree Authenticated actual complete original free frontier send/read completion.
 * @note Victim receipt frontier has only its actual coordinator consumer; there is no owner AcquireVictims. */
PooledPinParticipantNodes appendPooledVictimCoordinatorWave(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const qwen4exp::PooledImmutableCapturedVictimWave>> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& sentFree,const std::string& prefix);
}
