/**
 * @file PooledImmutableParticipantPublicationGraph.h
 * @brief Original participant-local publication, acquired-publication acknowledgment and actual Ready frontier.
 *
 * Every remote edge phase-reuses the already admitted original two channels;
 * the local participant uses its real owner producer and nominal-kind witness.
 * Private source responses and physical pins stay retained through subsequent
 * reader/release phases. This component grants no serving/native eligibility.
 */
#pragma once
#include "PooledImmutableParticipantSourceGraph.h"
namespace llaminar2::qwen4exp {
/** @brief Fixed nominal messages; header60 is required record56+device kind4, never slot padding. */
enum class PooledPublicationWireRole {PublicationHeader,PublicationGate,PublicationFrontier,AcquiredHeader,ReadyGate,ReadyHeader};
/** @brief Real remote source/fill successor over the exact original retained channels and PMA. */
class PooledImmutableCapturedPublicationWave final {
public:
    /** @brief Authenticate actual source/fill roots and bind fixed exact metadata without a new root/channel. */
    static std::shared_ptr<const PooledImmutableCapturedPublicationWave> bind(std::shared_ptr<const PooledImmutableCapturedSourceWave> source,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Reauthenticate original source/cohort/physical/transport generations. */
    void requireCurrent()const;
    /** @return Original source holder, retaining its provider and worker through true last reader. */
    const std::shared_ptr<const PooledImmutableCapturedSourceWave>& sourceWave()const noexcept{return source_;}
    /** @return Original fixed metadata endpoint; no row count or model value is read on host. */
    const CapturedTransferBinding& binding(PooledPublicationWireRole role,CapturedTransferEndpoint side)const;
    /** @return Original root subrange; exact wire bytes are distinct from storage capacity. */
    PooledPinWireSpan source(PooledPublicationWireRole role)const;
    PooledPinWireSpan destination(PooledPublicationWireRole role)const;
    /** @return Complete original root/channel/role/capture identity, excluding mutable protocol values. */
    uint64_t captureIdentity()const;
private:
    using Pair=std::pair<CapturedTransferBinding,CapturedTransferBinding>;
    /** @brief Only a fully checked original source publishes this immutable holder. */
    PooledImmutableCapturedPublicationWave(std::shared_ptr<const PooledImmutableCapturedSourceWave> source,std::shared_ptr<PhysicalMemoryAuthority> memory):source_(std::move(source)),memory_(std::move(memory)){}
    std::shared_ptr<const PooledImmutableCapturedSourceWave> source_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::array<std::optional<Pair>,6> bindings_;
};
/** @brief One real local binding or one actual remote publication channel holder. */
using PooledImmutablePublicationParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,std::shared_ptr<const PooledImmutableCapturedPublicationWave>>;
}
namespace llaminar2 {
/** @brief Actual Ready producer graph descriptor; no new reader/storage lifetime authority is created. */
struct PooledImmutablePublicationCompletion {
    PooledImmutableSourceFillCompletion source;
    std::vector<qwen4exp::PooledImmutablePublicationParticipant> owners;
    std::vector<std::string> publicationReceived,publicationImported,publicationGateSent,publicationFrontierSent;
    std::vector<std::string> acquiredReceived,acquiredImported,readyGateSent,readyReceived,readyImported;
    std::string publicationValidated,acquiredValidated,readyValidated,ready;
    std::string localAcquire,localMakeReady,localReadyReceipt;
    PooledPinParticipantNodes coordinator;
};
/** @brief Append actual remote owner Publish→Acquire→Ready producers, after authenticating the complete original fill graph. */
PooledPinParticipantNodes appendPooledPublicationOwner(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedPublicationWave> original,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& fillOutcome,const std::string& prefix);
/** @brief Append complete fixed metadata acquisition and success-only original coordinator publication. */
PooledImmutablePublicationCompletion appendPooledPublicationCoordinator(ComputeGraph& graph,BufferArena& arena,
    const PooledImmutableSourceFillCompletion& source,std::span<const qwen4exp::PooledImmutablePublicationParticipant> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& prefix);
/** @brief Authenticate original remote Ready send and complete actual owner publication/source producers before row-read insertion. */
void requirePooledPublicationOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedPublicationWave>& original,const std::string& readySent);
/** @brief Authenticate actual nominal producers/acquisitions and every bank-read frontier before any reader successor appends. */
void requirePooledPublicationCompletion(const ComputeGraph& graph,const PooledImmutablePublicationCompletion& original);
}
