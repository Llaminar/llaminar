/**
 * @file PooledImmutableParticipantReadGraph.h
 * @brief Original pinned-hit delivery and original-order PLE reader completion.
 *
 * The read program retains the real publication/source chain and two original
 * channels. Successful and failed paths issue the same fixed metadata program;
 * data messages use only original UInt64 live byte producers. The completion
 * descriptor names actual graph readers, never a host decision or pin ledger.
 */
#pragma once
#include "PooledImmutableParticipantPublicationGraph.h"
namespace llaminar2::qwen4exp {
/** @brief Exact read protocol roles on the existing forward/reverse channels. */
enum class PooledReadWireRole {Outcome,Requests,Receipt,Rows};
/** @brief Retained original remote owner/source binding; no new root or channel. */
class PooledImmutableCapturedReadWave final {
public:
    /** @brief Bind original directed Grant64/count/native-row producers under their sole original PMA. */
    static std::shared_ptr<const PooledImmutableCapturedReadWave> bind(
        std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Reject original storage/transport/runtime/geometry drift before native submission. */
    void requireCurrent()const;
    /** @return Actual prior publication holder, retaining source worker, roots and original channels. */
    const std::shared_ptr<const PooledImmutableCapturedPublicationWave>& publication()const noexcept{return publication_;}
    /** @return Exact immutable original source or receiver storage declaration. */
    PooledPinWireSpan source(PooledReadWireRole role)const;
    PooledPinWireSpan destination(PooledReadWireRole role)const;
    /** @return Original fixed/directed/counted endpoint; C0 requests/rows have no endpoint. */
    const PooledImmutableTransferBinding& binding(PooledReadWireRole role,CapturedTransferEndpoint side)const;
    /** @return Original acquired Invocation48, Grant64/count and terminal source outcome operands. */
    PooledNativeOperandSpans ownerInputs()const;
    /** @return Complete frozen original pointer/root/count/descriptor/channel identity, excluding live values. */
    std::uint64_t captureIdentity()const;
private:
    using Pair=std::pair<PooledImmutableTransferBinding,PooledImmutableTransferBinding>;
    /** @brief Publish an immutable holder only through the checked cold binder. */
    PooledImmutableCapturedReadWave(std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication,
        std::shared_ptr<PhysicalMemoryAuthority> memory):publication_(std::move(publication)),memory_(std::move(memory)){}
    std::shared_ptr<const PooledImmutableCapturedPublicationWave> publication_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::array<std::optional<Pair>,4> bindings_;
};
/** @brief Original local physical producer or real remote read binding, in canonical owner order. */
using PooledImmutableReadParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,std::shared_ptr<const PooledImmutableCapturedReadWave>>;
}
namespace llaminar2 {
/**
 * @brief Actual issued-read inventory and last native-row consumer for retirement insertion.
 *
 * `sealed` is the real SealReads18 join: success becomes RowsReady; a truthful
 * failed source remains ReadPlanned after every fixed receive/import drains.
 * `output` is the original decoder's real last native-row reader and completes
 * on both paths (failed execution writes no output). It follows `sealed` and
 * every request-send, receipt/row receive and import below. Terminal code must
 * authenticate this descriptor; a phase flag alone cannot authorize release.
 */
struct PooledImmutableReadCompletion {
    PooledImmutablePublicationCompletion publication;
    std::vector<qwen4exp::PooledImmutableReadParticipant> owners;
    std::vector<std::string> outcomeSent,requestSent,receiptReceived,rowsReceived,imported;
    std::string planned,sealed,output;
    qwen4exp::PooledNativeTensorSpan outputSpan;
    PooledPinParticipantNodes coordinator;
};
/** @brief Append actual remote pinned-read/copy/receipt send after authenticating its original Ready producer. */
PooledPinParticipantNodes appendPooledReadOwner(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedReadWave> original,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& readySent,const std::string& prefix);
/** @brief Append exact hit requests, serial row imports, complete map seal and original-order native decoder. */
PooledImmutableReadCompletion appendPooledReadCoordinator(ComputeGraph& graph,BufferArena& arena,
    const PooledImmutablePublicationCompletion& publication,std::span<const qwen4exp::PooledImmutableReadParticipant> owners,
    qwen4exp::PooledNativeTensorSpan output,const std::string& prefix,const std::string& outputName);
/** @brief Authenticate actual original producers, operand roots and complete issued-read/output ordering before retirement. */
void requirePooledReadCompletion(const ComputeGraph& graph,const PooledImmutableReadCompletion& original);
/** @brief Authenticate exact original owner receipt/row send chain before any physical release successor. */
void requirePooledReadOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedReadWave>& original,const std::string& rowsSent);
/** @return Actual original owner's final row-send/receipt stage in this cold recipient graph; missing/duplicate holders reject. */
std::string findPooledReadOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedReadWave>& original);

}
