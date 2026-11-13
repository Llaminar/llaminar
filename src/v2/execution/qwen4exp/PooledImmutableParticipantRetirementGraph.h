/**
 * @file PooledImmutableParticipantRetirementGraph.h
 * @brief Real last-reader, exact release and physical retirement over original retained participants.
 *
 * Remote owners receive a nominal last-reader gate; local owners depend directly
 * on the actual same-device reader. Neither path invents source retirement,
 * releases private rows early or creates a self-channel/second memory authority.
 */
#pragma once
#include "PooledImmutableParticipantReadGraph.h"
namespace llaminar2::qwen4exp {
/** @brief Required exact terminal messages on the original two directed channels. */
enum class PooledRetirementWireRole {ReadersRetired,Releases,Retired};
/** @brief Closed original read successor retaining every source/root/channel owner through terminal. */
class PooledImmutableCapturedRetirementWave final {
public:
    /** @brief Validate the actual original read/PMA and bind only original count/header/payload roots. */
    static std::shared_ptr<const PooledImmutableCapturedRetirementWave> bind(std::shared_ptr<const PooledImmutableCapturedReadWave> read,std::shared_ptr<PhysicalMemoryAuthority> memory);
    /** @brief Reject source/provider/allocator/capture-generation drift before recording or replay. */
    void requireCurrent()const;
    /** @return Actual original read owner; no initialized communicator or replacement roots are adopted. */
    const std::shared_ptr<const PooledImmutableCapturedReadWave>& read()const noexcept{return read_;}
    /** @return Exact fixed/counted endpoint; C0 Releases has no wire endpoint. */
    const PooledImmutableTransferBinding& binding(PooledRetirementWireRole role,CapturedTransferEndpoint side)const;
    /** @return Original fixed admitted source/receiver span; capacity is never the live release extent. */
    PooledPinWireSpan source(PooledRetirementWireRole role)const;
    PooledPinWireSpan destination(PooledRetirementWireRole role)const;
    /** @return Original invocation and exact owner-native gate/outcome operands for the fixed phase. */
    PooledNativeOperandSpans ownerInputs(pooled_native::OwnerAction action)const;
    /** @return Complete retained root/count/channel/action identity, excluding device-live values. */
    std::uint64_t captureIdentity()const;
private:
    using Pair=std::pair<PooledImmutableTransferBinding,PooledImmutableTransferBinding>;
    /** @brief Only the checked cold binder publishes this immutable original-owner holder. */
    PooledImmutableCapturedRetirementWave(std::shared_ptr<const PooledImmutableCapturedReadWave> read,std::shared_ptr<PhysicalMemoryAuthority> memory):read_(std::move(read)),memory_(std::move(memory)){}
    std::shared_ptr<const PooledImmutableCapturedReadWave> read_;
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    std::array<std::optional<Pair>,3> bindings_;
};
/** @brief One direct actual local member or actual original remote terminal binding. */
using PooledImmutableRetirementParticipant=std::variant<std::shared_ptr<const PooledImmutableLocalOwnerBinding>,std::shared_ptr<const PooledImmutableCapturedRetirementWave>>;
}
namespace llaminar2 {
struct PooledImmutableReplicatedConsumerCompletion;
struct PooledImmutableReplicatedReaderCompletion;
/** @brief Concrete original terminal producer/read inventory; no physical pin or execution-state shadow. */
struct PooledImmutableRetirementCompletion {
    PooledImmutableReadCompletion read;
    std::shared_ptr<const PooledImmutableReplicatedReaderCompletion> readers; ///< Exact actual compute ACK frontiers, independent of physical owner count.
    std::vector<qwen4exp::PooledImmutableRetirementParticipant> owners;
    std::vector<std::string> gatePacked,gateSent,releaseReceived,releaseImported,retiredReceived,retiredImported;
    std::string readersRetired,planned,reconciled,snapshot,finished,observed;
    std::string localDrain,localRelease,localRetire;
    PooledPinParticipantNodes coordinator;
};
/** @brief Append real remote last-reader gate→drain→exact releases→actual Retire→nominal Retired60. */
PooledPinParticipantNodes appendPooledRetirementOwner(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave> original,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& lastReadSent,const std::string& prefix,
    const PooledImmutableReplicatedConsumerCompletion* consumer = nullptr);
/** @brief Retire a colocated physical owner after its actual compact-native consumer ACK. */
PooledPinParticipantNodes appendPooledRetirementOwnerAfterCompactConsumer(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave> original,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& lastReadSent,
    const std::string& prefix,const PooledImmutableCompactConsumerCompletion& consumer);
/** @brief Append full ordered local/remote terminal protocol behind the authenticated real output/drain node. */
PooledImmutableRetirementCompletion appendPooledRetirementCoordinator(ComputeGraph& graph,BufferArena& arena,
    const PooledImmutableReadCompletion& read,std::span<const qwen4exp::PooledImmutableRetirementParticipant> owners,
    std::shared_ptr<const qwen4exp::IPooledImmutableNativeKernels> code,const std::string& prefix);
/** @brief Authenticate every actual owner phase and send before replacement/reset/close insertion. */
void requirePooledRetirementOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave>& original,const std::string& retiredSent,
    const PooledImmutableReplicatedConsumerCompletion* consumer = nullptr);
/** @brief Authenticate the same physical release chain and its original compact ACK dependency. */
void requirePooledRetirementOwnerAfterCompactConsumer(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave>& original,
    const std::string& retiredSent,const PooledImmutableCompactConsumerCompletion& consumer);
/** @brief Authenticate true last reader, exact original phases/roots/channels and all reuse/commit edges. */
void requirePooledRetirementCompletion(const ComputeGraph& graph,const PooledImmutableRetirementCompletion& original);
/** @return Exact original Retired producer following the same typed consumer ACK in this cold actual graph. */
std::string findPooledRetirementOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave>& original,
    const PooledImmutableReplicatedConsumerCompletion& consumer);
/** @return Unique actual physical terminal following this consumer's validated compact ACK. */
std::string findPooledRetirementOwnerAfterCompactConsumer(const ComputeGraph& graph,
    const std::shared_ptr<const qwen4exp::PooledImmutableCapturedRetirementWave>& original,
    const PooledImmutableCompactConsumerCompletion& consumer);

}
