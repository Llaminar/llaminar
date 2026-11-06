/**
 * @file PooledImmutableParticipantPinGraph.cpp
 * @brief Exact original initial-wave operands and explicit participant-local transport nodes.
 *
 * Only the first directed wave is materialized here. Existing source/cohort/PMA
 * owners remain authoritative. No stage hides a coordinator, host decision or
 * collective; each endpoint uses the public captured transfer authority on its
 * own exact stream and original canonical roots.
 */
#include "PooledImmutableParticipantPinGraph.h"
#include "PooledImmutableLocalHotGraph.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableLocalHotPhaseStage.h"
#include "execution/mpi_orchestration/DeviceInventory.h"
#include "execution/qwen4exp/PooledImmutableDirectedChannelGeometry.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "PooledImmutableParticipantFreeGraph.h"
#include "PooledImmutableParticipantLocalGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include <algorithm>
#include <stdexcept>
#include <set>
#include <type_traits>
#include "kernels/qwen4exp/PooledImmutableNativeAliasPolicy.h"
namespace llaminar2::qwen4exp
{
namespace
{
namespace n=pooled_native;namespace p=n::p;
/** @brief Complete immutable field combination; no device value or structure padding is hashed. */
uint64_t combine(uint64_t first,uint64_t second){return ((first^second)*1099511628211ULL)|1;}
/** @brief Retain one original named subrange, omitting only a schema-declared C0 operand. */
void input(PooledNativeOperandSpans& result,PooledNativeOperand role,const PooledImmutableNativeRoots& roots,PooledPinWireSpan span)
{
    if(!span.bytes)return;
    const auto root=span.buffer==BufferId::QFN_POOLED_OWNER_FRAME?roots.frame:roots.native;
    if(!root || root->is_view() || root->transferStorageOwner()!=root.get() ||
       span.offset>root->size_bytes() || span.bytes>root->size_bytes()-span.offset)
        throw std::invalid_argument("Pooled pin operand lost its original canonical root extent");
    result[static_cast<size_t>(role)]=PooledNativeTensorSpan{root,span.buffer,span.offset,span.bytes};
}
}
PooledImmutablePinEdge PooledImmutablePinEdge::fromPlans(PooledImmutableCoordinatorNativePlan c,PooledImmutableOwnerNativePlan o)
{return PooledImmutablePinEdge(std::move(c),std::move(o));}
PooledImmutablePinEdge::PooledImmutablePinEdge(PooledImmutableCoordinatorNativePlan c,PooledImmutableOwnerNativePlan o)
    :coordinator_(std::move(c)),owner_(std::move(o))
{
    coordinator_.requireOwnerPlan(owner_);
    const auto& a=coordinator_.parameters();const auto& b=owner_.parameters();
    if(a.table!=b.table || a.source_storage_generation!=b.source_storage_generation ||
       a.policy.participants()!=b.owner.participants() || a.native.reply_capacity<b.native.reply_capacity ||
       (remote() && (!b.native.ingress || b.native.ingress>b.native.bytes || sizeof(n::OwnerIngress)>b.native.bytes-b.native.ingress)))
        throw std::invalid_argument("Pooled pin edge requires original complete source/cohort/ingress geometry");
    identity_=combine(combine(combine(n::Version,a.identity),b.identity),coordinate());
}
bool PooledImmutablePinEdge::capturedChannelTransport()const noexcept
{return remote() && ImmutableRowTableDirectedTransportPolicy::supports(sender(), receiver(), owner_.peerAccess());}
size_t PooledImmutablePinEdge::maximumCommandBytes()const noexcept
{return PooledImmutableDirectedMessageGeometry::fromPlan(owner_).commands();}
PooledPinWireSpan PooledImmutablePinEdge::commandSource()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.pin_commands,
    size_t{coordinator_.parameters().native.reply_capacity}*sizeof(p::OwnerCommand)};}
PooledPinWireSpan PooledImmutablePinEdge::commandSlice()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.pin_slices+size_t{coordinate()}*sizeof(n::DirectedSlice),sizeof(n::DirectedSlice)};}
PooledPinWireSpan PooledImmutablePinEdge::commandDestination()const noexcept
{return {BufferId::QFN_POOLED_OWNER_FRAME,owner_.parameters().frame.commands,maximumCommandBytes()};}
PooledPinWireSpan PooledImmutablePinEdge::invocationSource()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.invocation,sizeof(n::Invocation)};}
PooledPinWireSpan PooledImmutablePinEdge::invocationDestination()const noexcept
{return remote()?PooledPinWireSpan{BufferId::QFN_POOLED_OWNER_NATIVE,owner_.parameters().native.ingress+offsetof(n::OwnerIngress,invocation),sizeof(n::Invocation)}:invocationSource();}
std::optional<PooledPinWireSpan> PooledImmutablePinEdge::incomingCount()const noexcept
{
    if(!hasCommands())return {};
    if(remote())return PooledPinWireSpan{BufferId::QFN_POOLED_OWNER_NATIVE,owner_.parameters().native.ingress+offsetof(n::OwnerIngress,record_bytes),sizeof(uint64_t)};
    const auto slice=commandSlice();return PooledPinWireSpan{slice.buffer,slice.offset+offsetof(n::DirectedSlice,live_bytes),sizeof(uint64_t)};
}
CapturedTransferChannelMemory PooledImmutablePinEdge::channelMemory()const
{
    if(!capturedChannelTransport())throw std::invalid_argument("Pooled native/rank edge requires its actual admitted live-source-offset transport; mapped substitution is forbidden");
    return PooledImmutableDirectedChannelGeometry::fromPlans(coordinator_,owner_).forwardMemory();
}
PooledPinWireSpan PooledImmutablePinEdge::replySource()const noexcept
{return {BufferId::QFN_POOLED_OWNER_NATIVE,owner_.parameters().native.replies,PooledImmutableDirectedMessageGeometry::fromPlan(owner_).grants()};}
PooledPinWireSpan PooledImmutablePinEdge::replyCountSource()const noexcept
{return {BufferId::QFN_POOLED_OWNER_NATIVE,owner_.parameters().native.progress+offsetof(n::OwnerProgress,reply_bytes),sizeof(uint64_t)};}
PooledPinWireSpan PooledImmutablePinEdge::replyDestination()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.pin_commands,size_t{coordinator_.parameters().native.reply_capacity}*sizeof(p::Grant)};}
std::optional<PooledPinWireSpan> PooledImmutablePinEdge::replyCountDestination()const noexcept
{return hasCommands()?std::optional(PooledPinWireSpan{BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.ingress_count,8}):std::nullopt;}
PooledPinWireSpan PooledImmutablePinEdge::pinHeaderSource()const noexcept
{return {BufferId::QFN_POOLED_OWNER_NATIVE,owner_.parameters().native.header,sizeof(p::PinHeader)};}
PooledPinWireSpan PooledImmutablePinEdge::pinHeaderDestination()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.ingress_frontier+size_t{coordinate()}*sizeof(p::PinHeader),sizeof(p::PinHeader)};}
PooledPinWireSpan PooledImmutablePinEdge::frontierSource()const noexcept
{return {BufferId::QFN_POOLED_COORDINATOR_NATIVE,coordinator_.parameters().native.ingress_frontier,PooledImmutableDirectedMessageGeometry::fromPlan(owner_).pinHeaders()};}
PooledPinWireSpan PooledImmutablePinEdge::frontierDestination()const noexcept
{return {BufferId::QFN_POOLED_OWNER_FRAME,owner_.parameters().frame.pin_headers,frontierSource().bytes};}
CapturedTransferChannelMemory PooledImmutablePinEdge::returnChannelMemory()const
{if(!capturedChannelTransport())throw std::invalid_argument("Pooled return edge requires actual admitted native/rank transport; no mapped substitute");return PooledImmutableDirectedChannelGeometry::fromPlans(coordinator_,owner_).reverseMemory();}
uint64_t PooledImmutablePinEdge::messageKey(PooledPinWireRole role)const
{
    if(role!=PooledPinWireRole::Invocation && role!=PooledPinWireRole::Commands && role!=PooledPinWireRole::Frontier && role!=PooledPinWireRole::Replies && role!=PooledPinWireRole::Header)throw std::invalid_argument("Unknown pooled initial-wave wire role");
    return combine(identity_,static_cast<unsigned>(role)+1);
}
PooledNativeOperandSpans PooledImmutablePinEdge::localPinInputs(const PooledImmutableCoordinatorPreparedStorage& storage)const
{
    if(remote() || owner_.parameters().owner.participants()!=1 || coordinate()!=0)
        throw std::invalid_argument("Pooled co-located multi-owner Pin needs its actual directed-offset reader; no whole-bank substitution");
    if(storage.plan().parameters().identity!=coordinator_.parameters().identity)
        throw std::invalid_argument("Local pin source is not its actual original prepared coordinator");
    (void)storage.captureIdentity();const auto roots=storage.roots();
    PooledNativeOperandSpans result{};input(result,PooledNativeOperand::Invocation,roots,invocationSource());
    if(hasCommands()){
        auto commands=commandSource();commands.bytes=maximumCommandBytes();input(result,PooledNativeOperand::Records,roots,commands);
        input(result,PooledNativeOperand::RecordBytes,roots,*incomingCount());
    }
    return result;
}
PooledImmutableCapturedPinEdge::PooledImmutableCapturedPinEdge(PooledImmutablePinEdge d,
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> c,std::shared_ptr<PooledImmutableOwnerPreparedStorage> o,
    std::shared_ptr<PhysicalMemoryAuthority> memory,PooledImmutableTransferChannel channel,PooledImmutableTransferBinding send,PooledImmutableTransferBinding receive,
    PooledImmutableTransferBinding frontierSend,PooledImmutableTransferBinding frontierReceive,
    std::optional<PooledImmutableTransferBinding> commands,std::optional<PooledImmutableTransferBinding> received,
    std::shared_ptr<const ClusterInventory> originalInventory)
    :declaration_(std::move(d)),coordinator_(std::move(c)),owner_(std::move(o)),memory_(std::move(memory)),
     coordinator_identity_(coordinator_->captureIdentity()),owner_identity_(owner_->captureIdentity()),
     original_inventory_(std::move(originalInventory)),
     channel_(std::move(channel)),invocation_send_(std::move(send)),invocation_receive_(std::move(receive)),frontier_send_(std::move(frontierSend)),frontier_receive_(std::move(frontierReceive)),command_send_(std::move(commands)),command_receive_(std::move(received)){}
std::shared_ptr<const PooledImmutableCapturedPinEdge> PooledImmutableCapturedPinEdge::bind(
    std::shared_ptr<PooledImmutableCoordinatorPreparedStorage> c,std::shared_ptr<PooledImmutableOwnerPreparedStorage> o,
    std::shared_ptr<PhysicalMemoryAuthority> memory,void* sendStream,void* receiveStream,
    std::shared_ptr<const ClusterInventory> originalInventory,
    std::shared_ptr<const PooledImmutableNativeCompletionStorage> nativeCompletion)
{
    if(isGraphCaptureActive() || !c || !o || !memory || !sendStream || !receiveStream ||
       reinterpret_cast<uintptr_t>(sendStream)<=2 || reinterpret_cast<uintptr_t>(receiveStream)<=2)
        throw std::invalid_argument("Pooled edge binding requires actual original prepared owners/PMA and exact cold setup streams");
    auto d=PooledImmutablePinEdge::fromPlans(c->plan(),o->plan());
    c->requireAuthority(memory);o->requireAuthority(memory);
    if(memory->worldRank()!=d.sender().world_rank)throw std::invalid_argument("Pooled edge lost its original local PMA rank");
    PooledImmutableTransferChannel channel;
    if(nativeCompletion) {
        // The original native bank authenticates communicator membership and
        // owns its metadata. It never allocates a mapped/peer payload slab.
        channel=PooledImmutableTransferChannel::native(std::move(nativeCompletion),d.sender().device,d.receiver().device,
            PooledImmutableFreeEdge::fromOriginal(d).forwardChannelMaximum());
    } else {
        (void)d.channelMemory();
        auto& transfer=TransferEngine::instance();
        const auto peer=ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(d.sender(),d.receiver(),o->plan().peerAccess());
        if(peer && !originalInventory)throw std::invalid_argument("Peer pooled edge requires its retained original rank inventory");
        channel=peer?transfer.createPeerCapturedTransferChannel(*memory,DeviceId::cpu(),d.sender().device,sendStream,
            d.receiver().device,receiveStream,PooledImmutableFreeEdge::fromOriginal(d).forwardChannelMaximum(),originalInventory->getRank(d.sender().world_rank)):
            transfer.createCapturedTransferChannel(*memory,DeviceId::cpu(),d.sender().device,sendStream,d.receiver().device,
                receiveStream,PooledImmutableFreeEdge::fromOriginal(d).forwardChannelMaximum());
    }
    const auto source=c->roots(),destination=o->roots();const auto invocation=d.invocationSource(),ingress=d.invocationDestination();
    auto send=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(PooledPinWireRole::Invocation),sizeof(n::Invocation)},source.native,invocation.offset);
    auto receive=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(PooledPinWireRole::Invocation),sizeof(n::Invocation)},destination.native,ingress.offset);
    const auto frontier=d.frontierSource(),targetFrontier=d.frontierDestination();
    auto frontierSend=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(PooledPinWireRole::Frontier),frontier.bytes},source.native,frontier.offset);
    auto frontierReceive=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(PooledPinWireRole::Frontier),frontier.bytes},destination.frame,targetFrontier.offset);
    std::optional<PooledImmutableTransferBinding> commands,received;
    if(d.hasCommands()){
        const auto region=d.commandSource(),slice=d.commandSlice(),target=d.commandDestination(),count=*d.incomingCount();
        commands=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(PooledPinWireRole::Commands),d.maximumCommandBytes()},source.native,region.offset).withDeviceSliced(source.native,slice.offset,region.bytes,0,PooledNativeCountedMessage::PinCommands);
        received=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(PooledPinWireRole::Commands),d.maximumCommandBytes()},destination.frame,target.offset).withDeviceCounted(destination.native,count.offset,PooledNativeCountedMessage::PinCommands);
    }
    // Finish both fallible allocations before publishing the physical binding.
    // A later wave failure then unwinds through this SAME RAII edge.
    auto result = std::shared_ptr<PooledImmutableCapturedPinEdge>(new PooledImmutableCapturedPinEdge(std::move(d),std::move(c),std::move(o),std::move(memory),
        std::move(channel),std::move(send),std::move(receive),std::move(frontierSend),std::move(frontierReceive),std::move(commands),std::move(received),std::move(originalInventory)));
    result->invocation_receive_.bindOriginalInvocation(*result->owner_);
    result->owns_invocation_ = true;
    return result;
}
PooledImmutableCapturedPinEdge::~PooledImmutableCapturedPinEdge() noexcept
{
    try { retireOriginalInvocation(); } catch (...) { std::terminate(); }
}
void PooledImmutableCapturedPinEdge::retireOriginalInvocation() const
{
    if (!owns_invocation_) return;
    invocation_receive_.releaseOriginalInvocation(*owner_);
    owns_invocation_ = false;
}
void PooledImmutableCapturedPinEdge::requireCurrent()const
{
    if (!owns_invocation_) throw std::logic_error("Pooled directed invocation binding retired with its original graphs");
    coordinator_->requireAuthority(memory_);owner_->requireAuthority(memory_);coordinator_->plan().requireOwnerPlan(owner_->plan());
    if(coordinator_->captureIdentity()!=coordinator_identity_ || owner_->captureIdentity()!=owner_identity_)
        throw std::logic_error("Pooled directed edge changed its original prepared source/owner roots");
    if(!channel_)throw std::logic_error("Pooled endpoint lost its original transfer owner");
    if(channel_.nativeCollective()) {
        if(channel_.endpointDevice(CapturedTransferEndpoint::Producer)!=declaration_.sender().device || channel_.endpointDevice(CapturedTransferEndpoint::Consumer)!=declaration_.receiver().device)
            throw std::logic_error("Pooled native endpoint changed its original participants");
    } else {
    const auto route = ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(declaration_.sender(),
        declaration_.receiver(), owner_->plan().peerAccess()) ?
        CapturedTransferPayloadStorage::PeerDevice : CapturedTransferPayloadStorage::MappedHost;
    if (!channel_ || channel_.captured()->payloadStorage() != route ||
        channel_.captured()->capacityBytes() != PooledImmutableFreeEdge::fromOriginal(declaration_).forwardChannelMaximum())
        throw std::logic_error("Pooled directed edge changed its original physical payload route/extent");
    }
}
PooledNativeOperandSpans PooledImmutableCapturedPinEdge::ownerInputs(n::OwnerAction action)const
{
    requireCurrent();if(action!=n::OwnerAction::Begin && action!=n::OwnerAction::Pin && action!=n::OwnerAction::AcquirePins)throw std::invalid_argument("Initial-wave ingress cannot supply another owner phase");
    PooledNativeOperandSpans result{};const auto roots=owner_->roots();input(result,PooledNativeOperand::Invocation,roots,declaration_.invocationDestination());
    if(action==n::OwnerAction::Pin && declaration_.hasCommands()){
        input(result,PooledNativeOperand::Records,roots,declaration_.commandDestination());input(result,PooledNativeOperand::RecordBytes,roots,*declaration_.incomingCount());
    }
    if(action==n::OwnerAction::AcquirePins)input(result,PooledNativeOperand::Frontier,roots,declaration_.frontierDestination());
    return result;
}
const PooledImmutableTransferBinding& PooledImmutableCapturedPinEdge::binding(PooledPinWireRole role,CapturedTransferEndpoint endpoint)const
{
    requireCurrent();if(endpoint!=CapturedTransferEndpoint::Producer && endpoint!=CapturedTransferEndpoint::Consumer)throw std::invalid_argument("Unknown pooled transport endpoint");
    if(role==PooledPinWireRole::Invocation)return endpoint==CapturedTransferEndpoint::Producer?invocation_send_:invocation_receive_;
    if(role==PooledPinWireRole::Frontier)return endpoint==CapturedTransferEndpoint::Producer?frontier_send_:frontier_receive_;
    if(role!=PooledPinWireRole::Commands || !declaration_.hasCommands())throw std::invalid_argument("C0/unknown pooled payload has no transfer binding");
    return endpoint==CapturedTransferEndpoint::Producer?*command_send_:*command_receive_;
}
uint64_t PooledImmutableCapturedPinEdge::captureIdentity()const
{
    requireCurrent();auto identity=combine(combine(declaration_.identity(),coordinator_identity_),owner_identity_);
    identity=combine(combine(identity,invocation_send_.captureIdentity()),invocation_receive_.captureIdentity());
    identity=combine(combine(identity,frontier_send_.captureIdentity()),frontier_receive_.captureIdentity());
    if(command_send_)identity=combine(combine(identity,command_send_->captureIdentity()),command_receive_->captureIdentity());return identity;
}
PooledImmutableCapturedPinReturn::PooledImmutableCapturedPinReturn(std::shared_ptr<const PooledImmutableCapturedPinEdge> forward,
    std::shared_ptr<PhysicalMemoryAuthority> memory,PooledImmutableTransferChannel channel,PooledImmutableTransferBinding send,PooledImmutableTransferBinding receive,
    std::optional<PooledImmutableTransferBinding> replies,std::optional<PooledImmutableTransferBinding> received)
    :forward_(std::move(forward)),memory_(std::move(memory)),channel_(std::move(channel)),header_send_(std::move(send)),header_receive_(std::move(receive)),
     reply_send_(std::move(replies)),reply_receive_(std::move(received)){}
std::shared_ptr<const PooledImmutableCapturedPinReturn> PooledImmutableCapturedPinReturn::bind(
    std::shared_ptr<const PooledImmutableCapturedPinEdge> forward,std::shared_ptr<PhysicalMemoryAuthority> memory,void* ownerStream,void* coordinatorStream)
{
    if(isGraphCaptureActive() || !forward || !memory || !ownerStream || !coordinatorStream ||
       reinterpret_cast<uintptr_t>(ownerStream)<=2 || reinterpret_cast<uintptr_t>(coordinatorStream)<=2)
        throw std::invalid_argument("Pooled Pin return needs original cold owners/PMA/exact setup streams");
    forward->requireCurrent();const auto& d=forward->declaration();
    forward->ownerStorage()->requireAuthority(memory);forward->coordinatorStorage()->requireAuthority(memory);
    if(memory->worldRank()!=d.sender().world_rank)throw std::invalid_argument("Pooled Pin return lost original source PMA rank");
    auto& transfer=TransferEngine::instance();const auto source=forward->ownerStorage()->roots(),destination=forward->coordinatorStorage()->roots();
    PooledImmutableTransferChannel channel;
    if(forward->channel_.nativeCollective()) {
        channel=PooledImmutableTransferChannel::native(forward->channel_.nativeStorage(),d.receiver().device,d.sender().device,
            PooledImmutableFreeEdge::fromOriginal(d).reverseChannelMaximum());
    } else {
        (void)d.returnChannelMemory();
        const auto peer=ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(d.sender(),d.receiver(),forward->ownerStorage()->plan().peerAccess());
        if(peer && !forward->original_inventory_)throw std::invalid_argument("Peer pooled return requires the same retained original inventory");
        channel=peer?transfer.createPeerCapturedTransferChannel(*memory,DeviceId::cpu(),d.receiver().device,ownerStream,
            d.sender().device,coordinatorStream,PooledImmutableFreeEdge::fromOriginal(d).reverseChannelMaximum(),forward->original_inventory_->getRank(d.sender().world_rank)):
            transfer.createCapturedTransferChannel(*memory,DeviceId::cpu(),d.receiver().device,ownerStream,d.sender().device,
                coordinatorStream,PooledImmutableFreeEdge::fromOriginal(d).reverseChannelMaximum());
    }
    const auto header=d.pinHeaderSource(),target=d.pinHeaderDestination();
    auto send=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(PooledPinWireRole::Header),header.bytes},source.native,header.offset);
    auto receive=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(PooledPinWireRole::Header),target.bytes},destination.native,target.offset);
    std::optional<PooledImmutableTransferBinding> replies,received;
    if(d.hasCommands()){
        const auto original=d.replySource(),count=d.replyCountSource(),incoming=d.replyDestination(),bytes=*d.replyCountDestination();
        replies=(channel).bind(CapturedTransferEndpoint::Producer,{d.messageKey(PooledPinWireRole::Replies),original.bytes},source.native,original.offset).withDeviceCounted(source.native,count.offset,PooledNativeCountedMessage::PinGrants);
        received=(channel).bind(CapturedTransferEndpoint::Consumer,{d.messageKey(PooledPinWireRole::Replies),original.bytes},destination.native,incoming.offset).withDeviceCounted(destination.native,bytes.offset,PooledNativeCountedMessage::PinGrants);
    }
    return std::shared_ptr<const PooledImmutableCapturedPinReturn>(new PooledImmutableCapturedPinReturn(std::move(forward),std::move(memory),
        std::move(channel),std::move(send),std::move(receive),std::move(replies),std::move(received)));
}
void PooledImmutableCapturedPinReturn::requireCurrent()const
{
    forward_->requireCurrent();
    forward_->coordinatorStorage()->requireAuthority(memory_);
    forward_->ownerStorage()->requireAuthority(memory_);
    if(!channel_)throw std::logic_error("Pooled endpoint lost its original transfer owner");
    if(channel_.nativeCollective()) {
        if(channel_.endpointDevice(CapturedTransferEndpoint::Producer)!=forward_->declaration().receiver().device || channel_.endpointDevice(CapturedTransferEndpoint::Consumer)!=forward_->declaration().sender().device)
            throw std::logic_error("Pooled native endpoint changed its original participants");
    } else {
    const auto route = ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(forward_->declaration().sender(),
        forward_->declaration().receiver(), forward_->ownerStorage()->plan().peerAccess()) ?
        CapturedTransferPayloadStorage::PeerDevice : CapturedTransferPayloadStorage::MappedHost;
    if (!channel_ || channel_.captured()->payloadStorage() != route ||
        channel_.captured()->capacityBytes() != PooledImmutableFreeEdge::fromOriginal(forward_->declaration()).reverseChannelMaximum())
        throw std::logic_error("Pooled directed return changed its original physical payload route/extent");
    }
}
const PooledImmutableTransferBinding& PooledImmutableCapturedPinReturn::binding(PooledPinWireRole role,CapturedTransferEndpoint endpoint)const
{
    requireCurrent();if(endpoint!=CapturedTransferEndpoint::Producer && endpoint!=CapturedTransferEndpoint::Consumer)throw std::invalid_argument("Unknown Pin return endpoint");
    if(role==PooledPinWireRole::Header)return endpoint==CapturedTransferEndpoint::Producer?header_send_:header_receive_;
    if(role!=PooledPinWireRole::Replies || !reply_send_)throw std::invalid_argument("C0/unknown Pin return has no Grant payload binding");
    return endpoint==CapturedTransferEndpoint::Producer?*reply_send_:*reply_receive_;
}
PooledNativeOperandSpans PooledImmutableCapturedPinReturn::coordinatorInputs(n::CoordinatorAction action)const
{
    requireCurrent();const auto& d=forward_->declaration();const auto roots=forward_->coordinatorStorage()->roots();PooledNativeOperandSpans result{};
    if(action==n::CoordinatorAction::SealPins)input(result,PooledNativeOperand::Frontier,roots,d.frontierSource());
    else if(action==n::CoordinatorAction::AcquirePins && d.hasCommands()){
        input(result,PooledNativeOperand::Records,roots,d.replyDestination());input(result,PooledNativeOperand::RecordBytes,roots,*d.replyCountDestination());
    }else throw std::invalid_argument("Pin return supplies only actual positive-capacity acquire or complete frontier operands");
    return result;
}
uint64_t PooledImmutableCapturedPinReturn::captureIdentity()const
{
    requireCurrent();auto result=combine(combine(forward_->captureIdentity(),header_send_.captureIdentity()),header_receive_.captureIdentity());
    if(reply_send_)result=combine(combine(result,reply_send_->captureIdentity()),reply_receive_->captureIdentity());return result;
}
}
namespace llaminar2
{
namespace
{
namespace q=qwen4exp;namespace n=q::pooled_native;
/** @brief One explicit local endpoint of a cold-bound original protocol edge. */
class PooledPinTransferStage final:public IComputeStage
{
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,endpoint_);
        return {q::PooledImmutableDirectedWireFamily::Pin,static_cast<std::uint32_t>(role_),endpoint_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice(),binding.nativeCollective()?q::PooledImmutableDirectedTransportKind::NativeCollective:q::PooledImmutableDirectedTransportKind::CapturedChannel};
    }

    /** @brief Retain the complete original edge before computing an immutable capture identity. */
    PooledPinTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedPinEdge> edge,
        q::PooledPinWireRole role,CapturedTransferEndpoint endpoint)
        :IComputeStage(endpoint==CapturedTransferEndpoint::Producer?edge->declaration().sender().device:edge->declaration().receiver().device),
         arena_(&arena),edge_(std::move(edge)),role_(role),endpoint_(endpoint)
    {requireDeclarations();identity_=graphCaptureVariantSignature();native_collective_=edge_->binding(role_,endpoint_).nativeCollective();}
    /** @return Exact actual producer ownership for declaration-time dependency authentication. */
    bool matches(const q::PooledImmutableCapturedPinEdge* edge,q::PooledPinWireRole role,CapturedTransferEndpoint endpoint)const noexcept
    {return edge_.get()==edge && role_==role && endpoint_==endpoint;}
    /** @brief Join only the original setup producer on this endpoint's exact capture stream. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override
    {
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)
            throw std::invalid_argument("Pooled transfer preparation requires its exact participant/stream");
        requireDeclarations();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled transfer identity changed");
        if(endpoint_==CapturedTransferEndpoint::Producer)edge_->coordinatorStorage()->acquireSetup(stream);
        else edge_->ownerStorage()->acquireSetup(stream);
        edge_->binding(role_,endpoint_).prepare(stream);setGPUStream(stream);return true;
    }
    /** @brief Enqueue one actual public transfer operation; it owns exact event-aware input/output publication. */
    bool execute(IDeviceContext* context)override
    {
        if(!context || context->deviceId()!=device())throw std::invalid_argument("Pooled transfer has a foreign execution participant");
        requireDeclarations();if(graphCaptureVariantSignature()!=identity_)throw std::logic_error("Pooled transfer source identity drift");
        edge_->binding(role_,endpoint_).enqueue(requireGPUStream());return true;
    }
    /** @return Explicit collective classification, without a topology support advertisement. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return native_collective_?CollectiveGraphCaptureAuthority::RuntimeBackend:CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Nominal component name. */
    std::string name()const override{return "PooledImmutableDirectedPin";}
    /** @return Exact original binding and endpoint identity, never live slice contents. */
    uint64_t graphCaptureVariantSignature()const override
    {
        requireDeclarations();return q::pooledPinEndpointCaptureIdentity(edge_->captureIdentity(),role_,endpoint_);
    }
    /** @return Setup event joins occur once before retained capture, never during replay. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Device-owned empty/shrinking live extents preserve the original physical row contract. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return This actual endpoint's native backend only. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return All packet/control roots belong to the original source-owned admission. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return Explicit local producer/consumer roles for graph dependency/publication validation. */
    StageBufferContract bufferContract()const override
    {
        requireDeclarations();auto result=StageBufferContract::build();
        if(endpoint_==CapturedTransferEndpoint::Producer)result.addInput(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");
        else{
            result.addPreallocatedInOut(BufferId::QFN_POOLED_OWNER_NATIVE,"INT8");
            if((role_==q::PooledPinWireRole::Commands || role_==q::PooledPinWireRole::Frontier))result.addPreallocatedInOut(BufferId::QFN_POOLED_OWNER_FRAME,"INT8");
        }
        return result;
    }
    /** @return Metadata only; no native count/offset is read for diagnostics. */
    StageDumpInfo buildDumpInfoImpl()const override
    {StageDumpInfo result;result.addScalarInt("pooled_pin_wire_role",static_cast<unsigned>(role_));result.addScalarInt("pooled_pin_endpoint",static_cast<unsigned>(endpoint_));return result;}
private:
    /** @brief Authenticate original local arena roots before any transfer/graph mutation. */
    void requireDeclarations()const
    {
        if(!arena_ || !edge_)throw std::invalid_argument("Pooled transfer lost its original participant declaration");
        edge_->requireCurrent();(void)edge_->binding(role_,endpoint_);
        if(endpoint_==CapturedTransferEndpoint::Producer){
            if(arena_->getTensor(BufferId::QFN_POOLED_COORDINATOR_NATIVE)!=edge_->coordinatorStorage()->roots().native.get())
                throw std::logic_error("Pooled sender arena does not retain its original coordinator root");
        }else{
            const auto roots=edge_->ownerStorage()->roots();
            if(arena_->getTensor(BufferId::QFN_POOLED_OWNER_NATIVE)!=roots.native.get() ||
               ((role_==q::PooledPinWireRole::Commands || role_==q::PooledPinWireRole::Frontier) && arena_->getTensor(BufferId::QFN_POOLED_OWNER_FRAME)!=roots.frame.get()))
                throw std::logic_error("Pooled receiver arena does not retain its original native/frame roots");
        }
    }
    BufferArena* arena_;
    std::shared_ptr<const q::PooledImmutableCapturedPinEdge> edge_;
    q::PooledPinWireRole role_;
    CapturedTransferEndpoint endpoint_;
    bool native_collective_=false; // Cold-selected original transport; capture policy never probes runtime state.
    uint64_t identity_=0;
};
/** @brief One actual reverse endpoint; every operation remains on this participant's original exact stream. */
class PooledPinReturnTransferStage final:public IComputeStage
{
public:
    /** @return Exact original semantic endpoint and native dispatch inputs before capture. */
    q::PooledImmutableDirectedTransferOperation directedTransferOperation()const{
        requireDeclarations();const auto& binding=edge_->binding(role_,endpoint_);
        return {q::PooledImmutableDirectedWireFamily::PinReturn,static_cast<std::uint32_t>(role_),endpoint_,
            binding.message().bytes,binding.extentSource(),binding.hasSourceSlice(),binding.nativeCollective()?q::PooledImmutableDirectedTransportKind::NativeCollective:q::PooledImmutableDirectedTransportKind::CapturedChannel};
    }

    /** @brief Retain the actual original owner/coordinator and one nominal return message. */
    PooledPinReturnTransferStage(BufferArena& arena,std::shared_ptr<const q::PooledImmutableCapturedPinReturn> edge,
        q::PooledPinWireRole role,CapturedTransferEndpoint endpoint)
        :IComputeStage(endpoint==CapturedTransferEndpoint::Producer?edge->forward()->declaration().receiver().device:edge->forward()->declaration().sender().device),
         arena_(&arena),edge_(std::move(edge)),role_(role),endpoint_(endpoint){requireDeclarations();identity_=graphCaptureVariantSignature();native_collective_=edge_->binding(role_,endpoint_).nativeCollective();}
    /** @return Exact retained original return channel/action/endpoint for following-wave ownership authentication. */
    bool matches(const q::PooledImmutableCapturedPinReturn* edge,q::PooledPinWireRole role,CapturedTransferEndpoint endpoint)const noexcept
    {return edge_.get()==edge && role_==role && endpoint_==endpoint;}
    /** @brief Acquire only the actual endpoint setup event before retained capture. */
    bool prepareGraphLaunch(IDeviceContext* context,void* stream)override{
        if(!context || context->deviceId()!=device() || !stream || reinterpret_cast<uintptr_t>(stream)<=2)throw std::invalid_argument("Pin return needs its original participant/exact stream");
        requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Pin return capture drift");
        if(endpoint_==CapturedTransferEndpoint::Producer)edge_->forward()->ownerStorage()->acquireSetup(stream);else edge_->forward()->coordinatorStorage()->acquireSetup(stream);
        edge_->binding(role_,endpoint_).prepare(stream);setGPUStream(stream);return true;}
    /** @brief Submit the actual event/publication-aware transfer; no count is read on the host. */
    bool execute(IDeviceContext* context)override{
        if(!context || context->deviceId()!=device())throw std::invalid_argument("Pin return has a foreign participant");
        requireDeclarations();if(identity_!=graphCaptureVariantSignature())throw std::logic_error("Pin return binding changed");
        edge_->binding(role_,endpoint_).enqueue(requireGPUStream());return true;}
    /** @return Explicit graph collective identity. */
    ComputeStageType type()const override{return ComputeStageType::QWEN4EXP_POOLED_DIRECTED_TRANSFER;}
    /** @return This original bound endpoint executes native device-counted transfer kernels, independently from collective-runtime capture. */
    CollectiveGraphCaptureAuthority collectiveGraphCaptureAuthority()const noexcept override{return native_collective_?CollectiveGraphCaptureAuthority::RuntimeBackend:CollectiveGraphCaptureAuthority::ParticipantLocalCapturedTransfer;}
    /** @return Nominal native metadata endpoint. */
    std::string name()const override{return "PooledImmutablePinReturn";}
    /** @return Complete retained owner/endpoint/role identity, excluding live device counts. */
    uint64_t graphCaptureVariantSignature()const override{requireDeclarations();return q::pooledPinEndpointCaptureIdentity(edge_->captureIdentity(),role_,endpoint_);}
    /** @return Original setup joins occur before capture only. */
    GraphLaunchPreparationPolicy graphLaunchPreparationPolicy()const override{return GraphLaunchPreparationPolicy::CaptureOnly;}
    /** @return Empty/shrinking requests preserve exact device-owned byte publication. */
    bool supportsPaddedPrefillRealLengthContract()const override{return true;}
    /** @return Only this participant's actual native backend. */
    bool supportsBackend(ComputeBackendType backend)const override{return device().is_cuda()?backend==ComputeBackendType::GPU_CUDA:backend==ComputeBackendType::GPU_ROCM;}
    /** @return All roots belong to the existing source-owned geometry/PMA. */
    StageBufferRequirements getBufferRequirements()const override{return {};}
    /** @return A read-only owner producer or original coordinator InOut receive bank. */
    StageBufferContract bufferContract()const override{requireDeclarations();auto c=StageBufferContract::build();if(endpoint_==CapturedTransferEndpoint::Producer)c.addInput(BufferId::QFN_POOLED_OWNER_NATIVE,"INT8");else c.addPreallocatedInOut(BufferId::QFN_POOLED_COORDINATOR_NATIVE,"INT8");return c;}
    /** @return Immutable nominal metadata only, never a live native observation. */
    StageDumpInfo buildDumpInfoImpl()const override{StageDumpInfo d;d.addScalarInt("pin_return_role",static_cast<unsigned>(role_));return d;}
private:
    /** @brief Authenticate the exact original local arena role before every launch/capture identity. */
    void requireDeclarations()const{
        if(!arena_ || !edge_)throw std::logic_error("Pin return lost its original edge");edge_->requireCurrent();(void)edge_->binding(role_,endpoint_);
        const bool owner=endpoint_==CapturedTransferEndpoint::Producer;const auto root=owner?edge_->forward()->ownerStorage()->roots().native:edge_->forward()->coordinatorStorage()->roots().native;
        if(arena_->getTensor(owner?BufferId::QFN_POOLED_OWNER_NATIVE:BufferId::QFN_POOLED_COORDINATOR_NATIVE)!=root.get())throw std::logic_error("Pin return arena replaced its actual original authority");}
    BufferArena* arena_;
    std::shared_ptr<const q::PooledImmutableCapturedPinReturn> edge_;
    q::PooledPinWireRole role_;
    CapturedTransferEndpoint endpoint_;
    bool native_collective_=false; // Cold-selected original transport; capture policy never probes runtime state.
    uint64_t identity_=0;
};
/** @brief Reject missing/foreign original native producer nodes before adding any endpoint. */
void requireProducer(const ComputeGraph& graph,const std::string& name,n::CoordinatorAction action,
    const q::PooledImmutableCoordinatorPreparedStorage* storage)
{
    const auto* node=graph.getNode(name);
    if(!node || !storage || node->device!=storage->plan().allocator().device)
        throw std::invalid_argument("Pooled transfer producer changed original coordinator device");
    if(action==n::CoordinatorAction::Begin){
        if(const auto* compact=dynamic_cast<const PooledImmutableLocalHotBeginStage*>(node->stage.get())){
            if(compact->source()->coordinator().get()!=storage)
                throw std::invalid_argument("Compact transfer producer substituted original coordinator");
            requirePooledCompactQueryProducer(graph,compact->query(),name);return;
        }
    }
    const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get());
    const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;
    if(!actual || actual->action!=action || actual->storage.get()!=storage)
        throw std::invalid_argument("Pooled transfer producer is not its actual original coordinator phase");
}
/** @brief Validate the complete pending node namespace before graph mutation. */
void requireNames(const ComputeGraph& graph,std::initializer_list<std::string> names)
{for(const auto& name:names)if(name.empty() || graph.getNode(name))throw std::invalid_argument("Pooled participant edge would replace an existing graph node");}
}
void requirePooledPinOwnerCompletion(const ComputeGraph& graph,
    const std::shared_ptr<const q::PooledImmutableCapturedPinReturn>& original,const std::string& acquiredPins)
{
    if(isGraphCaptureActive() || !original || acquiredPins.empty())throw std::invalid_argument("Owner Pin completion requires original cold return/channel owner");
    original->requireCurrent();const auto forward=original->forward();const auto storage=forward->ownerStorage();const auto device=storage->plan().allocator().device;
    const auto predecessor=[&](const ComputeNode* node){
        if(!node || node->device!=device || node->dependencies.size()!=1 || !graph.getNode(node->dependencies.front()))
            throw std::invalid_argument("Owner Pin retirement lacks its exact completed predecessor");
        return node->dependencies.front();
    };
    const auto phase=[&](const std::string& name,n::OwnerAction action){
        const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;
        const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;
        if(!actual || node->device!=device || actual->action!=action || actual->storage!=storage)
            throw std::invalid_argument("Owner Pin retirement substituted another original phase/storage");
        const auto expected=forward->ownerInputs(action==n::OwnerAction::SealPins?n::OwnerAction::Begin:action);const auto& inputs=actual->operands.spans();
        for(size_t i=0;i<expected.size();++i)if(bool(expected[i])!=bool(inputs[i]) || (expected[i] &&
            (expected[i]->root!=inputs[i]->root || expected[i]->buffer!=inputs[i]->buffer ||
             expected[i]->offset!=inputs[i]->offset || expected[i]->bytes!=inputs[i]->bytes)))
            throw std::invalid_argument("Owner Pin retirement changed original ingress/frontier operands");
        return node;
    };
    const auto forwardEndpoint=[&](const std::string& name,q::PooledPinWireRole role){
        const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledPinTransferStage*>(node->stage.get()):nullptr;
        if(!stage || node->device!=device || !stage->matches(forward.get(),role,CapturedTransferEndpoint::Consumer))
            throw std::invalid_argument("Owner Pin retirement replaced its actual original forward receiver");
        return node;
    };
    const auto returnEndpoint=[&](const std::string& name,q::PooledPinWireRole role){
        const auto* node=graph.getNode(name);const auto* stage=node?dynamic_cast<const PooledPinReturnTransferStage*>(node->stage.get()):nullptr;
        if(!stage || node->device!=device || !stage->matches(original.get(),role,CapturedTransferEndpoint::Producer))
            throw std::invalid_argument("Owner Pin retirement replaced its actual original return sender");
        return node;
    };
    auto current=predecessor(phase(acquiredPins,n::OwnerAction::AcquirePins));
    current=predecessor(forwardEndpoint(current,q::PooledPinWireRole::Frontier));
    current=predecessor(returnEndpoint(current,q::PooledPinWireRole::Header));
    current=predecessor(phase(current,n::OwnerAction::SealPins));
    if(forward->declaration().hasCommands())current=predecessor(returnEndpoint(current,q::PooledPinWireRole::Replies));
    current=predecessor(phase(current,n::OwnerAction::Pin));
    if(forward->declaration().hasCommands())current=predecessor(forwardEndpoint(current,q::PooledPinWireRole::Commands));
    current=predecessor(phase(current,n::OwnerAction::Begin));
    // Initial invocation may itself depend on the original outer controller;
    // only this component's actual receive identity is authenticated here.
    (void)forwardEndpoint(current,q::PooledPinWireRole::Invocation);
}
void requirePooledPinFrontierCompletion(const ComputeGraph& graph,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> owners,const std::string& completion)
{
    if(owners.empty() || !owners.front())throw std::invalid_argument("Missing original Pin frontier cohort");
    const auto coordinator=owners.front()->forward()->coordinatorStorage();
    if(owners.size()!=coordinator->plan().parameters().policy.participants())throw std::invalid_argument("Incomplete original Pin frontier cohort");
    std::string current=completion;
    for(size_t i=owners.size();i-->0;){
        if(!owners[i] || owners[i]->forward()->coordinatorStorage()!=coordinator || owners[i]->forward()->declaration().coordinate()!=i)
            throw std::invalid_argument("Pin frontier completion changed original source/owner order");
        owners[i]->requireCurrent();const auto* node=graph.getNode(current);
        const auto* stage=node?dynamic_cast<const PooledPinTransferStage*>(node->stage.get()):nullptr;
        if(!stage || !stage->matches(owners[i]->forward().get(),q::PooledPinWireRole::Frontier,CapturedTransferEndpoint::Producer) || node->dependencies.size()!=1)
            throw std::invalid_argument("Free wave requires actual complete original Pin frontier sends");
        current=node->dependencies.front();
    }
    requireProducer(graph,current,n::CoordinatorAction::SealPins,coordinator.get());
    const auto* sealed=dynamic_cast<const PooledImmutableNativePhaseStage*>(graph.getNode(current)->stage.get());
    const auto& actual=std::get<PooledImmutableNativePhaseStage::Coordinator>(sealed->phase()).operands.spans();
    const auto expected=owners.front()->coordinatorInputs(n::CoordinatorAction::SealPins);
    for(size_t i=0;i<expected.size();++i)if(bool(expected[i])!=bool(actual[i]) || (expected[i] &&
        (expected[i]->root!=actual[i]->root || expected[i]->buffer!=actual[i]->buffer ||
         expected[i]->offset!=actual[i]->offset || expected[i]->bytes!=actual[i]->bytes)))
        throw std::invalid_argument("Pin frontier predecessor changed its actual original complete owner frontier");
    const auto* sealNode=graph.getNode(current);
    if(sealNode->dependencies.size()!=1)throw std::invalid_argument("Pin frontier lacks its actual complete return receipt chain");
    current=sealNode->dependencies.front();
    for(size_t i=owners.size();i-->0;){
        const auto* header=graph.getNode(current);const auto* received=header?dynamic_cast<const PooledPinReturnTransferStage*>(header->stage.get()):nullptr;
        if(!received || !received->matches(owners[i].get(),q::PooledPinWireRole::Header,CapturedTransferEndpoint::Consumer))
            throw std::invalid_argument("Free wave replaced the actual preceding Pin return channel/owner");
        const auto* readFrontier=header;
        if(owners[i]->forward()->declaration().hasCommands()){
            if(header->dependencies.size()!=1)throw std::invalid_argument("Pin header lacks its actual Grant acquisition");
            const auto* acquire=graph.getNode(header->dependencies.front());const auto* stage=acquire?dynamic_cast<const PooledImmutableNativePhaseStage*>(acquire->stage.get()):nullptr;
            const auto* phase=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;
            if(!phase || phase->action!=n::CoordinatorAction::AcquirePins || phase->storage!=coordinator || acquire->dependencies.size()!=1)
                throw std::invalid_argument("Free wave lacks actual preceding Pin Grant consumer");
            const auto expectedInputs=owners[i]->coordinatorInputs(n::CoordinatorAction::AcquirePins);const auto& actualInputs=phase->operands.spans();
            for(size_t r=0;r<expectedInputs.size();++r)if(bool(expectedInputs[r])!=bool(actualInputs[r]) || (expectedInputs[r] &&
                (expectedInputs[r]->root!=actualInputs[r]->root || expectedInputs[r]->buffer!=actualInputs[r]->buffer ||
                 expectedInputs[r]->offset!=actualInputs[r]->offset || expectedInputs[r]->bytes!=actualInputs[r]->bytes)))
                throw std::invalid_argument("Free wave Pin Grant consumer changed original receive/count roles");
            readFrontier=graph.getNode(acquire->dependencies.front());const auto* grant=readFrontier?dynamic_cast<const PooledPinReturnTransferStage*>(readFrontier->stage.get()):nullptr;
            if(!grant || !grant->matches(owners[i].get(),q::PooledPinWireRole::Replies,CapturedTransferEndpoint::Consumer))
                throw std::invalid_argument("Free wave replaced its actual preceding Pin Grant channel");
        }
        // Re-authenticate the installed complete command-read frontier and
        // previous serial receiver, not merely an equal graph node name. Thus
        // a second unused reverse channel cannot be substituted for the one
        // whose actual Pin messages retired the shared banks.
        unsigned producer=0;std::vector<unsigned> sent(owners.size(),0);std::string prior;
        for(const auto& dependency:readFrontier->dependencies){const auto* node=graph.getNode(dependency);bool matched=false;
            if(node){const auto* native=dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get());const auto* phase=native?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&native->phase()):nullptr;
                if(phase && phase->action==n::CoordinatorAction::PinCommands && phase->storage==coordinator){++producer;matched=true;}
                const auto* transfer=dynamic_cast<const PooledPinTransferStage*>(node->stage.get());
                for(size_t j=0;j<owners.size();++j)if(transfer && transfer->matches(owners[j]->forward().get(),owners[j]->forward()->declaration().hasCommands()?q::PooledPinWireRole::Commands:q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Producer)){++sent[j];matched=true;}
                const auto* returned=dynamic_cast<const PooledPinReturnTransferStage*>(node->stage.get());
                if(i && returned && returned->matches(owners[i-1].get(),q::PooledPinWireRole::Header,CapturedTransferEndpoint::Consumer)){if(!prior.empty())throw std::invalid_argument("Duplicate preceding Pin return frontier");prior=dependency;matched=true;}
            }
            if(!matched)throw std::invalid_argument("Pin return command/read frontier has a foreign dependency");
        }
        if(producer!=1 || std::any_of(sent.begin(),sent.end(),[](unsigned count){return count!=1;}) || (i && prior.empty()))
            throw std::invalid_argument("Pin return did not retire every original command reader before shared-bank reuse");
        current=prior;
    }
}
PooledPinParticipantNodes appendPooledPinSender(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedPinEdge> edge,const std::string& beginProducer,
    const std::string& commandProducer,const std::string& prefix)
{
    if(isGraphCaptureActive() || !edge || prefix.empty())throw std::invalid_argument("Pooled sender wiring must precede capture");
    edge->requireCurrent();requireProducer(graph,beginProducer,n::CoordinatorAction::Begin,edge->coordinatorStorage().get());
    requireProducer(graph,commandProducer,n::CoordinatorAction::PinCommands,edge->coordinatorStorage().get());
    const auto* commandsNode=graph.getNode(commandProducer);
    if(std::find(commandsNode->dependencies.begin(),commandsNode->dependencies.end(),beginProducer)==commandsNode->dependencies.end())
        throw std::invalid_argument("Pooled sender requires the original direct Begin-to-PinCommands phase dependency");
    const auto invocation=prefix+"/invocation_send",commands=prefix+"/pin_send";requireNames(graph,{invocation,commands});
    auto first=std::make_unique<PooledPinTransferStage>(arena,edge,q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Producer);
    auto last=edge->declaration().hasCommands()?std::make_unique<PooledPinTransferStage>(arena,edge,q::PooledPinWireRole::Commands,CapturedTransferEndpoint::Producer):nullptr;
    graph.addNode(invocation,std::move(first),edge->declaration().sender().device);graph.addDependency(invocation,beginProducer);
    // Whole-root contracts serialize the immutable Invocation read before the
    // next writer, although PinCommands does not change those48 control bytes.
    graph.addDependency(commandProducer,invocation);
    if(last){graph.addNode(commands,std::move(last),edge->declaration().sender().device);graph.addDependency(commands,commandProducer);return {invocation,commands};}
    return {invocation,invocation};
}
PooledPinParticipantNodes appendPooledPinReceiver(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedPinEdge> edge,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,const std::string& prefix)
{
    if(isGraphCaptureActive() || !edge || !code || prefix.empty())throw std::invalid_argument("Pooled receiver wiring needs original storage/code before capture");
    edge->requireCurrent();const auto device=edge->declaration().receiver().device;
    if(code->backend()!=device.type)throw std::invalid_argument("Pooled receiver kernel bundle belongs to another native backend");
    const auto invocation=prefix+"/invocation_receive",begin=prefix+"/owner_begin",commands=prefix+"/pin_receive",pin=prefix+"/owner_pin";
    requireNames(graph,{invocation,begin,commands,pin});
    auto incoming=std::make_unique<PooledPinTransferStage>(arena,edge,q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Consumer);
    const auto& storage=edge->ownerStorage();
    auto start=std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::Begin,storage,
        q::PooledImmutableOwnerOperands(n::OwnerAction::Begin,storage->plan(),edge->ownerInputs(n::OwnerAction::Begin))});
    auto records=edge->declaration().hasCommands()?std::make_unique<PooledPinTransferStage>(arena,edge,q::PooledPinWireRole::Commands,CapturedTransferEndpoint::Consumer):nullptr;
    auto consume=std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::Pin,storage,
        q::PooledImmutableOwnerOperands(n::OwnerAction::Pin,storage->plan(),edge->ownerInputs(n::OwnerAction::Pin))});
    graph.addNode(invocation,std::move(incoming),device);graph.addNode(begin,std::move(start),device);graph.addDependency(begin,invocation);
    if(records){graph.addNode(commands,std::move(records),device);graph.addDependency(commands,begin);}
    graph.addNode(pin,std::move(consume),device);graph.addDependency(pin,edge->declaration().hasCommands()?commands:begin);return {invocation,pin};
}
void wirePooledPinReplyBankLifetime(ComputeGraph& graph,const q::PooledImmutableCoordinatorNativePlan& original,
    std::span<const PooledPinReplyOwnerNodes> owners,const std::string& commandsProduced)
{
    if(isGraphCaptureActive() || owners.size()!=original.parameters().policy.participants() || !graph.getNode(commandsProduced))
        throw std::invalid_argument("Pin reply lifetime requires original complete command producer/cohort before capture");
    std::set<std::string> names{commandsProduced};
    for(size_t i=0;i<owners.size();++i){const auto& o=owners[i];const bool positive=original.owners()[i].slots!=0;
        const bool direct=o.path==PooledPinReplyOwnerNodes::Path::DirectLocal;
        if(o.coordinate!=i || (direct?(!o.replyReceived.empty() || positive!=!o.replyAcquired.empty()):
            positive!=(!o.replyReceived.empty() && !o.replyAcquired.empty())) ||
           (!positive && (!o.replyReceived.empty() || !o.replyAcquired.empty())))throw std::invalid_argument("Pin reply lifetime changed original owner/C0/local order");
        for(const auto& name:{o.commandSent,o.replyReceived,o.replyAcquired,o.headerReceived}){
            if(name.empty()){if((positive && !(direct && name==o.replyReceived)) || name==o.commandSent || name==o.headerReceived)throw std::invalid_argument("Missing Pin reply phase node");continue;}
            if(!graph.getNode(name) || !names.insert(name).second)throw std::invalid_argument("Pin reply lifetime contains missing/aliased phase nodes");
        }
    }
    std::string prior;
    for(const auto& o:owners){const auto& first=o.path==PooledPinReplyOwnerNodes::Path::DirectLocal && !o.replyAcquired.empty()?o.replyAcquired:
            o.replyReceived.empty()?o.headerReceived:o.replyReceived;
        // This complete read frontier is essential even if another owner's
        // reply is already ready. pin_commands is one shared source/receive
        // phase union, not independently writable per-owner scratch capacity.
        graph.addDependency(first,commandsProduced);
        for(const auto& other:owners)graph.addDependency(first,other.commandSent);
        if(!prior.empty())graph.addDependency(first,prior);
        if(!o.replyReceived.empty()){graph.addDependency(o.replyAcquired,o.replyReceived);graph.addDependency(o.headerReceived,o.replyAcquired);}
        else if(!o.replyAcquired.empty())graph.addDependency(o.headerReceived,o.replyAcquired);
        prior=o.headerReceived;
    }
}
PooledPinParticipantNodes appendPooledPinOwnerReturn(ComputeGraph& graph,BufferArena& arena,
    std::shared_ptr<const q::PooledImmutableCapturedPinReturn> edge,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    const std::string& originalPin,const std::string& prefix)
{
    if(isGraphCaptureActive() || !edge || !code || prefix.empty())throw std::invalid_argument("Pin owner return must retain actual cold owners/code");
    edge->requireCurrent();const auto forward=edge->forward();const auto storage=forward->ownerStorage();const auto device=storage->plan().allocator().device;
    const auto* node=graph.getNode(originalPin);const auto* stage=node?dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get()):nullptr;
    const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;
    if(!actual || actual->action!=n::OwnerAction::Pin || actual->storage!=storage || code->backend()!=device.type)
        throw std::invalid_argument("Pin return producer is not the actual original owner Pin");
    const auto expected=forward->ownerInputs(n::OwnerAction::Pin);const auto& provided=actual->operands.spans();
    for(size_t i=0;i<expected.size();++i)if(bool(expected[i])!=bool(provided[i]) || (expected[i] &&
        (expected[i]->root!=provided[i]->root || expected[i]->buffer!=provided[i]->buffer || expected[i]->offset!=provided[i]->offset || expected[i]->bytes!=provided[i]->bytes)))
        throw std::invalid_argument("Pin return producer has another original ingress operand");
    const auto replies=prefix+"/reply_send",seal=prefix+"/seal_pins",header=prefix+"/header_send",frontier=prefix+"/frontier_receive",acquire=prefix+"/acquire_pins";
    requireNames(graph,{replies,seal,header,frontier,acquire});
    auto send=forward->declaration().hasCommands()?std::make_unique<PooledPinReturnTransferStage>(arena,edge,q::PooledPinWireRole::Replies,CapturedTransferEndpoint::Producer):nullptr;
    auto sealed=std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::SealPins,storage,
        q::PooledImmutableOwnerOperands(n::OwnerAction::SealPins,storage->plan(),forward->ownerInputs(n::OwnerAction::Begin))});
    auto sent=std::make_unique<PooledPinReturnTransferStage>(arena,edge,q::PooledPinWireRole::Header,CapturedTransferEndpoint::Producer);
    auto received=std::make_unique<PooledPinTransferStage>(arena,forward,q::PooledPinWireRole::Frontier,CapturedTransferEndpoint::Consumer);
    auto acquired=std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Owner{n::OwnerAction::AcquirePins,storage,
        q::PooledImmutableOwnerOperands(n::OwnerAction::AcquirePins,storage->plan(),forward->ownerInputs(n::OwnerAction::AcquirePins))});
    if(send){graph.addNode(replies,std::move(send),device);graph.addDependency(replies,originalPin);}
    graph.addNode(seal,std::move(sealed),device);graph.addDependency(seal,forward->declaration().hasCommands()?replies:originalPin);
    graph.addNode(header,std::move(sent),device);graph.addDependency(header,seal);
    graph.addNode(frontier,std::move(received),device);graph.addDependency(frontier,header);
    graph.addNode(acquire,std::move(acquired),device);graph.addDependency(acquire,frontier);
    return {forward->declaration().hasCommands()?replies:seal,acquire};
}
void requirePooledMixedPinCompletion(const ComputeGraph& graph,const PooledImmutableMixedPinCompletion& original)
{
    if(isGraphCaptureActive() || !original.cohort)throw std::invalid_argument("Mixed Free insertion requires the actual cold Pin cohort");
    original.cohort->requireCurrent();const auto storage=original.cohort->coordinatorStorage();const auto device=storage->plan().allocator().device;
    const auto& owners=original.cohort->owners();const auto local=std::get<std::shared_ptr<const q::PooledImmutableLocalOwnerBinding>>(owners[original.cohort->localCoordinate()]);
    if(original.replies.size()!=owners.size() || original.frontiers.size()!=owners.size() || original.coordinator.last!=original.frontiers.back())
        throw std::invalid_argument("Mixed Pin completion lost original full frontier boundaries");
    const auto node=[&](const std::string& name){const auto* value=graph.getNode(name);if(!value || value->device!=device)throw std::invalid_argument("Mixed Pin lost an original coordinator-local node");return value;};
    const auto depends=[&](const std::string& consumer,const std::string& producer){const auto* c=node(consumer);node(producer);if(std::find(c->dependencies.begin(),c->dependencies.end(),producer)==c->dependencies.end())throw std::invalid_argument("Mixed Pin lost an actual original producer/read frontier");};
    const auto ownerPhase=[&](const std::string& name,n::OwnerAction action){const auto* value=node(name);const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(value->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Owner>(&stage->phase()):nullptr;const auto expected=local->ownerOperands(action);if(!actual || actual->storage!=local->ownerStorage() || actual->action!=action || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Pin replaced its actual local owner phase/inputs");};
    const auto localCoordinator=[&](const std::string& name,n::CoordinatorAction action){const auto* value=node(name);const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(value->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;const auto expected=local->coordinatorOperands(action);if(!actual || actual->storage!=storage || actual->action!=action || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Pin replaced its actual local metadata/Grant producer");};
    requireProducer(graph,original.beginProducer,n::CoordinatorAction::Begin,storage.get());requireProducer(graph,original.commandsProduced,n::CoordinatorAction::PinCommands,storage.get());
    ownerPhase(original.localBegin,n::OwnerAction::Begin);ownerPhase(original.localPin,n::OwnerAction::Pin);ownerPhase(original.localSeal,n::OwnerAction::SealPins);ownerPhase(original.localAcquire,n::OwnerAction::AcquirePins);
    depends(original.localBegin,original.beginProducer);depends(original.commandsProduced,original.localBegin);depends(original.localPin,original.commandsProduced);depends(original.localSeal,original.localPin);
    requirePooledOriginalDependencies(graph,original.localBegin,{original.beginProducer});
    requirePooledOriginalDependencies(graph,original.localPin,{original.commandsProduced});
    requirePooledOriginalDependencies(graph,original.localSeal,{original.localPin});
    std::vector<std::string> commandDependencies{original.beginProducer,original.localBegin};
    std::string previous;
    for(size_t i=0;i<owners.size();++i){const auto& names=original.replies[i];const bool direct=i==local->coordinate();
        if(names.coordinate!=i || (names.path==PooledPinReplyOwnerNodes::Path::DirectLocal)!=direct)throw std::invalid_argument("Mixed Pin changed original owner order/local path");
        const bool positive=direct?local->hasPayload():std::get<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>>(owners[i])->forward()->declaration().hasCommands();
        if(positive!=!names.replyAcquired.empty() || (!direct && positive!=!names.replyReceived.empty()) || (direct && !names.replyReceived.empty()))throw std::invalid_argument("Mixed Pin changed original local/C0 payload phases");
        const auto first=direct && positive?names.replyAcquired:positive?names.replyReceived:names.headerReceived;
        depends(first,original.commandsProduced);for(const auto& other:original.replies)depends(first,other.commandSent);if(!previous.empty())depends(first,previous);
        std::vector<std::string> readers{original.commandsProduced};for(const auto& other:original.replies)readers.push_back(other.commandSent);if(!previous.empty())readers.push_back(previous);
        if(direct && positive)readers.push_back(original.localSeal);
        if(direct && !positive)readers.push_back(original.localSeal);
        requirePooledOriginalDependencies(graph,first,std::move(readers));
        if(direct){
            if(names.commandSent!=original.localPin)throw std::invalid_argument("Mixed Pin substituted local command-bank reader");
            if(positive){localCoordinator(names.replyAcquired,n::CoordinatorAction::AcquirePins);depends(names.replyAcquired,original.localSeal);depends(names.headerReceived,names.replyAcquired);}
            localCoordinator(names.headerReceived,n::CoordinatorAction::ImportLocalPinHeader);depends(names.headerReceived,original.localSeal);
            if(positive)requirePooledOriginalDependencies(graph,names.headerReceived,{original.localSeal,names.replyAcquired});
            if(original.frontiers[i]!=original.localAcquire)throw std::invalid_argument("Mixed Pin substituted local full-frontier consumer");
        }else{
            const auto edge=std::get<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>>(owners[i]);const auto forward=edge->forward();
            const auto forwardNode=[&](const std::string& name,q::PooledPinWireRole role){const auto* stage=dynamic_cast<const PooledPinTransferStage*>(node(name)->stage.get());if(!stage || !stage->matches(forward.get(),role,CapturedTransferEndpoint::Producer))throw std::invalid_argument("Mixed Pin replaced its actual remote forward channel");};
            forwardNode(names.commandSent,positive?q::PooledPinWireRole::Commands:q::PooledPinWireRole::Invocation);forwardNode(original.frontiers[i],q::PooledPinWireRole::Frontier);
            std::string invocation;
            for(const auto& dependency:node(original.commandsProduced)->dependencies){const auto* candidate=graph.getNode(dependency);const auto* stage=candidate?dynamic_cast<const PooledPinTransferStage*>(candidate->stage.get()):nullptr;
                if(stage && stage->matches(forward.get(),q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Producer)){if(!invocation.empty())throw std::invalid_argument("Mixed Pin duplicated original Invocation reader");invocation=dependency;}}
            if(invocation.empty() || (!positive && invocation!=names.commandSent))throw std::invalid_argument("Mixed Pin lost original Invocation reader before command-bank writer");
            requirePooledOriginalDependencies(graph,invocation,{original.beginProducer});commandDependencies.push_back(invocation);
            requirePooledOriginalDependencies(graph,names.commandSent,{positive?original.commandsProduced:original.beginProducer});
            const auto returned=[&](const std::string& name,q::PooledPinWireRole role){const auto* stage=dynamic_cast<const PooledPinReturnTransferStage*>(node(name)->stage.get());if(!stage || !stage->matches(edge.get(),role,CapturedTransferEndpoint::Consumer))throw std::invalid_argument("Mixed Pin replaced its actual remote return channel");};
            if(positive){returned(names.replyReceived,q::PooledPinWireRole::Replies);const auto* stage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(names.replyAcquired)->stage.get());const auto* actual=stage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&stage->phase()):nullptr;const q::PooledImmutableCoordinatorOperands expected(n::CoordinatorAction::AcquirePins,storage->plan(),{},edge->coordinatorInputs(n::CoordinatorAction::AcquirePins));if(!actual || actual->storage!=storage || actual->action!=n::CoordinatorAction::AcquirePins || actual->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Pin replaced its remote Grant acquisition");depends(names.replyAcquired,names.replyReceived);depends(names.headerReceived,names.replyAcquired);requirePooledOriginalDependencies(graph,names.replyAcquired,{names.replyReceived});requirePooledOriginalDependencies(graph,names.headerReceived,{names.replyAcquired});}
            returned(names.headerReceived,q::PooledPinWireRole::Header);
        }
        previous=names.headerReceived;
    }
    const auto* sealStage=dynamic_cast<const PooledImmutableNativePhaseStage*>(node(original.seal)->stage.get());const auto* seal=sealStage?std::get_if<PooledImmutableNativePhaseStage::Coordinator>(&sealStage->phase()):nullptr;
    q::PooledNativeOperandSpans spans{};spans[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{storage->roots().native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,storage->plan().parameters().native.ingress_frontier,owners.size()*sizeof(n::p::PinHeader)};
    const q::PooledImmutableCoordinatorOperands expected(n::CoordinatorAction::SealPins,storage->plan(),{},std::move(spans));
    if(!seal || seal->storage!=storage || seal->action!=n::CoordinatorAction::SealPins || seal->operands.declarationIdentity()!=expected.declarationIdentity())throw std::invalid_argument("Mixed Pin replaced complete original SealPins frontier");
    requirePooledOriginalDependencies(graph,original.commandsProduced,std::move(commandDependencies));
    depends(original.seal,previous);requirePooledOriginalDependencies(graph,original.seal,{previous});previous=original.seal;for(const auto& frontier:original.frontiers){depends(frontier,previous);requirePooledOriginalDependencies(graph,frontier,{previous});previous=frontier;}
}
PooledImmutableMixedPinCompletion appendPooledPinCoordinatorMembers(ComputeGraph& graph,BufferArena& arena,
    std::span<const q::PooledImmutablePinParticipant> owners,std::shared_ptr<const q::IPooledImmutableNativeKernels> code,
    std::span<const std::string> commandSent,const std::string& beginProducer,const std::string& commandsProduced,const std::string& prefix)
{
    if(isGraphCaptureActive() || owners.empty() || !code || prefix.empty() || commandSent.size()!=owners.size())
        throw std::invalid_argument("Pin gather needs its actual complete original cohort/code/producers");
    std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> coordinator;
    std::shared_ptr<const q::PooledImmutableLocalOwnerBinding> local;
    std::vector<std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> remote(owners.size());
    for(size_t i=0;i<owners.size();++i)std::visit([&](const auto& binding){
        if(!binding)throw std::invalid_argument("Pin gather has no original owner");binding->requireCurrent();
        using Binding=typename std::decay_t<decltype(binding)>::element_type;
        std::shared_ptr<q::PooledImmutableCoordinatorPreparedStorage> c;std::shared_ptr<q::PooledImmutableOwnerPreparedStorage> o;
        if constexpr(std::is_same_v<Binding,const q::PooledImmutableLocalOwnerBinding>){
            if(local || !commandSent[i].empty())throw std::invalid_argument("Local Pin has duplicate ownership or a fabricated send node");
            local=binding;c=binding->coordinatorStorage();o=binding->ownerStorage();
        }else{remote[i]=binding;c=binding->forward()->coordinatorStorage();o=binding->forward()->ownerStorage();}
        if(!coordinator)coordinator=c;
        if(c!=coordinator || o->plan().parameters().owner.coordinate()!=i)throw std::invalid_argument("Pin gather changed original coordinator/member order");
        coordinator->plan().requireOwnerPlan(o->plan());
    },owners[i]);
    const auto& plan=coordinator->plan();const auto device=plan.allocator().device;
    if(owners.size()!=plan.parameters().policy.participants() || code->backend()!=device.type)
        throw std::invalid_argument("Pin gather changed original source/cohort/backend");
    requireProducer(graph,commandsProduced,n::CoordinatorAction::PinCommands,coordinator.get());
    PooledImmutableMixedPinCompletion result{};result.commandsProduced=commandsProduced;result.beginProducer=beginProducer;
    if(local){
        requireProducer(graph,beginProducer,n::CoordinatorAction::Begin,coordinator.get());
        const auto* commands=graph.getNode(commandsProduced);
        if(std::find(commands->dependencies.begin(),commands->dependencies.end(),beginProducer)==commands->dependencies.end())
            throw std::invalid_argument("Local Pin requires original direct Begin-to-PinCommands dependency");
        result.cohort=std::make_shared<const q::PooledImmutableMixedCohort>(q::PooledImmutableMixedCohort::fromOriginal(std::vector<q::PooledImmutablePinParticipant>(owners.begin(),owners.end())));
        result.localBegin=prefix+"/local_begin";result.localPin=prefix+"/local_pin";result.localSeal=prefix+"/local_seal_pins";
        result.localAcquire=prefix+"/owner"+std::to_string(local->coordinate())+"/frontier_local_acquire";
        requireNames(graph,{result.localBegin,result.localPin,result.localSeal,result.localAcquire});
    }
    std::vector<PooledPinReplyOwnerNodes> names;std::vector<std::unique_ptr<IComputeStage>> replies,acquires,headers,frontiers;
    const auto localOwner=[&](n::OwnerAction action){return std::make_unique<PooledImmutableNativePhaseStage>(arena,code,
        PooledImmutableNativePhaseStage::Owner{action,local->ownerStorage(),local->ownerOperands(action)});};
    auto begin=local?localOwner(n::OwnerAction::Begin):nullptr;
    auto pin=local?localOwner(n::OwnerAction::Pin):nullptr;
    auto localSeal=local?localOwner(n::OwnerAction::SealPins):nullptr;
    for(size_t i=0;i<owners.size();++i){
        const bool direct=local && i==local->coordinate();const bool positive=direct?local->hasPayload():remote[i]->forward()->declaration().hasCommands();
        const auto base=prefix+"/owner"+std::to_string(i);
        if(!direct){
            const auto* producer=graph.getNode(commandSent[i]);const auto* actual=producer?dynamic_cast<const PooledPinTransferStage*>(producer->stage.get()):nullptr;
            if(!actual || !actual->matches(remote[i]->forward().get(),positive?q::PooledPinWireRole::Commands:q::PooledPinWireRole::Invocation,CapturedTransferEndpoint::Producer))
                throw std::invalid_argument("Pin gather command frontier is not its actual original forward endpoint");
        }
        PooledPinReplyOwnerNodes nms{static_cast<uint32_t>(i),direct?result.localPin:commandSent[i],positive && !direct?base+"/reply_receive":"",
            positive?base+"/reply_acquire":"",base+(direct?"/header_local_import":"/header_receive"),
            direct?PooledPinReplyOwnerNodes::Path::DirectLocal:PooledPinReplyOwnerNodes::Path::Remote};
        requireNames(graph,{base+"/reply_receive",base+"/reply_acquire",nms.headerReceived,base+"/frontier_send"});
        if(direct){
            replies.push_back(nullptr);
            acquires.push_back(positive?std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{
                n::CoordinatorAction::AcquirePins,coordinator,local->coordinatorOperands(n::CoordinatorAction::AcquirePins)}):nullptr);
            headers.push_back(std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{
                n::CoordinatorAction::ImportLocalPinHeader,coordinator,local->coordinatorOperands(n::CoordinatorAction::ImportLocalPinHeader)}));
            frontiers.push_back(localOwner(n::OwnerAction::AcquirePins));
        }else{
            const auto& edge=remote[i];const auto& forward=edge->forward();
            replies.push_back(positive?std::make_unique<PooledPinReturnTransferStage>(arena,edge,q::PooledPinWireRole::Replies,CapturedTransferEndpoint::Consumer):nullptr);
            acquires.push_back(positive?std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{
                n::CoordinatorAction::AcquirePins,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::AcquirePins,plan,{},edge->coordinatorInputs(n::CoordinatorAction::AcquirePins))}):nullptr);
            headers.push_back(std::make_unique<PooledPinReturnTransferStage>(arena,edge,q::PooledPinWireRole::Header,CapturedTransferEndpoint::Consumer));
            frontiers.push_back(std::make_unique<PooledPinTransferStage>(arena,forward,q::PooledPinWireRole::Frontier,CapturedTransferEndpoint::Producer));
        }
        names.push_back(std::move(nms));
    }
    const auto seal=prefix+"/seal_pins";requireNames(graph,{seal});
    q::PooledNativeOperandSpans full{};const auto roots=coordinator->roots();
    full[static_cast<unsigned>(q::PooledNativeOperand::Frontier)]=q::PooledNativeTensorSpan{roots.native,BufferId::QFN_POOLED_COORDINATOR_NATIVE,
        plan.parameters().native.ingress_frontier,owners.size()*sizeof(n::p::PinHeader)};
    auto sealed=std::make_unique<PooledImmutableNativePhaseStage>(arena,code,PooledImmutableNativePhaseStage::Coordinator{
        n::CoordinatorAction::SealPins,coordinator,q::PooledImmutableCoordinatorOperands(n::CoordinatorAction::SealPins,plan,{},std::move(full))});
    if(local){
        graph.addNode(result.localBegin,std::move(begin),device);graph.addDependency(result.localBegin,beginProducer);graph.addDependency(commandsProduced,result.localBegin);
        graph.addNode(result.localPin,std::move(pin),device);graph.addDependency(result.localPin,commandsProduced);
        graph.addNode(result.localSeal,std::move(localSeal),device);graph.addDependency(result.localSeal,result.localPin);
    }
    for(size_t i=0;i<owners.size();++i){
        if(replies[i])graph.addNode(names[i].replyReceived,std::move(replies[i]),device);
        if(acquires[i])graph.addNode(names[i].replyAcquired,std::move(acquires[i]),device);
        graph.addNode(names[i].headerReceived,std::move(headers[i]),device);
        if(local && i==local->coordinate()){
            graph.addDependency(names[i].headerReceived,result.localSeal);
            if(!names[i].replyAcquired.empty())graph.addDependency(names[i].replyAcquired,result.localSeal);
        }
    }
    wirePooledPinReplyBankLifetime(graph,plan,names,commandsProduced);
    graph.addNode(seal,std::move(sealed),device);graph.addDependency(seal,names.back().headerReceived);
    std::string last=seal;
    for(size_t i=0;i<owners.size();++i){const auto sent=local && i==local->coordinate()?result.localAcquire:prefix+"/owner"+std::to_string(i)+"/frontier_send";
        graph.addNode(sent,std::move(frontiers[i]),device);graph.addDependency(sent,last);last=sent;result.frontiers.push_back(sent);}
    const auto& first=names.front();result.coordinator={first.path==PooledPinReplyOwnerNodes::Path::DirectLocal && !first.replyAcquired.empty()?first.replyAcquired:
        first.replyReceived.empty()?first.headerReceived:first.replyReceived,last};
    result.replies=names;result.seal=seal;
    return result;
}
PooledPinParticipantNodes appendPooledPinCoordinatorReturn(ComputeGraph& graph,BufferArena& arena,
    std::span<const std::shared_ptr<const q::PooledImmutableCapturedPinReturn>> owners,
    std::shared_ptr<const q::IPooledImmutableNativeKernels> code,std::span<const std::string> commandSent,
    const std::string& commandsProduced,const std::string& prefix)
{
    std::vector<q::PooledImmutablePinParticipant> members;for(const auto& owner:owners)members.emplace_back(owner);
    return appendPooledPinCoordinatorMembers(graph,arena,members,std::move(code),commandSent,{},commandsProduced,prefix).coordinator;
}

}

namespace llaminar2 {
/** @brief Inspect only the exact original endpoint stage types without another object/vtable allocation. */
std::optional<qwen4exp::PooledImmutableDirectedTransferOperation> describePooledPinTransfer(const IComputeStage& stage){
    if(const auto* original=dynamic_cast<const PooledPinTransferStage*>(&stage))return original->directedTransferOperation();
    if(const auto* original=dynamic_cast<const PooledPinReturnTransferStage*>(&stage))return original->directedTransferOperation();
    return std::nullopt;
}
}
