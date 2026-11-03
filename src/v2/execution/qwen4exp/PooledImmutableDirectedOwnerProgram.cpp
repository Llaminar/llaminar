/**
 * @file PooledImmutableDirectedOwnerProgram.cpp
 * @brief Declare and seal the original flat directed-owner native operation recipe.
 *
 * Native invocation counts come from the launcher-shared owner plan and the
 * TransferEngine-shared transfer selector. Addressed message extents come from
 * the same native geometry used by actual bindings, including C0 metadata.
 */
#include "PooledImmutableDirectedOwnerProgram.h"
#include "PooledImmutableDirectedMessageGeometry.h"
#include "PooledImmutableDirectedOperation.h"
#include "PooledImmutableParticipantRetirementGraph.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/compute_stages/stages/qwen4exp/PooledImmutableNativePhaseStage.h"
#include "kernels/qwen4exp/PooledImmutableOwnerLaunchPlan.h"
#include "backends/NativeOrderingEventProgram.h"
#include "utils/Sha256.h"
#include <variant>
#include <vector>
#include <limits>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
namespace p=adaptive_lookup::pooled;
using Family=PooledImmutableDirectedWireFamily;
/** @brief Exact native branch/address ABI, independent of physical pointers. */
struct OwnerOperation {n::OwnerAction action;n::SourceCondition condition;n::RecordLayout records;bool operator==(const OwnerOperation&)const=default;};
using Operation=std::variant<OwnerOperation,PooledImmutableDirectedTransferOperation>;
using Recipe=std::vector<Operation>;
/** @brief Append the source-declared recipe in precisely its original stage order. */
Recipe recipe(const PooledImmutableDirectedMessageGeometry& g){
    Recipe out;out.reserve(80);const bool positive=g.replyCapacity()!=0;
    const auto owner=[&](n::OwnerAction action,n::SourceCondition condition=n::SourceCondition::Always,n::RecordLayout records=n::RecordLayout::Compact){out.emplace_back(OwnerOperation{action,condition,records});};
    const auto wire=[&](Family family,auto role,CapturedTransferEndpoint endpoint,std::size_t bytes,bool counted=false){
        if(!bytes)throw std::logic_error("Directed recipe cannot publish an absent payload endpoint");
        out.emplace_back(PooledImmutableDirectedTransferOperation{family,static_cast<std::uint32_t>(role),endpoint,bytes,
            counted?(endpoint==CapturedTransferEndpoint::Producer?CapturedTransferExtentSource::ProducerDevice:CapturedTransferExtentSource::ProducerPublication):CapturedTransferExtentSource::FixedMessage,false});
    };
    constexpr auto receive=CapturedTransferEndpoint::Consumer,send=CapturedTransferEndpoint::Producer;
    wire(Family::Pin,PooledPinWireRole::Invocation,receive,sizeof(n::Invocation));owner(n::OwnerAction::Begin);
    if(positive)wire(Family::Pin,PooledPinWireRole::Commands,receive,g.commands(),true);
    owner(n::OwnerAction::Pin);
    if(positive)wire(Family::PinReturn,PooledPinWireRole::Replies,send,g.grants(),true);
    owner(n::OwnerAction::SealPins);wire(Family::PinReturn,PooledPinWireRole::Header,send,sizeof(p::PinHeader));
    wire(Family::Pin,PooledPinWireRole::Frontier,receive,g.pinHeaders());owner(n::OwnerAction::AcquirePins);
    owner(n::OwnerAction::Availability);wire(Family::Free,PooledFreeWireRole::Availability,send,sizeof(n::Availability));
    wire(Family::Free,PooledFreeWireRole::Plan,receive,sizeof(n::FreePlanHeader));
    if(positive)wire(Family::Free,PooledFreeWireRole::Decisions,receive,g.decisions(),true);
    owner(n::OwnerAction::AcceptFree,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);owner(n::OwnerAction::ExecuteFree);
    if(positive)wire(Family::Free,PooledFreeWireRole::Grants,send,g.grants(),true);
    owner(n::OwnerAction::FreeReceipt);wire(Family::Free,PooledFreeWireRole::Receipt,send,sizeof(p::FreeAllocationReceipt));
    wire(Family::Free,PooledFreeWireRole::Frontier,receive,g.freeReceipts());owner(n::OwnerAction::AcquireFree);
    wire(Family::Victim,PooledVictimWireRole::Plan,receive,sizeof(n::VictimPlanHeader));
    if(positive)wire(Family::Victim,PooledVictimWireRole::Decisions,receive,g.decisions(),true);
    owner(n::OwnerAction::AcceptVictim,n::SourceCondition::Always,n::RecordLayout::TransportedAllocationPrefix);owner(n::OwnerAction::ExecuteVictim);
    if(positive)wire(Family::Victim,PooledVictimWireRole::Grants,send,g.grants(),true);
    owner(n::OwnerAction::VictimReceipt);wire(Family::Victim,PooledVictimWireRole::Receipt,send,sizeof(p::VictimAllocationReceipt));
    wire(Family::Source,PooledSourceWireRole::InitialFrontier,receive,sizeof(n::SourceFrontier));owner(n::OwnerAction::SealSource);
    wire(Family::Source,PooledSourceWireRole::SealReceipt,send,sizeof(n::OwnerSourceSealReceipt));
    if(positive){wire(Family::Source,PooledSourceWireRole::Fills,receive,g.fills(),true);wire(Family::Source,PooledSourceWireRole::Payload,receive,g.rows(),true);}
    wire(Family::Source,PooledSourceWireRole::TerminalFrontier,receive,sizeof(n::SourceFrontier));owner(n::OwnerAction::DrainSourceFailure);
    owner(n::OwnerAction::ApplySourceDelivery,n::SourceCondition::AfterSuccess);owner(n::OwnerAction::ValidateFills,n::SourceCondition::AfterSuccess);owner(n::OwnerAction::DrainSourceFailure);
    owner(n::OwnerAction::PublishWire);wire(Family::Publication,PooledPublicationWireRole::PublicationHeader,send,n::NominalHeaderWireBytes);
    wire(Family::Publication,PooledPublicationWireRole::PublicationGate,receive,sizeof(n::SourceFrontier));
    wire(Family::Publication,PooledPublicationWireRole::PublicationFrontier,receive,g.publicationHeaders());
    owner(n::OwnerAction::AcquirePublication,n::SourceCondition::AfterSuccess);owner(n::OwnerAction::PublicationAcquiredWire);
    wire(Family::Publication,PooledPublicationWireRole::AcquiredHeader,send,n::NominalHeaderWireBytes);
    wire(Family::Publication,PooledPublicationWireRole::ReadyGate,receive,sizeof(n::SourceFrontier));
    owner(n::OwnerAction::MakeReady,n::SourceCondition::AfterSuccess);owner(n::OwnerAction::ReadyReceipt,n::SourceCondition::AfterSuccess);owner(n::OwnerAction::ReadyWire);
    wire(Family::Publication,PooledPublicationWireRole::ReadyHeader,send,n::NominalHeaderWireBytes);
    wire(Family::Read,PooledReadWireRole::Outcome,receive,sizeof(n::SourceFrontier));
    if(positive)wire(Family::Read,PooledReadWireRole::Requests,receive,g.grants(),true);
    owner(n::OwnerAction::ReadRows,n::SourceCondition::AfterSuccess);wire(Family::Read,PooledReadWireRole::Receipt,send,sizeof(n::HitReadReceipt));
    if(positive)wire(Family::Read,PooledReadWireRole::Rows,send,g.rows(),true);
    wire(Family::Retirement,PooledRetirementWireRole::ReadersRetired,receive,sizeof(n::ReadersRetiredGate));
    owner(n::OwnerAction::AcquireReaderRetirement);owner(n::OwnerAction::DrainSourceFailure);owner(n::OwnerAction::Release);
    if(positive)wire(Family::Retirement,PooledRetirementWireRole::Releases,send,g.releases(),true);
    owner(n::OwnerAction::Retire);owner(n::OwnerAction::RetiredWire);wire(Family::Retirement,PooledRetirementWireRole::Retired,send,n::NominalHeaderWireBytes);
    return out;
}

/** @return Complete named native-root layout identity, excluding table/generation/pointers. */
std::uint64_t layoutIdentity(const PooledPhysicalOwnerOffsets& c,const PooledPhysicalFrameOffsets& f,const n::OwnerOffsets& o){
    std::uint64_t h=1469598103934665603ULL;
    for(const std::uint64_t word:{n::Version,c.control,c.slots,c.payload,c.bytes,
        f.control,f.commands,f.fill_order,f.replies,f.acknowledgements,f.allocation_control,f.accepted_plan,f.initial_pins,
        f.validation_a,f.validation_b,f.validation_counts,f.pin_headers,f.publish_headers,f.free_headers,f.victim_headers,f.release_receipts,f.bytes,
        o.progress,o.header,o.replies,o.ingress,o.source_outcome,o.source_fills,o.source_payload,o.bytes,std::uint64_t{o.reply_capacity}})
        h=(h^word)*1099511628211ULL;
    return h;
}
/** @brief Serialize fixed-width words in a stable order before hashing the actual portable dispatch ABI. */
void append(std::string& bytes,std::uint64_t word){for(unsigned i=0;i<8;++i)bytes.push_back(static_cast<char>((word>>(8*i))&255));}
/** @return Source-derived class and strict flat native envelope for one ordered recipe. */
std::pair<std::array<std::uint8_t,32>,GPUFlatGraphLimits> seal(const Recipe& operations,std::uint32_t replyCapacity,std::uint32_t keyBits,std::uint32_t rootMask){
    std::string bytes;bytes.reserve(operations.size()*64+128);
    for(const auto word:{std::uint64_t{1},n::Version,std::uint64_t{sizeof(n::OwnerArgs)},std::uint64_t{alignof(n::OwnerArgs)},
        std::uint64_t{sizeof(CapturedTransferChannelDeviceBinding)},std::uint64_t{alignof(CapturedTransferChannelDeviceBinding)},
        std::uint64_t{keyBits},std::uint64_t{rootMask}})append(bytes,word);
    std::uint64_t nodes=0;
    for(const auto& operation:operations){
        append(bytes,operation.index());
        if(const auto* owner=std::get_if<OwnerOperation>(&operation)){
            const auto count=n::ownerKernelCount(owner->action,replyCapacity);nodes+=count;
            append(bytes,static_cast<std::uint64_t>(owner->action));append(bytes,static_cast<std::uint64_t>(owner->condition));
            append(bytes,static_cast<std::uint64_t>(owner->records));append(bytes,count);
        }else{
            const auto& wire=std::get<PooledImmutableDirectedTransferOperation>(operation);const auto count=wire.kernelCount();nodes+=count;
            append(bytes,static_cast<std::uint64_t>(wire.family));append(bytes,wire.role);append(bytes,static_cast<std::uint64_t>(wire.endpoint));
            append(bytes,wire.maximumBytes);append(bytes,static_cast<std::uint64_t>(wire.extent));append(bytes,wire.sourceSlice);append(bytes,count);
        }
    }
    if(!nodes || nodes>std::numeric_limits<std::uint32_t>::max())throw std::overflow_error("Directed owner native graph scope overflow");
    const auto hex=sha256BytesHex(bytes);if(!hex || hex->size()!=64)throw std::runtime_error("Directed owner program dispatch SHA256 failed");
    std::array<std::uint8_t,32> digest{};
    const auto digit=[](char c)->unsigned{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::logic_error("Directed owner dispatch hash was not canonical hexadecimal");};
    for(std::size_t i=0;i<digest.size();++i)digest[i]=static_cast<std::uint8_t>(16*digit((*hex)[2*i])+digit((*hex)[2*i+1]));
    // Every original stage and every internal multi-kernel dispatch uses one
    // exact stream, forming a chain. No child, conditional, callback, allocation,
    // copy, memset or publication event is part of this narrowly authored graph.
    return {digest,{static_cast<std::uint32_t>(nodes),static_cast<std::uint32_t>(nodes-1),GPUFlatGraphLimits::bit(GPUFlatGraphNodeKind::Kernel)}};
}
/** @return Exact metadata only from an original typed endpoint class; reject arbitrary stage lookalikes. */
std::optional<PooledImmutableDirectedTransferOperation> actualTransfer(const IComputeStage& stage){
    for(const auto describe:{describePooledPinTransfer,describePooledFreeTransfer,describePooledVictimTransfer,describePooledSourceTransfer,
        describePooledPublicationTransfer,describePooledReadTransfer,describePooledRetirementTransfer})if(auto value=describe(stage))return value;
    return std::nullopt;
}
/** @brief Require the exact supported participant relationship before authoring a flat graph. */
void requirePlacement(PhysicalMemoryAllocatorIdentity coordinator, PhysicalMemoryAllocatorIdentity owner,
    std::uint32_t coordinate, std::uint32_t participants, std::optional<PeerAccessCoverage> observation)
{
    if (coordinate >= participants || owner == coordinator)
        throw std::invalid_argument("Directed owner graph requires its distinct declared storage participant");
    ImmutableRowTableDirectedTransportPolicy::require(coordinator, owner, observation);
}
}
PooledRemoteGraphProgram PooledImmutableDirectedOwnerProgram::declare(const PooledImmutablePhysicalOwnerGeometry& physical,
    const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate){
    const auto native=PooledImmutableOwnerNativePlan::rootOffsets(physical,coordinator,coordinate);const auto& owner=physical.owners()[coordinate];
    const auto policy=coordinator.parameters();requirePlacement(coordinator.coordinator(),owner.allocator,coordinate,policy.participants(),physical.peerAccess());
    const auto mask=NativeOrderingEventProgram::pooledOwner(owner.cache.bytes!=0).nonemptyRootMask();
    const std::uint32_t bits=physical.rowKeyWidth()==adaptive_lookup::RowKeyWidth::UInt32?32u:64u;
    const PooledRemoteGraphGeometry geometry{coordinator.coordinator(),owner.allocator,coordinate,policy.participants(),owner.slots,
        policy.maximumOriginal(),policy.maximumUnique(),physical.extent().rows(),physical.rowBytes(),physical.slotStride(),
        layoutIdentity(owner.cache,owner.frame,native),bits,mask,physical.peerAccess()};
    const auto [digest,limits]=seal(recipe(PooledImmutableDirectedMessageGeometry::fromGeometry(physical,coordinator,coordinate)),native.reply_capacity,bits,mask);
    return PooledRemoteGraphProgram(geometry,digest,limits);
}
PooledRemoteGraphProgram PooledImmutableDirectedOwnerProgram::fromOriginalPlans(const PooledImmutableOwnerNativePlan& owner,
    const PooledImmutableCoordinatorNativePlan& coordinator){
    coordinator.requireOwnerPlan(owner);const auto& o=owner.parameters();
    requirePlacement(coordinator.allocator(),owner.allocator(),o.owner.coordinate(),o.owner.participants(),owner.peerAccess());
    const auto mask=NativeOrderingEventProgram::pooledOwner(o.cache.bytes!=0).nonemptyRootMask();
    const std::uint32_t bits=o.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32?32u:64u;
    const PooledRemoteGraphGeometry geometry{coordinator.allocator(),owner.allocator(),o.owner.coordinate(),o.owner.participants(),o.owner.capacity(),
        o.maximum_original,o.owner.maximumUnique(),o.owner.sourceRows(),o.row_bytes,o.slot_stride,layoutIdentity(o.cache,o.frame,o.native),bits,mask,owner.peerAccess()};
    const auto [digest,limits]=seal(recipe(PooledImmutableDirectedMessageGeometry::fromPlan(owner)),o.native.reply_capacity,bits,mask);
    return PooledRemoteGraphProgram(geometry,digest,limits);
}
void PooledImmutableDirectedOwnerProgram::requireActual(const ComputeGraph& graph,const PooledImmutableOwnerNativePlan& owner,
    const PooledImmutableCoordinatorNativePlan& coordinator,const PooledRemoteGraphProgram& program){
    const auto& o=owner.parameters();
    if(program!=fromOriginalPlans(owner,coordinator))throw std::logic_error("Directed owner actual native geometry/dispatch differs from its original declaration");
    const auto expected=recipe(PooledImmutableDirectedMessageGeometry::fromPlan(owner));
    const auto& order=graph.getExecutionOrder();if(order.size()!=expected.size())throw std::logic_error("Directed owner actual stage population differs from its original recipe");
    for(std::size_t i=0;i<order.size();++i){
        const auto* node=graph.getNode(order[i]);if(!node || node->device!=owner.allocator().device || !node->stage ||
            node->dependencies!=(i?std::vector<std::string>{order[i-1]}:std::vector<std::string>{}))
            throw std::logic_error("Directed owner must preserve the original exact single-stream chain");
        if(const auto* phase=dynamic_cast<const PooledImmutableNativePhaseStage*>(node->stage.get())){
            const auto* actual=std::get_if<PooledImmutableNativePhaseStage::Owner>(&phase->phase());const auto* declared=std::get_if<OwnerOperation>(&expected[i]);
            if(!actual || !declared || actual->storage->plan().parameters().identity!=o.identity ||
                OwnerOperation{actual->action,actual->operands.sourceCondition(),actual->operands.recordLayout()}!=*declared)
                throw std::logic_error("Directed owner changed an actual original kernel phase/address ABI");
        }else{
            const auto actual=actualTransfer(*node->stage);const auto* declared=std::get_if<PooledImmutableDirectedTransferOperation>(&expected[i]);
            if(!actual || !declared || *actual!=*declared)throw std::logic_error("Directed owner changed an actual addressed transfer role/count/slice/dispatch");
        }
    }
}
}
