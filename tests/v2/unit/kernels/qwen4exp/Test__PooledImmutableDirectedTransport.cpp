/**
 * @file Test__PooledImmutableDirectedTransport.cpp
 * @brief Replay exact directed channel floor/growth/runtime capacities without devices or invented source permission.
 *
 * Recorded table/generation metadata is used only to compare pure runtime plan
 * projections. No worker, source permit, native executable or certificate is
 * created. Real captured wire extents remain covered by the original native
 * cross-vendor owner replays and the TransferEngine counted-message suite.
 */
#include "execution/qwen4exp/PooledImmutableDirectedTransport.h"
#include "execution/qwen4exp/PooledImmutableSourceRootBinding.h"
#include "execution/qwen4exp/PooledImmutableNativeTransfer.h"
#include "execution/qwen4exp/PooledImmutableReplicatedReaderContract.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/DeviceGraphCaptureController.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"
#include "../../../fixtures/collective/PooledNativeCollectiveReplayAccess.h"
#include <gtest/gtest.h>
#include <array>
#include <map>
#include <set>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
struct Shape {q::PooledImmutablePhysicalOwnerGeometry physical;q::PooledImmutableCoordinatorGeometry coordinator;std::vector<PhysicalMemoryResource> resources;};
/** @return Original declared owner order, never compute-rank order or an admission permission. */
Shape shape(DeviceId source,std::span<const std::uint32_t> slots,bool sameVendor=false,std::optional<PeerAccessCoverage> peerAccess=std::nullopt){
    const PhysicalMemoryAllocatorIdentity coordinator{0,source};std::vector<PhysicalMemoryAllocatorIdentity> order;
    std::vector<ImmutableRowTableLayoutOwner> owners;std::vector<PhysicalMemoryResource> resources{{0,source,1ULL<<34,1ULL<<34}};bool positive=false;
    // Include actual local C0 owner followed by storage-only opposite-vendor owners.
    order.push_back(coordinator);owners.push_back({{coordinator,ImmutableRowStorage::VRAM},0,0,{}});
    for(std::size_t i=0;i<slots.size();++i){const auto type=sameVendor?source.type:(source.is_cuda()?DeviceType::ROCm:DeviceType::CUDA);
        const auto device=type==DeviceType::CUDA?DeviceId::cuda(i):DeviceId::rocm(i);const PhysicalMemoryAllocatorIdentity owner{0,device};
        order.push_back(owner);owners.push_back({{owner,ImmutableRowStorage::VRAM},0,slots[i],{}});positive|=slots[i]!=0;
        resources.push_back({0,device,1ULL<<34,1ULL<<34});}
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},positive?std::optional(coordinator):std::nullopt,512,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,peerAccess);
    auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},1024);
    return {std::move(physical),std::move(policy),std::move(resources)};
}
/** @return Typed source contribution by actual allocator; no admission/free arithmetic is duplicated. */
std::map<DeviceId,std::size_t> bytes(const PhysicalMemoryPlan& plan){std::map<DeviceId,std::size_t> result;for(const auto& item:plan.resources())result[item.resource().device]+=item.bytes(PhysicalMemoryOwner::ActivationTransportStaging);return result;}
const PhysicalMemoryResource host{0,DeviceId::cpu(),1ULL<<34,1ULL<<34};
}
TEST(PooledImmutableDirectedTransport, OrderedZeroPartialAndShrinkingCandidateGrowthIsExact){
    for(const auto source:{DeviceId::cuda(7),DeviceId::rocm(7)})for(const auto count:{1u,2u,4u})for(unsigned replay=0;replay<20;++replay){
        std::vector<std::uint32_t> slots(count);for(unsigned i=0;i<count;++i)slots[i]=std::array{0u,1u,63u,64u,65u,512u,0u,2u}[(replay+i)%8];
        const auto s=shape(source,slots);auto mandatory=bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(s.physical,s.coordinator,host,s.resources));
        const auto growth=bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(s.physical,s.coordinator,host,s.resources));
        const auto complete=bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources));
        for(const auto& [device,value]:growth)mandatory[device]+=value;
        EXPECT_EQ(mandatory,complete);
        EXPECT_EQ(complete.at(source),count*2*sizeof(CapturedTransferCursor));
        for(const auto& owner:s.physical.owners())if(owner.allocator.device!=source)EXPECT_EQ(complete.at(owner.allocator.device),2*sizeof(CapturedTransferCursor));
    }
}
TEST(PooledImmutableDirectedTransport, PlanningAndOriginalRuntimePinEdgeUseIdenticalPhysicalChannels){
    for(const auto capacity:{0u,1u,64u,512u}){
        const std::array slots{capacity,0u,capacity/2};const auto s=shape(DeviceId::cuda(7),slots);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(s.physical,s.coordinator,701,17);
        std::size_t hostBytes=0;
        for(std::uint32_t coordinate=1;coordinate<s.physical.owners().size();++coordinate){
            const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(s.physical,s.coordinator,coordinate,701,17);
            const auto edge=q::PooledImmutablePinEdge::fromPlans(coordinator,owner);
            const auto free=q::PooledImmutableFreeEdge::fromOriginal(edge);
            EXPECT_EQ(edge.hasCommands(),owner.parameters().native.reply_capacity!=0);
            EXPECT_EQ(free.hasPayload(),edge.hasCommands());
            if(!edge.hasCommands()){
                EXPECT_EQ(edge.maximumCommandBytes(),0u);EXPECT_EQ(edge.replySource().bytes,0u);
                EXPECT_FALSE(edge.incomingCount());EXPECT_FALSE(edge.replyCountDestination());
                EXPECT_EQ(free.maximumDecisionBytes(),0u);EXPECT_FALSE(free.decisionCount());
            }
            const auto planned=q::PooledImmutableDirectedChannelGeometry::fromGeometry(s.physical,s.coordinator,coordinate);
            EXPECT_EQ(edge.channelMemory().mapped_host_bytes,planned.forwardMemory().mapped_host_bytes);
            EXPECT_EQ(edge.returnChannelMemory().mapped_host_bytes,planned.reverseMemory().mapped_host_bytes);
            hostBytes+=edge.channelMemory().mapped_host_bytes+edge.returnChannelMemory().mapped_host_bytes;
        }
        EXPECT_EQ(bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources)).at(DeviceId::cpu()),hostBytes);
    }
}
/** @test Explicit NoPeer retains compact mapped channel costs and original per-device graph identity. */
TEST(PooledImmutableDirectedTransport, ExplicitNoPeerKeepsExactFloorGrowthAndNativeOwnerIdentity)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto count : {1u, 3u, 7u})
            for (const auto capacity : {0u, 64u}) {
                const DeviceId source{type,static_cast<int>(count)};
                const std::vector<std::uint32_t> slots(count, capacity);
                const auto s = shape(source, slots, true, PeerAccessCoverage::None);
                auto floor = bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(s.physical,s.coordinator,host,s.resources));
                const auto growth = bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(s.physical,s.coordinator,host,s.resources));
                const auto full = bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources));
                for (const auto& [device, value] : growth) floor[device] += value;
                EXPECT_EQ(floor, full);
                const auto coordinator = q::PooledImmutableCoordinatorNativePlan::fromGeometry(s.physical,s.coordinator,701,17);
                for (std::uint32_t i = 1; i < s.physical.owners().size(); ++i) {
                    const auto owner = q::PooledImmutableOwnerNativePlan::fromGeometry(s.physical,s.coordinator,i,701,17);
                    const auto edge = q::PooledImmutablePinEdge::fromPlans(coordinator,owner);
                    EXPECT_EQ(owner.peerAccess(), PeerAccessCoverage::None);
                    EXPECT_TRUE(edge.capturedChannelTransport());
                    EXPECT_NO_THROW(edge.channelMemory());
                }
            }
}
TEST(PooledImmutableDirectedTransport, ForeignResourcesAndUnsupportedNativeRoutesCannotAcquireMappedBOM){
    const std::array slots{64u,0u};const auto s=shape(DeviceId::cuda(7),slots);
    auto missing=s.resources;missing.pop_back();EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,missing),std::invalid_argument);
    auto duplicate=s.resources;duplicate.push_back(duplicate.front());EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,duplicate),std::invalid_argument);
    const auto native=shape(DeviceId::cuda(7),slots,true);EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(native.physical,native.coordinator,host,native.resources),std::invalid_argument);
    auto foreign=host;foreign.world_rank=1;EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,foreign,s.resources),std::invalid_argument);
    EXPECT_THROW(q::PooledImmutableDirectedTransport::create({}),std::invalid_argument);
}

/** @test Counted native metadata has one disjoint immutable operand/receipt slot per actual message and participant. */
TEST(PooledImmutableDirectedTransport, NativeCompletionLayoutHasExactPositiveMessagePopulation)
{
    using Message = q::PooledNativeCountedMessage;
    const q::PooledNativeCompletionLayout layout(true);
    std::array<bool, 24> occupied{};
    for (std::size_t message = 0; message < static_cast<std::size_t>(Message::Count); ++message) {
        const auto nominal = static_cast<Message>(message);
        EXPECT_EQ(layout.receiptOffset(nominal), message * sizeof(ncclLlaminarByteCompletion));
        for (const int participant : {1, 0}) {
            const auto offset = layout.operandOffset(nominal, participant);
            ASSERT_EQ(offset % sizeof(ncclLlaminarCompletionOperand), 0u);
            const auto index = offset / sizeof(ncclLlaminarCompletionOperand);
            ASSERT_LT(index, occupied.size());
            EXPECT_FALSE(occupied[index]);
            occupied[index] = true;
            EXPECT_LE(offset + sizeof(ncclLlaminarCompletionOperand), layout.operandBytes());
        }
    }
    EXPECT_TRUE(std::all_of(occupied.begin(), occupied.end(), [](bool value) { return value; }));
    EXPECT_EQ(layout.operandBytes(), 1536u);
    EXPECT_EQ(layout.receiptBytesPerParticipant(), 768u);
    EXPECT_THROW(layout.slot(Message::Count), std::invalid_argument);
    EXPECT_THROW(layout.operandOffset(Message::ReaderResult, -1), std::invalid_argument);
    EXPECT_THROW(layout.operandOffset(Message::ReaderResult, 2), std::invalid_argument);
}

/** @test Native completion admission authenticates page-rounded roots while publishing only logical operands. */
TEST(PooledImmutableDirectedTransport, NativeCompletionMappedBackingKeepsLogicalSpanSeparate)
{
    for (const bool positive : {false, true}) {
        const q::PooledNativeCompletionLayout layout(positive);
        const auto mapped_bytes = TransferEngine::mappedHostRegionAllocationBytes(layout.operandBytes());
        const auto receipt_bytes = layout.receiptBytesPerParticipant();
        ASSERT_GT(mapped_bytes, layout.operandBytes());
        EXPECT_EQ(layout.operandAllocationBytes(), mapped_bytes);
        EXPECT_TRUE(layout.matchesStorageBytes(mapped_bytes, receipt_bytes));
        // Original source admission compared the physical mapped capacity with
        // this smaller logical span and rejected every real native transport.
        EXPECT_NE(mapped_bytes, layout.operandBytes());
        EXPECT_FALSE(layout.matchesStorageBytes(layout.operandBytes(), receipt_bytes));
        EXPECT_FALSE(layout.matchesStorageBytes(mapped_bytes - 1, receipt_bytes));
        EXPECT_FALSE(layout.matchesStorageBytes(mapped_bytes + 1, receipt_bytes));
        EXPECT_FALSE(layout.matchesStorageBytes(mapped_bytes, receipt_bytes - 1));
        EXPECT_FALSE(layout.matchesStorageBytes(mapped_bytes, receipt_bytes + 1));
        const auto last = layout.operandOffset(q::PooledNativeCountedMessage::ReaderResult, 1);
        EXPECT_EQ(last + sizeof(ncclLlaminarCompletionOperand), layout.operandBytes());
        EXPECT_LT(last + sizeof(ncclLlaminarCompletionOperand), mapped_bytes);
    }
}

/** @test C0 still receives a fresh native result receipt; it has no fictitious optional owner messages. */
TEST(PooledImmutableDirectedTransport, NativeCompletionC0RetainsOnlyOriginalReaderResult)
{
    using Message = q::PooledNativeCountedMessage;
    const q::PooledNativeCompletionLayout zero(false), positive(true);
    EXPECT_EQ(zero.slots(), 1u);
    EXPECT_EQ(zero.operandBytes(), 2 * sizeof(ncclLlaminarCompletionOperand));
    EXPECT_EQ(zero.receiptBytesPerParticipant(), sizeof(ncclLlaminarByteCompletion));
    EXPECT_EQ(zero.operandOffset(Message::ReaderResult, 0), 0u);
    EXPECT_EQ(zero.operandOffset(Message::ReaderResult, 1), sizeof(ncclLlaminarCompletionOperand));
    EXPECT_EQ(zero.receiptOffset(Message::ReaderResult), 0u);
    for (std::size_t message = 0; message < static_cast<std::size_t>(Message::ReaderResult); ++message)
        EXPECT_THROW(zero.slot(static_cast<Message>(message)), std::invalid_argument);
    // Capacity/reset selection never changes a nominal message identity. A
    // positive family has its separate admitted bank; it cannot alias C0 slots.
    EXPECT_EQ(positive.slot(Message::ReaderResult), 11u);
    const ncclLlaminarByteCompletion missing{};
    const ncclLlaminarCompletionAuthority original{
        reinterpret_cast<const volatile std::uint32_t*>(0x1000), reinterpret_cast<const void*>(0x2000)};
    EXPECT_FALSE(ncclLlaminarCompletionMatches(missing, 17, 3, 1, 0, 51, original, 0, 0));
}
}

namespace llaminar2::test {
/** @brief Original native quote prices one completion bank and controls, never mapped/peer payload capacity. */
TEST(PooledImmutableDirectedTransport, NativeInlineOwnersKeepExactProtocolFloorAndNoSecondExecutable)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto capacity : {0u, 1u, 64u, 512u}) {
            const DeviceId source{type, 1}, reader{type, 0};
            const std::array slots{capacity};
            const auto original = shape(source, slots, true);
            const std::array consumers{PhysicalMemoryAllocatorIdentity{0, source}, PhysicalMemoryAllocatorIdentity{0, reader}};
            const auto geometry = q::PooledImmutableReplicatedReaderGeometry::native(consumers[0], consumers, 256, 64);
            const auto actual = q::PooledImmutableDirectedTransport::completeBOM(original.physical, original.coordinator,
                host, original.resources, &geometry);
            const auto expected = q::PooledImmutableNativeCompletionStorage::memoryBOM(host,
                std::array{original.resources[0], original.resources[1]}, q::PooledNativeCompletionLayout(capacity != 0));
            EXPECT_EQ(bytes(actual), bytes(expected));
            auto floor = bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(original.physical, original.coordinator,
                host, original.resources, &geometry));
            const auto growth = bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(original.physical, original.coordinator,
                host, original.resources, &geometry));
            for (const auto& [device, value] : growth) floor[device] += value;
            EXPECT_EQ(floor, bytes(actual));
            for (const auto& line : actual.resources())
                EXPECT_EQ(line.bytes(PhysicalMemoryOwner::NativeStorageGraphExecutable), 0u);
            const auto controls = bytes(geometry.completeBOM(host, original.resources));
            EXPECT_EQ(controls.at(source), sizeof(q::replicated_reader::Control));
            EXPECT_EQ(controls.at(reader), sizeof(q::replicated_reader::Control));
            EXPECT_TRUE(!controls.contains(DeviceId::cpu()) || controls.at(DeviceId::cpu()) == 0);
            EXPECT_THROW(geometry.resultMemory(consumers[1]), std::logic_error);
            EXPECT_THROW(geometry.acknowledgementMemory(consumers[1]), std::logic_error);
            auto missing = original.resources; missing.pop_back();
            EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(original.physical, original.coordinator,
                host, missing, &geometry), std::invalid_argument);
        }
}
} // namespace llaminar2::test

#include "PooledNativeOperandReplay.inc"

namespace llaminar2::test {
/** @test Optional absence rejects matching; bound recorded views check exact identity without granting native readiness. */
TEST(PooledImmutableDirectedTransport, NativeCompletionAbsentViewRejectsMatching)
{
    EXPECT_FALSE(NativeReceiveCompletionView::notRequired().matches(17, 3, 1, 0, 51, 0, 0));
    std::uint32_t abort = 0, foreignAbort = 0;
    const int communicator = 1, foreignCommunicator = 2;
    const ncclLlaminarCompletionAuthority original{&abort, &communicator};
    ncclLlaminarByteCompletion receipt{17, 3, 1, 0, &abort, &communicator, 51,
        ncclLlaminarCompletionPhase::Completed, 0};
    const auto view = PooledNativeCollectiveReplayAccess::recorded(original).receive(&receipt);
    ASSERT_TRUE(view.valid());
    EXPECT_TRUE(view.matches(17, 3, 1, 0, 51, 0, 0));
    EXPECT_FALSE(view.matches(17, 3, 2, 0, 51, 0, 0));
    EXPECT_FALSE(view.matches(17, 3, 1, 1, 51, 0, 0));
    EXPECT_FALSE(view.matches(17, 3, 1, 0, 51, 1, 0));
    EXPECT_FALSE(view.matches(17, 3, 1, 0, 51, 0, 1));
    receipt.nativeAbortFlag = &foreignAbort;
    EXPECT_FALSE(view.matches(17, 3, 1, 0, 51, 0, 0));
    receipt.nativeAbortFlag = &abort;
    receipt.nativeCommunicator = &foreignCommunicator;
    EXPECT_FALSE(view.matches(17, 3, 1, 0, 51, 0, 0));
}
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Record the actual production metadata contract without creating a native transport or permit. */
class ReaderContractReplayStage final:public testing::MockComputeStage {
public:
    /** @brief Retain a production declaration for metadata-only graph planning.
     * @param device Participant whose dependency ledger will consume the contract.
     * @param contract Original source producer or reader metadata declaration.
     * @param boundary Whether this stage begins the next retained child. */
    ReaderContractReplayStage(DeviceId device,StageBufferContract contract,bool boundary=false)
        :MockComputeStage(ComputeStageType::COPY,"reader_contract",device),contract_(std::move(contract)),boundary_(boundary){}
    /** @return The unchanged production buffer declaration supplied at construction. */
    StageBufferContract bufferContract()const override{return contract_;}
    /** @return Whether the real graph splitter must start a retained child here. */
    bool requiresGraphCaptureSegmentBoundaryBefore()const override{return boundary_;}
private:StageBufferContract contract_;bool boundary_;
};
}
/** @test Source metadata imports its exact original frame through the production graph dependency planner. */
TEST(PooledImmutableDirectedTransport, SourceReaderFrameDeclarativeDependencyReplay){
    namespace rr=qwen4exp::replicated_reader;
    using Ledger=GraphCaptureDependencyLedger;
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}){
        auto frame=std::make_shared<INT8Tensor>(std::vector<size_t>{64});
        auto foreign=std::make_shared<INT8Tensor>(std::vector<size_t>{64});
        auto control=std::make_shared<INT32Tensor>(std::vector<size_t>{16});
        BufferArena arena;
        EXPECT_THROW(qwen4exp::detail::checkedReplicatedReaderSourceFrame(arena,frame.get()),std::logic_error);
        ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_POOLED_COORDINATOR_FRAME,frame));
        ASSERT_TRUE(arena.registerExternalBuffer(BufferId::MTP_GENERATION_CONTROL,control));
        EXPECT_EQ(qwen4exp::detail::checkedReplicatedReaderSourceFrame(arena,frame.get()),frame.get());
        EXPECT_THROW(qwen4exp::detail::checkedReplicatedReaderSourceFrame(arena,foreign.get()),std::logic_error);
        DeviceGraphExecutor executor;executor.setArena(&arena);
        int stream_token;void* const stream=&stream_token;
        for(int replay=0;replay<20;++replay)for(const auto action:{rr::Action::PrepareResultHeader,rr::Action::PrepareResult,rr::Action::ConfirmAck}){
            for(const bool old_contract:{true,false}){
                auto contract=qwen4exp::detail::replicatedReaderPhaseContract(action,BufferId::MTP_NORM_HIDDEN,{});
                if(old_contract){contract=StageBufferContract::build();contract.addPreallocatedInOut(BufferId::MTP_GENERATION_CONTROL,"INT32");}
                ComputeGraph graph;
                auto producer=StageBufferContract::build();producer.addOutput(BufferId::QFN_POOLED_COORDINATOR_FRAME,"INT8");
                graph.addNode("original_source_frame",std::make_unique<ReaderContractReplayStage>(device,std::move(producer)),device);
                graph.addNode("source_header",std::make_unique<ReaderContractReplayStage>(device,std::move(contract),true),device);
                graph.addDependency("source_header","original_source_frame");
                DeviceGraphExecutor::GraphSegmentCache plan;
                DeviceGraphCaptureController::buildCapturePlan(graph,plan,nullptr,false,false,
                    DeviceGraphExecutor::GraphReplayPlanPolicy::RequireRetainedParentComposition);
                ASSERT_EQ(plan.segments.size(),2u);
                const auto& child=plan.segments.back();
                EXPECT_EQ(child.retained_parent_input_ids.empty(),old_contract);
                auto ledger=executor.planGraphCaptureDependencies(graph,child.stage_names,device,stream,"replicated_source_frame",child.retained_parent_input_ids,
                    Ledger::ExternalInputAuthority::BindDeclaredAddressesOnly);
                ASSERT_TRUE(ledger);ledger->beginCapture();ledger->beginStage(graph.getNode("source_header")->stage.get());
                EXPECT_EQ(ledger->classifyInput(frame.get(),device,stream),old_contract?
                    Ledger::InputDisposition::StrictExternal:Ledger::InputDisposition::RetainedParentRecorded);
                EXPECT_EQ(ledger->classifyInput(foreign.get(),device,stream),Ledger::InputDisposition::StrictExternal);
                ledger->endCapture();graph.clear();
            }
        }
        for(const auto action:{rr::Action::AcceptResultHeader,rr::Action::AcknowledgeResult}){
            const auto contract=qwen4exp::detail::replicatedReaderPhaseContract(action,BufferId::MTP_NORM_HIDDEN,{});
            const auto reads=contract.allArenaReads();
            EXPECT_TRUE(std::none_of(reads.begin(),reads.end(),[](const auto& row){return row.id==BufferId::QFN_POOLED_COORDINATOR_FRAME;}));
        }
    }
}
}

namespace llaminar2::test {
/** @test Real source geometry selects local registration by allocator membership, independent of owner or reader count. */
TEST(PooledImmutableDirectedTransport, RemoteOnlySourceRootRegistration){
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const bool has_local:{false,true})for(const unsigned compute_readers:{1u,2u}){
            const PhysicalMemoryAllocatorIdentity source{0,device};
            const PhysicalMemoryAllocatorIdentity remote{0,device.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0)};
            std::vector<PhysicalMemoryAllocatorIdentity> order;
            std::vector<ImmutableRowTableLayoutOwner> declared;
            if(has_local){order.push_back(source);declared.push_back({{source,ImmutableRowStorage::VRAM},0,3,{}});}
            order.push_back(remote);declared.push_back({{remote,ImmutableRowStorage::VRAM},0,3,{}});
            std::vector<PhysicalMemoryAllocatorIdentity> consumers{source};
            if(compute_readers==2)consumers.push_back({0,device.is_cuda()?DeviceId::cuda(1):DeviceId::rocm(1)});
            const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
            const auto layout=ImmutableRowTableLayoutGeometry::checked(
                ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
                ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,declared,consumers,{},
                {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},
                {},source,64,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator);
            const auto physical=qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,source);
            const auto policy=qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},64);
            const auto coordinator=qwen4exp::PooledImmutableCoordinatorNativePlan::fromGeometry(physical,policy,701,17);
            std::vector<qwen4exp::PooledImmutableOwnerNativePlan> owners;
            for(unsigned i=0;i<order.size();++i)owners.push_back(qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(physical,policy,i,701,17));
            const ImmutableRowTableSource descriptor{.rows=0x100000100ULL,.row_elements=160,.format=GGUFTensorType::IQ4_NL};
            const auto geometry=qwen4exp::PooledImmutableSourceGeometry::fromPlans(owners,coordinator,descriptor,compute_readers);
            EXPECT_EQ(geometry.parameters().local_owner!=qwen4exp::adaptive_lookup::None,has_local);
            for(int lifetime=0;lifetime<20;++lifetime){
                BufferArena arena;unsigned local_calls=0,coordinator_calls=0,source_calls=0;
                const auto root=[&](BufferId id){return arena.registerExternalBuffer(id,
                    std::make_shared<INT8Tensor>(std::vector<size_t>{64}));};
                const auto local=[&](std::uint32_t coordinate){
                    // Checked standard indexing retains the original absent-local
                    // projection failure, without manufacturing a local owner.
                    EXPECT_EQ(owners.at(coordinate).allocator(),source);++local_calls;
                    EXPECT_TRUE(root(BufferId::QFN_POOLED_OWNER_CACHE));
                    EXPECT_TRUE(root(BufferId::QFN_POOLED_OWNER_FRAME));
                    EXPECT_TRUE(root(BufferId::QFN_POOLED_OWNER_NATIVE));
                };
                if(!has_local)EXPECT_THROW(local(geometry.parameters().local_owner),std::out_of_range);
                qwen4exp::detail::bindOriginalPooledSourceGraphRoots(geometry,local,[&]{
                    ++coordinator_calls;EXPECT_TRUE(root(BufferId::QFN_POOLED_COORDINATOR_POLICY));
                    EXPECT_TRUE(root(BufferId::QFN_POOLED_COORDINATOR_FRAME));EXPECT_TRUE(root(BufferId::QFN_POOLED_COORDINATOR_NATIVE));
                },[&]{++source_calls;EXPECT_TRUE(root(BufferId::QFN_POOLED_SOURCE_FRAME));EXPECT_TRUE(root(BufferId::QFN_POOLED_SOURCE_NATIVE));});
                EXPECT_EQ(local_calls,has_local?1u:0u);EXPECT_EQ(coordinator_calls,1u);EXPECT_EQ(source_calls,1u);
                EXPECT_EQ(arena.isRegistered(BufferId::QFN_POOLED_OWNER_FRAME),has_local);
                EXPECT_TRUE(arena.isRegistered(BufferId::QFN_POOLED_COORDINATOR_FRAME));
                EXPECT_TRUE(arena.isRegistered(BufferId::QFN_POOLED_SOURCE_FRAME));
            }
        }
}
}

namespace llaminar2::test {
/** @test Pure readers retain only native receipt/control roots; payload ownership is priced exactly once per declared owner. */
TEST(PooledImmutableDirectedTransport, NativeOwnerSubsetsKeepExactOriginalCompletionBOM)
{
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})
        for(unsigned mask=1;mask<4;++mask)
            for(int source=0;source<2;++source)
                for(const std::uint32_t capacity:{0u,1u,64u,512u}){
                    const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId(type,source)};
                    const std::array consumers{PhysicalMemoryAllocatorIdentity{0,DeviceId(type,1)},PhysicalMemoryAllocatorIdentity{0,DeviceId(type,0)}};
                    const std::array resources{PhysicalMemoryResource{0,consumers[0].device,1ull<<34,1ull<<34},
                        PhysicalMemoryResource{0,consumers[1].device,1ull<<34,1ull<<34}};
                    std::vector<PhysicalMemoryAllocatorIdentity> order;std::vector<ImmutableRowTableLayoutOwner> owners;
                    for(unsigned participant=0;participant<2;++participant)if(mask&(1u<<participant)){
                        const PhysicalMemoryAllocatorIdentity owner{0,DeviceId(type,int(participant))};order.push_back(owner);
                        owners.push_back({{owner,ImmutableRowStorage::VRAM},0,capacity,{}});
                    }
                    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
                    const auto layout=ImmutableRowTableLayoutGeometry::checked(
                        ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),ImmutableRowTableTiers::of(tiers),
                        ImmutableRowTableResidency::Adaptive,owners,{consumers.begin(),consumers.end()},{},
                        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},
                        capacity?std::optional{coordinator}:std::nullopt,256,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,std::nullopt,{},coordinator);
                    const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
                    const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},256);
                    const auto readers=q::PooledImmutableReplicatedReaderGeometry::native(coordinator,consumers,256,64);
                    const bool remotePositive=capacity && (mask&(1u<<(1-source)));
                    const auto expected=q::PooledImmutableNativeCompletionStorage::memoryBOM(host,resources,q::PooledNativeCompletionLayout(remotePositive));
                    const auto complete=q::PooledImmutableDirectedTransport::completeBOM(physical,policy,host,resources,&readers);
                    EXPECT_EQ(bytes(complete),bytes(expected));
                    auto floor=bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(physical,policy,host,resources,&readers));
                    for(const auto& [device,value]:bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(physical,policy,host,resources,&readers)))floor[device]+=value;
                    EXPECT_EQ(floor,bytes(complete));
                    EXPECT_EQ(physical.owners().size(),order.size());
                    for(const auto& line:complete.resources())EXPECT_EQ(line.bytes(PhysicalMemoryOwner::NativeStorageGraphExecutable),0u);
                }
}
}

namespace llaminar2::test {
/** @test Original edge identities never alias same-nominal operands or receipts across plural owners/readers. */
TEST(PooledImmutableDirectedTransport, AddressedCompletionSlotsAreExactForEveryOriginalStar) {
    using Message=q::PooledNativeCountedMessage;
    for(const std::size_t degree:{2u,4u,8u})for(std::size_t source=0;source<degree;++source)
        for(unsigned mask=0;mask<(1u<<degree);++mask) {
            std::array<std::uint8_t,q::kMaximumPooledNativeParticipants> flags{};
            std::size_t expectedSlots=0;
            for(std::size_t peer=0;peer<degree;++peer)if(peer!=source){flags[peer]=(mask>>peer)&1u;expectedSlots+=flags[peer]?12:1;}
            const q::PooledNativeCompletionLayout layout(degree,source,std::span(flags.data(),degree));
            std::set<std::size_t> operands;
            EXPECT_EQ(layout.operandBytes(),2*expectedSlots*sizeof(ncclLlaminarCompletionOperand));
            std::size_t receiptTotal=0;
            for(std::size_t participant=0;participant<degree;++participant) {
                std::set<std::size_t> receipts;
                for(std::size_t peer=0;peer<degree;++peer) {
                    if(participant==peer || (participant!=source && peer!=source))continue;
                    const auto edge=participant==source?peer:participant;
                    for(std::size_t message=0;message<std::size_t(Message::Count);++message){
                        const auto nominal=static_cast<Message>(message);
                        if(!flags[edge] && nominal!=Message::ReaderResult){
                            EXPECT_THROW(layout.operandOffset(nominal,int(participant),int(peer)),std::invalid_argument);continue;
                        }
                        const auto operand=layout.operandOffset(nominal,int(participant),int(peer));
                        const auto receipt=layout.receiptOffset(nominal,int(participant),int(peer));
                        EXPECT_TRUE(operands.insert(operand).second);EXPECT_TRUE(receipts.insert(receipt).second);
                        EXPECT_LE(operand+sizeof(ncclLlaminarCompletionOperand),layout.operandBytes());
                        EXPECT_LE(receipt+sizeof(ncclLlaminarByteCompletion),layout.receiptBytesPerParticipant(participant));
                    }
                }
                receiptTotal+=layout.receiptBytesPerParticipant(participant);
                EXPECT_EQ(receipts.size()*sizeof(ncclLlaminarByteCompletion),layout.receiptBytesPerParticipant(participant));
                EXPECT_TRUE(layout.matchesStorageBytes(layout.operandAllocationBytes(),layout.receiptBytesPerParticipant(participant),participant));
            }
            EXPECT_EQ(operands.size(),2*expectedSlots);EXPECT_EQ(receiptTotal,2*expectedSlots*sizeof(ncclLlaminarByteCompletion));
            if(degree>2)EXPECT_THROW(layout.operandOffset(Message::ReaderResult,0),std::invalid_argument);
            EXPECT_THROW(layout.receiptOffset(Message::ReaderResult,int(source),int(source)),std::invalid_argument);
            EXPECT_THROW(layout.operandOffset(Message::ReaderResult,-1,int(source)),std::invalid_argument);
        }
}
/** @test Only named actual endpoints may invoke native addressed bytes; unrelated members remain silent. */
TEST(PooledImmutableDirectedTransport, AddressedByteEdgesRetainExactOriginalCoordinates) {
    for(int degree:{2,4,8})for(int producer=0;producer<degree;++producer)for(int consumer=0;consumer<degree;++consumer){
        if(producer==consumer){EXPECT_THROW((NativeByteEdge{producer,consumer}),std::invalid_argument);continue;}
        const NativeByteEdge edge(producer,consumer);
        for(int participant=0;participant<degree;++participant){
            const bool member=participant==producer || participant==consumer;
            EXPECT_EQ(edge.contains(degree,participant),member);
            if(member)EXPECT_EQ(edge.peer(participant),participant==producer?consumer:producer);
            else EXPECT_THROW(edge.peer(participant),std::invalid_argument);
        }
        EXPECT_FALSE(edge.contains(degree,-1));EXPECT_FALSE(edge.contains(degree,degree));
    }
}
/** @test Mandatory/complete/growth BOMs price only each original incident completion edge at every admitted degree. */
TEST(PooledImmutableDirectedTransport, AddressedNativeBOMUsesActualOwnerEdgesEveryDegree) {
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const std::size_t degree:{2u,4u,8u})
        for(const std::size_t source:std::array<std::size_t,2>{0,degree-1})for(const unsigned pattern:{0u,1u,2u,3u})for(const std::uint32_t capacity:{0u,1u,64u,512u}){
            const PhysicalMemoryAllocatorIdentity coordinator{0,DeviceId(type,int(source))};
            std::vector<PhysicalMemoryAllocatorIdentity> consumers,order;std::vector<PhysicalMemoryResource> resources;
            std::vector<ImmutableRowTableLayoutOwner> owners;std::array<std::uint8_t,q::kMaximumPooledNativeParticipants> flags{};
            for(std::size_t participant=0;participant<degree;++participant){
                const PhysicalMemoryAllocatorIdentity owner{0,DeviceId(type,int(participant))};consumers.push_back(owner);
                resources.push_back({0,owner.device,1ull<<34,1ull<<34});
                const bool owns=pattern==0?participant==source:pattern==1?participant==(source+1)%degree:pattern==2?participant%2==0:true;
                if(owns){order.push_back(owner);owners.push_back({{owner,ImmutableRowStorage::VRAM},0,capacity,{}});
                    flags[participant]=participant!=source && capacity!=0;}
            }
            const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
            const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
                ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,consumers,{},
                {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},
                capacity?std::optional{coordinator}:std::nullopt,256,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,std::nullopt,{},coordinator);
            const auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
            const auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},256);
            const auto readers=q::PooledImmutableReplicatedReaderGeometry::native(coordinator,consumers,256,64);
            const auto expected=q::PooledImmutableNativeCompletionStorage::memoryBOM(host,resources,
                q::PooledNativeCompletionLayout(degree,source,std::span(flags.data(),degree)));
            const auto complete=q::PooledImmutableDirectedTransport::completeBOM(physical,policy,host,resources,&readers);
            EXPECT_EQ(bytes(complete),bytes(expected));
            auto floor=bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(physical,policy,host,resources,&readers));
            for(const auto&[device,value]:bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(physical,policy,host,resources,&readers)))floor[device]+=value;
            EXPECT_EQ(floor,bytes(complete));
        }
}
/** @test Native reader operation keys validate actual plural geometry and stay stable across exclusive bucket shapes. */
TEST(PooledImmutableDirectedTransport, AddressedReaderKeysCoverEveryOriginalConsumerAndBucket) {
    for(const std::uint32_t degree:{2u,4u,8u})for(std::uint32_t coordinate=0;coordinate<degree;++coordinate)
        for(const auto wire:{q::replicated_reader::Wire::Header,q::replicated_reader::Wire::Payload,q::replicated_reader::Wire::Ack}) {
            const auto first=q::replicated_reader::nativeMessageKey(wire,{coordinate,degree,16,64},q::pooled_native::Version);
            EXPECT_EQ(first,q::replicated_reader::nativeMessageKey(wire,{coordinate,degree,256,64},q::pooled_native::Version));
            EXPECT_THROW(q::replicated_reader::nativeMessageKey(wire,{degree,degree,16,64},q::pooled_native::Version),std::invalid_argument);
        }
}

}
