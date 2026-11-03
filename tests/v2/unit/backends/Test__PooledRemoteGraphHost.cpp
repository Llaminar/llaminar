/**
 * @file Test__PooledRemoteGraphHost.cpp
 * @brief Original-source remote host claims, structural seal and retirement through actual production ownership.
 *
 * Only native SDK results are doubled. The descriptor comes from the actual
 * directed builder, and recorded bounds pass the actual immutable selector.
 * Every process-retained population is isolated in a true threadsafe child.
 */
#include "fixtures/backends/PooledRemoteGraphHostReplayAccess.h"
#include "fixtures/execution/RetainedGraphParticipantsReplayAccess.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "backends/GPUGraphMemoryContract.h"
#include "loaders/ModelLoader.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <limits>
#include "execution/local_execution/graph/ComputeGraph.h"
namespace llaminar2::test {
namespace {
using Access=PooledRemoteGraphHostReplayAccess;
/** @brief Recorded structural SDK result with independent creator/inspection/instantiate observations. */
struct Trace {GPUFlatGraphStructure structure;int creators=0,destroyed=0,inspections=0,instantiations=0,launches=0,preparations=0,finiteCopies=0;const PhysicalMemoryAuthority* preparedMemory=nullptr;void* preparedStream=nullptr;bool failCreate=false,failInspect=false,failPrepare=false,throwPrepare=false;};
/** @brief The fake native object occupies exactly sizeof(Capture); production state remains in the owner. */
class Capture final:public IGPUGraphCapture {
public:
    Capture(void* stream,Trace& trace):inner_(stream),trace_(trace){}
    ~Capture()override{++trace_.destroyed;}
    bool beginCapture()override{return inner_.beginCapture();}
    bool endCapture()override{return inner_.endCapture();}
    bool instantiate()override{++trace_.instantiations;return inner_.instantiate();}
    /** @brief Double the native preparation result; the actual facade authenticates original authority first. */
    bool prepareRuntimeContextStorage(const std::shared_ptr<PhysicalMemoryAuthority>& memory)override{
        ++trace_.preparations;trace_.preparedMemory=memory.get();trace_.preparedStream=executionStream();
        if(trace_.throwPrepare)throw std::runtime_error("recorded native context preparation failure");
        return !trace_.failPrepare;
    }
    bool appendFiniteBodySiblings(std::span<const IGPUGraphCapture* const> bodies)override{
        if(hasExecutable() || bodies.size()!=1 || !bodies.front() || bodies.front()==this || bodies.front()->hasExecutable())return false;
        ++trace_.finiteCopies;return true;
    }
    bool launch()override{++trace_.launches;return inner_.launch();}
    void* executionStream()const noexcept override{return inner_.executionStream();}
    GraphUpdateResult tryUpdate()override{return GraphUpdateResult::Failed;}
    bool supportsExecutableUpdate()const noexcept override{return false;}
    bool hasExecutable()const override{return inner_.hasExecutable();}
    std::size_t residentMemoryBytes()const noexcept override{return 0;}
    std::size_t nodeCount()const override{return inner_.nodeCount();}
    void reset()override{inner_.reset();}
    const char* backendName()const override{return "recorded-remote-host";}
    bool inspectFlatGraphStructure(const GPUFlatGraphLimits&,std::span<std::byte> scratch,GPUFlatGraphStructure& out,std::string*)const override{
        ++trace_.inspections;if(scratch.empty() || trace_.failInspect)return false;out=trace_.structure;return true;
    }
private:testing::MockGPUGraphCapture inner_;Trace& trace_;
};
/** @brief Actual prepared-object/stack construction with native creator calls doubled. */
class Worker final:public testing::MockWorkerGPUContext {
public:
    Worker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
    Trace trace;
    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void* stream)override{++trace.creators;if(trace.failCreate)return {};return std::make_unique<Capture>(stream,trace);}
    GPUStreamExecutionState queryStreamExecutionState(void*,std::string_view)override{return GPUStreamExecutionState::Complete;}
};
/** @return Closed actual source-authored remote program; no caller-supplied node count or bound exists. */
PooledRemoteGraphProgram program(DeviceId remote,std::uint32_t capacity){
    const auto source=remote.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
    const PhysicalMemoryAllocatorIdentity coordinator{0,source},owner{0,remote};const std::array order{coordinator,owner};
    std::vector<ImmutableRowTableLayoutOwner> owners{{{coordinator,ImmutableRowStorage::VRAM},0,0,{}},{{owner,ImmutableRowStorage::VRAM},0,capacity,{}}};
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,4096,9000),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},
        capacity?std::optional(coordinator):std::nullopt,16);
    const auto physical=qwen4exp::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    const auto policy=qwen4exp::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},32);
    return qwen4exp::PooledImmutableDirectedOwnerProgram::declare(physical,policy,1);
}
/** @brief One actual Preparing/admitted source with all original graph programs priced before creation. */
struct Source {
    std::shared_ptr<NativeContextPreparation> source;std::optional<PreparedGPUContextUsePermit> original;
    std::shared_ptr<PhysicalMemoryAuthority> memory;PooledRemoteGraphProgram descriptor;NativeAuxiliaryResourceSelection selected;
    Source(DeviceId device,std::uint32_t capacity,std::size_t instances=1,bool raw=false,std::size_t unitBytes=4096,std::size_t resourceBytes=128u<<20,bool modelBudget=false):descriptor(program(device,capacity)),selected(prepare(device,unitBytes,resourceBytes,modelBudget)){
        source->freezeForPlanning();PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
        const auto& gpu=source->originalResource(device);const auto& cpu=source->originalResource(DeviceId::cpu());
        for(std::size_t i=0;i<instances;++i){const auto host=Access::bom(descriptor,cpu,gpu,raw?std::nullopt:std::optional(selected),raw?nativeGraphCaptureObjectBytes(device):sizeof(Capture));for(const auto& row:host.resources())plan.add(row);}
        plan.add(gpu,PhysicalMemoryOwner::NativeStorageGraphExecutable,instances*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
        if (modelBudget) {
            const auto model = RetainedParticipantGraphExecutable::generalBOM(cpu, gpu);
            for (const auto& row : model.resources()) plan.add(row);
        }
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    }
    /**
     * @brief Bind one recorded source with exact independently priced model headroom.
     * @param device Original recorded native allocator and worker.
     * @param unitBytes Closed replay family host-unit allowance.
     * @param resourceBytes Existing context/storage resource headroom on each allocator.
     * @param modelBudget Add precisely the original General model BOM to each recorded resource.
     * @return Recorded original-source selection; no actual GPU resource is queried.
     * @throws std::overflow_error If the exact recorded capacity is unrepresentable.
     */
    NativeAuxiliaryResourceSelection prepare(DeviceId device,std::size_t unitBytes,std::size_t resourceBytes,bool modelBudget){
        NativeContextPreparationReplayAccess::seed(device,[device](auto stack){return NativeContextPreparationReplayAccess::construct<Worker>(std::move(stack),device.ordinal);});
        std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),resourceBytes,resourceBytes},PhysicalMemoryResource{0,device,resourceBytes,resourceBytes}};
        if(modelBudget){
            // The recorded resource has its original storage/context headroom
            // PLUS exactly one model's typed General BOM. The policy allowance
            // is unchanged; no empirical or anonymous capacity is introduced.
            const auto model=RetainedParticipantGraphExecutable::generalBOM(observed[0],observed[1]);
            for(const auto& row:model.resources())for(auto& resource:observed)if(resource.device==row.resource().device){
                if(row.totalBytes()>std::numeric_limits<std::size_t>::max()-resourceBytes)
                    throw std::overflow_error("Recorded model fixture capacity overflows");
                resource.total_bytes=resource.admission_available_bytes=resourceBytes+row.totalBytes();
            }
        }
        source=NativeContextPreparationReplayAccess::source<Worker>(device,observed);original.emplace(source->acquire(device));memory=source->memory();
        return Access::selection(*original,descriptor,unitBytes);
    }
    Worker& worker(){return static_cast<Worker&>(*original->worker());}
    auto create(){return Access::create(*original,descriptor,selected,sizeof(Capture));}
    void record(RetainedParticipantGraphExecutable& owner){worker().submitAndWait([&]{auto& capture=Access::recording(owner);
        ASSERT_TRUE(capture.beginCapture());
        ASSERT_TRUE(capture.endCapture());
        ASSERT_TRUE(capture.prepareRuntimeContextStorage(memory));
        EXPECT_TRUE(capture.instantiate());});}
    void exact(){auto& out=worker().trace.structure;out={};out.nodes=descriptor.limits().maximumNodes;out.edges=descriptor.limits().maximumEdges;out.kinds[0]=out.nodes;out.linearChain=true;}
    void reset(){const auto device=original->device();original.reset();auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextPreparationReplayAccess::physicalReset(scope);}
};
}
void recordedLifecycle(std::uint32_t capacity){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
            Source source(device,capacity);source.exact();auto owner=source.create();
            EXPECT_EQ(owner->hostQualification(),GraphHostQualification::RecordedReplay);auto& facade=Access::recording(*owner);source.record(*owner);
            for(std::uint64_t epoch=1;epoch<=20;++epoch){EXPECT_TRUE(Access::submit(*owner,epoch));
        EXPECT_THROW(Access::submit(*owner,epoch),std::logic_error);
        EXPECT_TRUE(Access::observe(*owner));}
            EXPECT_EQ(source.worker().trace.preparations,1);
            EXPECT_EQ(source.worker().trace.preparedMemory,source.memory.get());
            EXPECT_EQ(source.worker().trace.preparedStream,source.worker().defaultStream());
            // The unchanged actual pool guard rejects growth for this original
            // prepared population; accepting native preparation cannot bypass it.
            EXPECT_THROW(GPUDeviceContextPool::instance().requireOrdinaryContextStackGrowth(device,source.memory),std::runtime_error);
            EXPECT_EQ(source.worker().trace.inspections,1);
        EXPECT_EQ(source.worker().trace.instantiations,1);
        EXPECT_EQ(source.worker().trace.launches,20);
            Access::retire(*owner);
        EXPECT_THROW(facade.beginCapture(),std::logic_error);
        EXPECT_EQ(source.worker().trace.destroyed,1);
        EXPECT_THROW(Access::recording(*owner),std::logic_error);owner.reset();
            EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),source.selected.hostBytes);
            source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),source.selected.hostBytes);
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(PooledRemoteGraphHost,RecordedOriginalC0ClaimsSealReplayAndSurviveReset){recordedLifecycle(0);}
TEST(PooledRemoteGraphHost,RecordedOriginalPositiveClaimsSealReplayAndSurviveReset){recordedLifecycle(3);}
TEST(PooledRemoteGraphHost,WrongSourceAndFlatShapeRejectBeforeNativeInstantiation){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device,3,5);source.exact();for(unsigned identity=0;identity<4;++identity){auto wrong=source.selected;
            if(identity==0)++wrong.contextGeneration;if(identity==1)wrong.sourceIdentity=std::string(64,'a');
            if(identity==2)wrong.resourceSource=std::string(64,'a');if(identity==3)wrong.lifetime=NativePhysicalLifetime::ContextGeneration;
            EXPECT_THROW(Access::create(*source.original,source.descriptor,wrong,sizeof(Capture)),std::invalid_argument);
        EXPECT_EQ(source.worker().trace.creators,0);
        }
        for(unsigned fault=0;fault<5;++fault){source.exact();auto& structure=source.worker().trace.structure;
            if(fault==0)--structure.nodes;if(fault==1)--structure.edges;if(fault==2)structure.linearChain=false;if(fault==3){--structure.kinds[0];++structure.kinds[1];}if(fault==4)source.worker().trace.failInspect=true;
            auto owner=source.create();source.worker().submitAndWait([&]{auto& capture=Access::recording(*owner);
        ASSERT_TRUE(capture.beginCapture());
        ASSERT_TRUE(capture.endCapture());
        EXPECT_THROW(capture.instantiate(),std::runtime_error);});
            EXPECT_THROW(Access::submit(*owner,1),std::logic_error);owner.reset();
        }
        EXPECT_EQ(source.worker().trace.instantiations,0);
        EXPECT_EQ(source.worker().trace.destroyed,5);source.reset();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(PooledRemoteGraphHost,MissingPlannedBytesAndCreatorPrefixKeepPreciseOriginalOwnership){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device,0);
        EXPECT_THROW(RetainedParticipantGraphExecutable::create(*source.original,source.worker().defaultStream(),source.descriptor),std::logic_error);
        EXPECT_EQ(source.worker().trace.creators,0);source.worker().trace.failCreate=true;
        EXPECT_THROW(source.create(),std::runtime_error);
        EXPECT_EQ(source.worker().trace.creators,1);
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),source.selected.hostBytes);
        source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),source.selected.hostBytes);std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(PooledRemoteGraphHost,ExplicitRawDescriptorUsesSameSealWithoutInventedHostLimit){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device,3,1,true);source.exact();auto owner=Access::raw(*source.original,source.descriptor);
        EXPECT_EQ(owner->hostQualification(),GraphHostQualification::UnqualifiedDiagnostic);source.record(*owner);
        EXPECT_TRUE(Access::submit(*owner,1));
        EXPECT_TRUE(Access::observe(*owner));owner.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),0u);
        source.reset();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

TEST(PooledRemoteGraphHost,FamilySelectionKeepsExactCaptureAndAdmitsOnlyPlannedInstances){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        // Four declared instances deliberately exceed the old observed-three
        // sample. Only original PMA bytes, not an empirical count, admit them.
        Source source(device,3,4,false,std::size_t{1}<<30,std::size_t{16}<<30);
        const auto c0=program(device,0);const auto& positive=source.descriptor;
        const auto scope=*source.selected.remoteFamily;
        const auto selected=Access::familySelection(*source.original,c0,positive,scope);
        EXPECT_NE(c0.captureIdentity(),positive.captureIdentity());
        EXPECT_EQ(selected.program,NativeAuxiliaryResourceProgram::pooledRemoteOwner(c0));
        auto observedSmall=scope;observedSmall.native=c0.limits();
        const auto grown=Access::familySelection(*source.original,positive,c0,observedSmall);
        EXPECT_EQ(grown.program,NativeAuxiliaryResourceProgram::pooledRemoteOwner(positive));
        source.exact();std::vector<std::shared_ptr<RetainedParticipantGraphExecutable>> owners;
        for(unsigned i=0;i<4;++i){owners.push_back(source.create());source.record(*owners.back());}
        EXPECT_EQ(source.worker().trace.creators,4);
        EXPECT_THROW(source.create(),std::logic_error); // Unplanned fifth fails before SDK creation.
        EXPECT_EQ(source.worker().trace.creators,4);
        for(auto& owner:owners){EXPECT_TRUE(Access::submit(*owner,1));EXPECT_TRUE(Access::observe(*owner));}
        owners.clear();
        EXPECT_THROW(source.create(),std::logic_error); // Physical commitments survive handle destruction.
        EXPECT_EQ(source.worker().trace.creators,4);
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),4*source.selected.hostBytes);
        source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeStorageGraphExecutable),4*source.selected.hostBytes);
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @test Production prepare-before-instantiate preserves original PMA and terminal failure semantics. */
TEST(PooledRemoteGraphHost,FlatPreparationRejectsForeignAuthorityAndRetainsNativeFailures)
{
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device,3,3);source.exact();
        for(unsigned fault=0;fault<3;++fault){
            auto owner=source.create();source.worker().trace.failPrepare=fault==1;source.worker().trace.throwPrepare=fault==2;
            const auto before=source.worker().trace.preparations;
            source.worker().submitAndWait([&]{
                auto& capture=Access::recording(*owner);ASSERT_TRUE(capture.beginCapture());ASSERT_TRUE(capture.endCapture());
                if(fault==0){
                    // Same pointer, different ownership: a shallow alias must
                    // fail before reaching the actual native preparation call.
                    const std::shared_ptr<PhysicalMemoryAuthority> foreign(source.memory.get(),[](auto*){});
                    EXPECT_THROW(capture.prepareRuntimeContextStorage(foreign),std::invalid_argument);
                    EXPECT_EQ(source.worker().trace.preparations,before);
                }else if(fault==1)EXPECT_FALSE(capture.prepareRuntimeContextStorage(source.memory));
                else EXPECT_THROW(capture.prepareRuntimeContextStorage(source.memory),std::runtime_error);
                EXPECT_THROW(capture.prepareRuntimeContextStorage(source.memory),std::logic_error);
                EXPECT_THROW(capture.instantiate(),std::logic_error);
            });
            owner.reset();
        }
        EXPECT_EQ(source.worker().trace.preparations,2);EXPECT_EQ(source.worker().trace.instantiations,0);
        EXPECT_EQ(source.worker().trace.destroyed,3);source.reset();
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

}

namespace llaminar2::test {
/** @test A fully committed original model/follower pool cannot consume the planned storage definition. */
TEST(PooledRemoteGraphHost, StoragePurposeSurvivesFullyReservedModelAndFailedCreation)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    for (const auto device : {DeviceId::cuda(0), DeviceId::rocm(0)}) {
        ASSERT_EXIT(( [&] {
            Source source(device, 3, 2, false, 4096, 256u << 20, true);
            const auto model = RetainedParticipantGraphExecutable::generalBOM(
                source.source->originalResource(DeviceId::cpu()), source.source->originalResource(device));
            for(const auto& row:model.resources()){
                EXPECT_EQ(row.resource().total_bytes, (256u << 20)+row.totalBytes());
                EXPECT_EQ(row.resource().admission_available_bytes, row.resource().total_bytes);
            }
            std::vector<PhysicalMemoryOwnerReservation> modelReservations;
            for (const auto& row : model.resources()) {
                const auto bytes = row.bytes(PhysicalMemoryOwner::NativeGraphExecutable);
                modelReservations.push_back(source.memory->reserveNewAllocations(
                    row.resource().device, PhysicalMemoryOwner::NativeGraphExecutable, bytes));
            }
            source.worker().trace.failCreate = true;
            EXPECT_THROW(source.create(), std::runtime_error);
            source.worker().trace.failCreate = false;
            source.exact();
            auto owner = source.create();
            source.record(*owner);
            EXPECT_TRUE(Access::submit(*owner, 1));
            EXPECT_TRUE(Access::observe(*owner));
            Access::retire(*owner);
            owner.reset();
            for (const auto& row : model.resources()) {
                EXPECT_EQ(source.memory->reservedBytes(row.resource().device,
                    PhysicalMemoryOwner::NativeGraphExecutable),
                    row.bytes(PhysicalMemoryOwner::NativeGraphExecutable));
            }
            source.reset();
            std::_Exit(::testing::Test::HasFailure() ? 2 : 0);
        }()), ::testing::ExitedWithCode(0), "");
    }
}
namespace {
/** @brief Recorded source invocation checks the exact immutable seal without executing model work. */
class EmbeddedSourceInvocation final:public IRetainedGraphParticipantInvocation {
public:
    explicit EmbeddedSourceInvocation(RetainedGraphParticipantSourceIdentity original):original_(original){}
    void requireCurrent(const RetainedGraphParticipantSourceSeal& seal)const override{
        if(seal.identity()!=original_)throw std::logic_error("recorded source seal changed");
    }
    RetainedGraphParticipantSubmission enqueueOriginalParticipants()const override{
        throw std::logic_error("Graph-only attachment test must never submit source or storage");
    }
private:RetainedGraphParticipantSourceIdentity original_;
};
/** @brief Double only the original source-plan certification; owner lifetime and native facade are actual bodies. */
class EmbeddedSourcePlan final:public IRetainedGraphParticipantSourcePlan {
public:
    EmbeddedSourcePlan(PreparedGPUContextUsePermit original,ComputeGraph& graph,const IGPUGraphCapture& parent)
        :original_(std::move(original)),graph_(graph),parent_(parent){}
    const PreparedGPUContextUsePermit& originalSource()const noexcept override{return original_;}
    void authenticatePreparationStream(std::shared_ptr<const RetainedGraphParticipantPreparationStream> preparation)const override{
        preparation->requireLive(original_);
    }
    std::uint64_t captureIdentity()const noexcept override{return 177;}
    void authenticateRecordingOwner(const IGPUGraphCapture&)const override{}
    std::shared_ptr<IRetainedGraphParticipantInvocation> bindOriginalSource(const RetainedGraphParticipantSourceSeal& seal)const override{
        const auto& value=seal.identity();
        if(value.graph!=&graph_ || value.device!=original_.device() || value.original_parent!=&parent_ ||
           !parent_.hasExecutable() || value.recording_stream!=parent_.executionStream() ||
           !value.capture_generation || !value.capture_variant_signature ||
           value.topology_generation!=graph_.topologyGeneration() || graph_.retainedParticipantSourcePlan().get()!=this)
            throw std::logic_error("recorded source plan rejected foreign transaction");
        ++bindings;return std::make_shared<EmbeddedSourceInvocation>(value);
    }
    RetainedGraphParticipantDefinitionDomain definitionDomain()const noexcept override{return {original_.device(),this};}
    void requirePreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& preparation)const override{
        preparation.requireLive(original_);const auto& value=preparation.definition();
        if(value.graph!=&graph_ || value.arena!=&graph_ || value.plan.get()!=this || value.variant!=91 ||
           value.topology_generation!=graph_.topologyGeneration() || value.domain!=definitionDomain())
            throw std::logic_error("Recorded original source definition changed");
    }
    void requireCurrentPreparedSourceDefinition(const RetainedGraphParticipantPreparationStream& value)const override{requirePreparedSourceDefinition(value);}
    void requireCurrentStorageDefinition(const RetainedParticipantGraphExecutable& value)const override{requirePreparedStorageDefinition(value);}
    void requirePreparedStorageDefinition(const RetainedParticipantGraphExecutable& value)const override{
        if(&value!=storage)throw std::logic_error("Recorded source substituted its original storage owner");
    }
    void acquirePreparedStorageDefinition(const RetainedParticipantGraphExecutable& value,void* stream)const override{
        requirePreparedStorageDefinition(value);if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)
            throw std::logic_error("Recorded original setup frontier lost its parent stream");
        ++setupJoins;
    }
    const RetainedParticipantGraphExecutable* storage=nullptr;
    mutable unsigned setupJoins=0;
    mutable unsigned bindings=0;
private:PreparedGPUContextUsePermit original_;ComputeGraph& graph_;const IGPUGraphCapture& parent_;
};
/** @brief Native attachment observation preserves the single original progress authority. */
class EmbeddedProgress final:public IGraphCaptureAuxiliaryBranch {
public:
    explicit EmbeddedProgress(DeviceId device):device_(device){}
    const void* authorityIdentity()const noexcept override{return this;}
    DeviceId device()const noexcept override{return device_;}
    std::string_view name()const noexcept override{return "original-progress";}
    bool attach(IGPUGraphCapture&)noexcept override{return false;}
    bool attachWithFiniteBody(IGPUGraphCapture& parent,std::span<const IGPUGraphCapture* const> bodies)noexcept override{
        ++attachments;
        EXPECT_FALSE(parent.hasExecutable());EXPECT_EQ(bodies.size(),1u);
        if(bodies.size()!=1)return false;
        EXPECT_TRUE(bodies[0]);EXPECT_FALSE(bodies[0]->hasExecutable());
        return true;
    }
    unsigned attachments=0;
private:DeviceId device_;
};
}
/** @test Embedded original definitions seal without instantiate/enqueue and retain exact source lifetime. */
TEST(PooledRemoteGraphHost, RecordedStorageBodyAuthenticatesSourceAndRetiresAfterBorrow)
{
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto remote:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        const auto descriptor=program(remote,3);
        const auto coordinator=descriptor.geometry().coordinator.device;
        const std::array devices{coordinator,remote};
        const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),256u<<20,256u<<20},
            PhysicalMemoryResource{0,coordinator,256u<<20,256u<<20},PhysicalMemoryResource{0,remote,256u<<20,256u<<20}};
        for(const auto device:devices)NativeContextPreparationReplayAccess::seed(device,[device](auto stack){
            return NativeContextPreparationReplayAccess::construct<Worker>(std::move(stack),device.ordinal);
        });
        auto source=NativeContextPreparationReplayAccess::source<Worker>(devices,observed);
        {
            auto originalSource=source->acquire(coordinator),originalOwner=source->acquire(remote);
            auto& worker=static_cast<Worker&>(*originalOwner.worker());
            auto memory=source->memory();
            const auto selected=Access::selection(originalOwner,descriptor);
            source->freezeForPlanning();
            PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
            const auto storageBOM=Access::bom(descriptor,source->originalResource(DeviceId::cpu()),
                source->originalResource(remote),selected,sizeof(Capture));
            for(const auto& row:storageBOM.resources())plan.add(row);
            plan.add(source->originalResource(remote),PhysicalMemoryOwner::NativeStorageGraphExecutable,
                GPUGraphMemoryContract::reservationBytesPerExecutable(remote));
            source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
            auto& shape=worker.trace.structure;
            shape.nodes=descriptor.limits().maximumNodes;shape.edges=descriptor.limits().maximumEdges;
            shape.kinds[0]=shape.nodes;shape.linearChain=true;
            auto owner=Access::create(originalOwner,descriptor,selected,sizeof(Capture));
            Access::claimStorage(*owner);
            worker.submitAndWait([&]{
                auto& capture=Access::recording(*owner);
                ASSERT_TRUE(capture.beginCapture());ASSERT_TRUE(capture.endCapture());
                EXPECT_THROW(Access::sealStorage(*owner),std::logic_error); // Actual native backing is mandatory.
                ASSERT_TRUE(capture.prepareRuntimeContextStorage(memory));
                Access::sealStorage(*owner);
                EXPECT_FALSE(capture.hasExecutable());
                EXPECT_THROW(capture.instantiate(),std::logic_error);
            });
            EXPECT_EQ(worker.trace.inspections,1);EXPECT_EQ(worker.trace.instantiations,0);EXPECT_EQ(worker.trace.launches,0);
            EXPECT_THROW(Access::submit(*owner,1),std::logic_error);
            EXPECT_THROW(Access::observe(*owner),std::logic_error);
            {
                ComputeGraph sourceGraph;
                testing::MockGPUGraphCapture sourceParent(originalSource.worker()->defaultStream());
                ASSERT_TRUE(sourceParent.beginCapture());ASSERT_TRUE(sourceParent.endCapture());
                auto sourcePlan=std::make_shared<EmbeddedSourcePlan>(originalSource,sourceGraph,sourceParent);
                sourceGraph.setRetainedParticipantSourcePlan(sourcePlan);
                sourcePlan->storage=owner.get();
                auto preparation=RetainedGraphParticipantsReplayAccess::definition(originalSource,sourceParent.executionStream(),
                    {&sourceGraph,&sourceGraph,sourceGraph.topologyGeneration(),91,sourcePlan->definitionDomain(),sourcePlan});
                const auto identity=RetainedGraphParticipantSourceIdentity{.graph=&sourceGraph,.arena=&sourceGraph,
                    .device=coordinator,.recording_stream=sourceParent.executionStream(),
                    .topology_generation=sourceGraph.topologyGeneration(),.capture_variant_signature=91,
                    .capture_generation=17,.original_parent=&sourceParent};
                auto seal=RetainedGraphParticipantsReplayAccess::sourceSeal(identity);
                EXPECT_THROW(Access::storageBody(*owner,originalSource,preparation),std::invalid_argument);
                EXPECT_THROW(sourcePlan->bindOriginalSource(seal),std::logic_error); // A cold token never substitutes for executable proof.
                {
                    auto body=Access::storageBody(*owner,originalOwner,preparation);
                    EXPECT_THROW(Access::retire(*owner),std::logic_error);
                    testing::MockGPUGraphCapture parent(reinterpret_cast<void*>(0x9181));
                    ASSERT_TRUE(parent.beginCapture());ASSERT_TRUE(parent.endCapture());
                    EmbeddedProgress progress(remote),foreign(coordinator);
                    RetainedGraphParticipantsReplayAccess::bindSource(progress,preparation);RetainedGraphParticipantsReplayAccess::bindSource(foreign,preparation);
                    testing::MockGPUGraphCapture executable(reinterpret_cast<void*>(0x9182));
                    ASSERT_TRUE(executable.beginCapture());ASSERT_TRUE(executable.endCapture());ASSERT_TRUE(executable.instantiate());
                    EXPECT_THROW(body.attachTo(parent,progress),std::logic_error); // Actual original worker is required.
                    worker.submitAndWait([&]{
                        EXPECT_THROW(body.attachTo(parent,foreign),std::logic_error);
                        EXPECT_THROW(body.attachTo(executable,progress),std::logic_error);
                        ASSERT_TRUE(body.attachTo(parent,progress));
                        EXPECT_THROW(body.attachTo(parent,progress),std::logic_error);
                    });
                    EXPECT_EQ(progress.attachments,1u);EXPECT_EQ(foreign.attachments,0u);EXPECT_EQ(sourcePlan->setupJoins,1u);
                    ASSERT_TRUE(sourceParent.instantiate());EXPECT_NO_THROW(sourcePlan->bindOriginalSource(seal));
                    {
                        auto sdma=Access::storageBody(*owner,originalOwner,preparation);
                        Trace sdmaTrace;Capture sdmaParent(reinterpret_cast<void*>(0x9183),sdmaTrace);
                        ASSERT_TRUE(sdmaParent.beginCapture());ASSERT_TRUE(sdmaParent.endCapture());
                        EXPECT_THROW(sdma.attachTo(sdmaParent,nullptr),std::logic_error);
                        worker.submitAndWait([&]{ASSERT_TRUE(sdma.attachTo(sdmaParent,nullptr));
                            EXPECT_THROW(sdma.attachTo(sdmaParent,nullptr),std::logic_error);});
                        EXPECT_EQ(sdmaTrace.finiteCopies,1);EXPECT_EQ(sourcePlan->setupJoins,2u);
                        EXPECT_FALSE(sdmaParent.hasExecutable());
                        sdmaParent.reset(); // Original native definition/clone precedes its body borrow.
                    }
                    auto moved=std::move(body);
                    EXPECT_THROW(body.requireCurrent(),std::logic_error);
                    EXPECT_NO_THROW(moved.requireCurrent());
                    RetainedGraphParticipantsReplayAccess::retire(*preparation); // Mirrors original cache reset before source graph retirement.
                    EXPECT_THROW(moved.requireCurrent(),std::logic_error);
                    parent.reset(); // Destroy original clone owner before the retained definition borrow.
                }
                EXPECT_EQ(worker.trace.destroyed,0);
                sourceGraph.clear();
            }
            Access::retire(*owner);EXPECT_EQ(worker.trace.destroyed,1);owner.reset();
        }
        for(const auto device:devices){auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextPreparationReplayAccess::physicalReset(reset);}
        source->retireReleasedContexts();
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}

} // namespace llaminar2::test
