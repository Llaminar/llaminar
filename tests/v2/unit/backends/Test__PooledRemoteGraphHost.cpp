/**
 * @file Test__PooledRemoteGraphHost.cpp
 * @brief Original-source remote host claims, structural seal and retirement through actual production ownership.
 *
 * Only native SDK results are doubled. The descriptor comes from the actual
 * directed builder, and recorded bounds pass the actual immutable selector.
 * Every process-retained population is isolated in a true threadsafe child.
 */
#include "fixtures/backends/PooledRemoteGraphHostReplayAccess.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "execution/qwen4exp/PooledImmutableDirectedOwnerProgram.h"
#include "backends/GPUGraphMemoryContract.h"
#include "loaders/ModelLoader.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
namespace {
using Access=PooledRemoteGraphHostReplayAccess;
/** @brief Recorded structural SDK result with independent creator/inspection/instantiate observations. */
struct Trace {GPUFlatGraphStructure structure;int creators=0,destroyed=0,inspections=0,instantiations=0,launches=0,preparations=0;const PhysicalMemoryAuthority* preparedMemory=nullptr;void* preparedStream=nullptr;bool failCreate=false,failInspect=false,failPrepare=false,throwPrepare=false;};
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
    Source(DeviceId device,std::uint32_t capacity,std::size_t instances=1,bool raw=false,std::size_t unitBytes=4096,std::size_t resourceBytes=128u<<20):descriptor(program(device,capacity)),selected(prepare(device,unitBytes,resourceBytes)){
        source->freezeForPlanning();PhysicalMemoryPlanBuilder plan(source->mandatoryBOM());
        const auto& gpu=source->originalResource(device);const auto& cpu=source->originalResource(DeviceId::cpu());
        for(std::size_t i=0;i<instances;++i){const auto host=Access::bom(descriptor,cpu,gpu,raw?std::nullopt:std::optional(selected),raw?nativeGraphCaptureObjectBytes(device):sizeof(Capture));for(const auto& row:host.resources())plan.add(row);}
        plan.add(gpu,PhysicalMemoryOwner::NativeGraphExecutable,instances*GPUGraphMemoryContract::reservationBytesPerExecutable(device,GPUGraphExecutableClass::General));
        source->admitFinal(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()));
    }
    NativeAuxiliaryResourceSelection prepare(DeviceId device,std::size_t unitBytes,std::size_t resourceBytes){
        NativeContextPreparationReplayAccess::seed(device,[device](auto stack){return NativeContextPreparationReplayAccess::construct<Worker>(std::move(stack),device.ordinal);});
        const std::array observed{PhysicalMemoryResource{0,DeviceId::cpu(),resourceBytes,resourceBytes},PhysicalMemoryResource{0,device,resourceBytes,resourceBytes}};
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
            EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),source.selected.hostBytes);
            source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),source.selected.hostBytes);
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
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),source.selected.hostBytes);
        source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),source.selected.hostBytes);std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(PooledRemoteGraphHost,ExplicitRawDescriptorUsesSameSealWithoutInventedHostLimit){
    GTEST_FLAG_SET(death_test_style,"threadsafe");for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})ASSERT_EXIT(([&]{
        Source source(device,3,1,true);source.exact();auto owner=Access::raw(*source.original,source.descriptor);
        EXPECT_EQ(owner->hostQualification(),GraphHostQualification::UnqualifiedDiagnostic);source.record(*owner);
        EXPECT_TRUE(Access::submit(*owner,1));
        EXPECT_TRUE(Access::observe(*owner));owner.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),0u);
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
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),4*source.selected.hostBytes);
        source.reset();
        EXPECT_EQ(source.memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeGraphExecutable),4*source.selected.hostBytes);
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
