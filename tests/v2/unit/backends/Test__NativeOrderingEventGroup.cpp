/** @file Test__NativeOrderingEventGroup.cpp
 * @brief Actual closed program, bound-growth and original reservation lifecycle replay without hardware.
 */
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::test {
namespace {
template<class T>concept PublicCompletion=requires(T& group){group.completeNativeHandles();group.beginNativeAttempt();};
static_assert(!PublicCompletion<NativeOrderingEventGroup>);
/** @return Complete recorded class metadata, never a native observation/certificate. */
NativeContextBirthKey recordedKey(){NativeContextBirthKey k;k.device=DeviceId::cuda(0);k.source_program_version=1;k.visible_device_count=2;k.compute_units=80;k.native_wave_width=32;
    k.physical_uuid="recorded-uuid";k.architecture="recorded";k.runtime_version="1";k.driver_version="1";k.visibility_identity=std::string(64,'c');k.pthread_defaults_identity=std::string(64,'d');
    k.source_sha256=k.backend_sha256=k.runtime_sha256=k.driver_sha256=k.runtime_policy_sha256=std::string(64,'a');k.worker_stack_bytes=8192;k.worker_guard_bytes=4096;k.context_object_bytes=100;k.context_object_alignment=64;
    k.primary_streams=k.primary_blas_handles=k.secondary_blas_handles=1;k.dependency_events=32;return k;}
}
TEST(NativeOrderingEventGroup, ClosedProgramsMatchActualRootBitmapAndGrowth){
    const auto owner=NativeOrderingEventProgram::pooledOwner(false),full=NativeOrderingEventProgram::pooledOwner(true);
    EXPECT_EQ(owner.orderingEvents(),3u);EXPECT_EQ(full.orderingEvents(),4u);EXPECT_EQ(NativeOrderingEventProgram::pooledSource().nonemptyRootMask(),6u);
    EXPECT_NO_THROW(owner.requireRootBytes({0,10,20}));EXPECT_THROW(owner.requireRootBytes({1,10,20}),std::invalid_argument);
    const std::array devices{DeviceId::cuda(0)};const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices);
    PhysicalMemoryResource cpu{0,DeviceId::cpu(),1u<<20,1u<<20},gpu{0,devices[0],1u<<20,1u<<20};
    const auto floor=projection.bom(owner,cpu,gpu),actual=projection.bom(full,cpu,gpu),growth=projection.growthBOM(owner,full,cpu,gpu);
    EXPECT_EQ(floor.totalBytes()+growth.totalBytes(),actual.totalBytes());EXPECT_EQ(projection.growthBOM(owner,owner,cpu,gpu).totalBytes(),0u);
    EXPECT_THROW(projection.growthBOM(full,owner,cpu,gpu),std::invalid_argument);
    EXPECT_THROW(projection.growthBOM(owner,NativeOrderingEventProgram::pooledCoordinator(true),cpu,gpu),std::invalid_argument);
    // An unchanged program must still resolve its exact recorded source/program.
    const auto incomplete=NativeOrderingEventGroupReplayAccess::projection(std::span<const DeviceId>{});
    EXPECT_THROW(incomplete.growthBOM(owner,owner,cpu,gpu),std::invalid_argument);
    auto foreign=gpu;foreign.device=DeviceId::rocm(0);EXPECT_THROW(projection.growthBOM(owner,owner,cpu,foreign),std::invalid_argument);
}
TEST(NativeOrderingEventGroup, ActualSelectorBindsBackendFamilyAndFullLiveSource){
    auto key=recordedKey();const auto program=NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledOwner(false));
    const auto portable=NativeAuxiliaryResourceCatalog::portableSourceIdentity(key);const auto resource=std::string(64,'e');
    const std::array rows{detail::NativeAuxiliaryQualifiedClass{key.device.type,program,3072,6144,std::string_view("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),NativePhysicalLifetime::ProcessRuntime}};
    const auto selected=NativeOrderingEventGroupReplayAccess::select(key,1,program,resource,rows);EXPECT_EQ(selected.hostBytes,3072u);
    auto changed=key;changed.physical_uuid="other-recorded-uuid";changed.visibility_identity=std::string(64,'f');
    EXPECT_EQ(NativeAuxiliaryResourceCatalog::portableSourceIdentity(changed),portable);EXPECT_NE(NativeAuxiliaryResourceCatalog::liveSourceIdentity(changed),selected.sourceIdentity);
    EXPECT_NO_THROW(NativeOrderingEventGroupReplayAccess::select(changed,2,program,resource,rows));
    changed=key;changed.compute_units++;EXPECT_NO_THROW(NativeOrderingEventGroupReplayAccess::select(changed,1,program,resource,rows));
    changed=key;changed.native_wave_width++;EXPECT_NO_THROW(NativeOrderingEventGroupReplayAccess::select(changed,1,program,resource,rows));
    changed=key;changed.runtime_policy_sha256=std::string(64,'f');EXPECT_NO_THROW(NativeOrderingEventGroupReplayAccess::select(changed,1,program,resource,rows));
    EXPECT_THROW(NativeOrderingEventGroupReplayAccess::select(key,1,NativeAuxiliaryResourceProgram::readAheadSide(),resource,rows),std::runtime_error);
    const auto rewired=NativeOrderingEventGroupReplayAccess::select(key,1,program,std::string(64,'f'),rows);
    EXPECT_NE(rewired.resourceSource,selected.resourceSource);
}
TEST(NativeOrderingEventGroup, OriginalReservationPrefixCompletesOnlyAtValidatedReset){
    auto& pool=GPUDeviceContextPool::instance();pool.registerNvidiaFactory([](int ordinal){return std::make_unique<testing::MockWorkerGPUContext>(ordinal);},1);
    auto worker=pool.retainContext(DeviceId::cuda(0));const auto program=NativeOrderingEventProgram::pooledOwner(false);const std::array devices{DeviceId::cuda(0)};
    const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices);PhysicalMemoryResource cpu{0,DeviceId::cpu(),1u<<20,1u<<20},gpu{0,devices[0],1u<<20,1u<<20};
    auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(projection.bom(program,cpu,gpu));auto memory=std::make_shared<PhysicalMemoryAuthority>(certificate,0);
    {const auto original=pool.retainExistingContextProvenance(devices[0]);auto group=NativeOrderingEventGroupReplayAccess::claim(memory,original,program,projection);
        EXPECT_EQ(memory->reservedBytes(gpu.device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),6144u);
        NativeOrderingEventGroupReplayAccess::requireOriginal(*group,memory,original,program);
        auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory.get(),[](auto*){});EXPECT_THROW(NativeOrderingEventGroupReplayAccess::requireOriginal(*group,alias,original,program),std::logic_error);
        NativeOrderingEventGroupReplayAccess::begin(*group);NativeOrderingEventGroupReplayAccess::complete(*group);group.reset();
        EXPECT_EQ(memory->reservedBytes(gpu.device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),6144u);
        // The same original admitted population cannot be created again while
        // its physical commitment survives. Preserve the actual PMA cause.
        try {
            auto duplicate=NativeOrderingEventGroupReplayAccess::claim(memory,original,program,projection);
            FAIL()<<"Expected original ordering population exhaustion";
        } catch(const std::logic_error& failure) {
            const std::string message=failure.what();
            EXPECT_NE(message.find("program_kind=0 root_mask=6 event_count=3"),std::string::npos);
            EXPECT_NE(message.find("device=CUDA:0 context_generation="+std::to_string(original.generation())),std::string::npos);
            EXPECT_NE(message.find("allocation_device=CPU"),std::string::npos);
            EXPECT_NE(message.find("cause={Physical memory reservation exceeds admitted native_immutable_lookup_ordering_event bytes"),std::string::npos);
        }
        EXPECT_EQ(memory->reservedBytes(gpu.device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),6144u);}
    worker.reset();{auto scope=pool.beginExclusiveGenerationRetirement(devices[0]);NativeOrderingEventGroupReplayAccess::validatedReset(scope);}
    EXPECT_EQ(memory->reservedBytes(gpu.device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);EXPECT_EQ(memory->reservedBytes(cpu.device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
}
/** @test A rejected root setup before any creator releases its same-PMA claims without reset or process exit. */
TEST(NativeOrderingEventGroup, UnattemptedClaimsCancelWithoutPhysicalRetirement){
    enum class Completion { ScopeExit, ExplicitEmptyCompletion };
    auto& pool=GPUDeviceContextPool::instance();
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const auto lifetime:{NativePhysicalLifetime::ContextGeneration,NativePhysicalLifetime::ProcessRuntime})
            for(const auto completion:{Completion::ScopeExit,Completion::ExplicitEmptyCompletion}){
                testing::installHardwareFreeGPUContextFactories();
                auto worker=pool.retainContext(device);
                const auto program=NativeOrderingEventProgram::pooledOwner(true);
                const std::array devices{device};
                const auto projection=NativeOrderingEventGroupReplayAccess::projection(devices,lifetime);
                const PhysicalMemoryResource cpu{0,DeviceId::cpu(),1u<<20,1u<<20},gpu{0,device,1u<<20,1u<<20};
                auto memory=std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(projection.bom(program,cpu,gpu)),0);
                {
                    const auto original=pool.retainExistingContextProvenance(device);
                    auto group=NativeOrderingEventGroupReplayAccess::claim(memory,original,program,projection);
                    EXPECT_GT(memory->reservedBytes(device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
                    if(completion==Completion::ExplicitEmptyCompletion)NativeOrderingEventGroupReplayAccess::complete(*group);
                    group.reset();
                    EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
                    EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
                    // Capacity can be claimed again while the original worker
                    // is still alive; this cannot depend on a later reset.
                    auto retry=NativeOrderingEventGroupReplayAccess::claim(memory,original,program,projection);
                    retry.reset();
                    EXPECT_EQ(memory->reservedBytes(device,PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
                }
                worker.reset();
                auto scope=pool.beginExclusiveGenerationRetirement(device);
                NativeOrderingEventGroupReplayAccess::validatedReset(scope);
            }
}
}
