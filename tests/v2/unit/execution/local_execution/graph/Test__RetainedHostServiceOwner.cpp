/** @file Test__RetainedHostServiceOwner.cpp
 * @brief Replay actual CPU0 admission, thread progress, role concurrency and exact retirement.
 *
 * Only opaque reviewed input is recorded; allocation, original pthread defaults,
 * actual production ticket state and process lifetime use the production bodies.
 */
#include "execution/local_execution/graph/RetainedHostServiceProgram.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
#include <latch>
#include <array>
#include <atomic>
#include <pthread.h>
#include <dlfcn.h>
#include <unistd.h>
#include <thread>
#include <sched.h>
#include <limits>
namespace llaminar2::test {
namespace {
PhysicalMemoryResource cpu(){return {0,DeviceId::cpu(),128u<<20,128u<<20};}
std::shared_ptr<PhysicalMemoryAuthority> admit(const PhysicalMemoryPlan& plan){return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan),0);}
bool increment(void* n){++*static_cast<unsigned*>(n);return true;}
/** @return Legitimate original supported page growth; scoped defaults are restored before returning. */
RetainedHostServiceProgram grownOriginalIO(){
    struct Defaults final {
        pthread_attr_t saved{},changed{};
        bool restore=false;
        Defaults(){
            if(pthread_getattr_default_np(&saved))throw std::runtime_error("Read original pthread defaults failed");
            if(pthread_getattr_default_np(&changed)){pthread_attr_destroy(&saved);throw std::runtime_error("Read changed pthread defaults failed");}
        }
        ~Defaults(){if(restore&&pthread_setattr_default_np(&saved))std::terminate();pthread_attr_destroy(&changed);pthread_attr_destroy(&saved);}
    } defaults;
    const auto page=static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
    std::size_t stack=0,guard=0;
    if(pthread_attr_getstacksize(&defaults.changed,&stack)||pthread_attr_getguardsize(&defaults.changed,&guard)||
       !page||stack>std::numeric_limits<std::size_t>::max()-page||guard>std::numeric_limits<std::size_t>::max()-page)
        throw std::runtime_error("Unsupported original pthread geometry observation");
    if(pthread_attr_setstacksize(&defaults.changed,stack+page)||pthread_attr_setguardsize(&defaults.changed,guard+page))
        throw std::runtime_error("Supported changed pthread geometry rejected");
    if(pthread_setattr_default_np(&defaults.changed))throw std::runtime_error("Install scoped pthread geometry failed");
    defaults.restore=true;
    return RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
}

}
TEST(RetainedHostServiceOwner, DeclaredBothRolesUseActualCPU0ThreadAndRetainOnlyOriginalOpaque){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        const auto io=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO);
        const auto ticket=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
        PhysicalMemoryPlanBuilder builder(io.bom(cpu()));const auto ticketBOM=ticket.bom(cpu());for(const auto& b:ticketBOM.resources())builder.add(b);
        auto memory=admit(builder.build());
        const RetainedHostServiceInstance ioKey{0,DeviceId::cuda(0),RetainedHostServiceRole::ImmutableLookupIO};
        const RetainedHostServiceInstance ticketKey{0,DeviceId::cuda(0),RetainedHostServiceRole::ExecutorParentTicket};
        RetainedHostServiceDeclarations::Builder declarations(0);declarations.add(ioKey,io).add(ticketKey,ticket);
        auto declared=std::move(declarations).freeze();const std::array selected{ioKey,ticketKey};
        auto bound=declared->bind(memory,selected);
        auto first=PreparedRetainedHostService::materialize(bound->take(ioKey));
        EXPECT_THROW(bound->take(ioKey),std::logic_error);
        auto second=PreparedRetainedHostService::materialize(bound->take(ticketKey));
        EXPECT_NE(&first->worker(),&second->worker());
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),io.allocationBytes()+ticket.allocationBytes());
        unsigned value=0;for(unsigned i=0;i!=20;++i){auto accepted=first->worker().dispatch(increment,&value);EXPECT_TRUE(first->worker().await(accepted));EXPECT_THROW((void)first->worker().await(accepted),std::logic_error);}
        EXPECT_EQ(value,20u);
        struct Block{std::latch started{2},finish{1};};Block block;
        const auto wait=[](void* ptr){auto& b=*static_cast<Block*>(ptr);b.started.count_down();b.finish.wait();return true;};
        auto one=first->worker().dispatch(wait,&block);auto two=second->worker().dispatch(wait,&block);block.started.wait();
        EXPECT_THROW((void)first->worker().dispatch(increment,&value),std::logic_error);block.finish.count_down();
        EXPECT_TRUE(first->worker().await(one));EXPECT_TRUE(second->worker().await(two));
        first.reset();second.reset();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime),8192u);
        std::weak_ptr<PhysicalMemoryAuthority> original=memory;memory.reset();EXPECT_FALSE(original.expired());
        GPUDeviceProcessLifetime::retireActive();EXPECT_FALSE(original.expired());
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
TEST(RetainedHostServiceOwner, InstalledUnknownClassCannotPrice){
    const auto shape=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO).resourceDeclaration();
    const std::array absent{RetainedHostServiceClassProjection{RetainedHostServiceRole::ImmutableLookupIO,std::string(64,'a'),std::string(64,'b'),shape}};
    const auto projection=RetainedHostServiceBudgetProjection::installed(absent);
    EXPECT_THROW(projection.program(RetainedHostServiceRole::ImmutableLookupIO),std::invalid_argument);
    const auto control=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
    const auto original=grownOriginalIO();
    const auto restored=RetainedHostServiceProgram::original(RetainedHostServiceRole::ImmutableLookupIO);
    EXPECT_EQ(restored.stackBytes(),control.stackBytes());
    EXPECT_EQ(restored.guardBytes(),control.guardBytes());
    EXPECT_EQ(original.stackBytes(),control.stackBytes()+static_cast<std::size_t>(sysconf(_SC_PAGESIZE)));
    EXPECT_EQ(original.guardBytes(),control.guardBytes()+static_cast<std::size_t>(sysconf(_SC_PAGESIZE)));
    EXPECT_NO_THROW(original.bom(cpu()));
    EXPECT_EQ(original.resourceIdentity(),control.resourceIdentity());
    EXPECT_EQ(original.allocationBytes(),control.allocationBytes()+original.pageBytes());
    auto memory=admit(original.bom(cpu()));
    EXPECT_THROW(PreparedRetainedHostService::materialize(RetainedHostServiceReplayAccess::claim(memory,original)),std::logic_error);
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime),0u);
    auto foreign=cpu();foreign.device=DeviceId{DeviceType::CPU,1};
    EXPECT_THROW(original.bom(foreign),std::invalid_argument);
    EXPECT_THROW(RetainedHostServiceProgram::original(static_cast<RetainedHostServiceRole>(99)),std::invalid_argument);
}
/** @test Optional unselected roles do not consume the selected concurrency budget or need a class. */
TEST(RetainedHostServiceOwner, FrozenOptionalRolesArePricedOnlyWhenSelected){
    auto recorded=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
    auto unused=grownOriginalIO();
    auto memory=admit(recorded.bom(cpu()));
    const RetainedHostServiceInstance a{0,DeviceId::cuda(0),RetainedHostServiceRole::ExecutorParentTicket};
    const RetainedHostServiceInstance b{0,DeviceId::cuda(1),RetainedHostServiceRole::ExecutorParentTicket};
    const RetainedHostServiceInstance c{0,DeviceId::rocm(0),RetainedHostServiceRole::ExecutorParentTicket};
    const RetainedHostServiceInstance io{0,DeviceId::cuda(0),RetainedHostServiceRole::ImmutableLookupIO};
    RetainedHostServiceDeclarations::Builder builder(0);builder.add(a,recorded).add(b,recorded).add(c,recorded).add(io,unused);
    const auto declarations=std::move(builder).freeze();
    EXPECT_NO_THROW(declarations->bind(memory,std::array{a,b}));
    EXPECT_NO_THROW(declarations->bind(memory,std::array{a,b,c}));
    EXPECT_NO_THROW(declarations->bom(io,cpu()));
    EXPECT_THROW(declarations->program({1,a.consumer,a.role}),std::logic_error);
}
/** @test Explicit process defaults override inherited affinity identically to the original std::thread. */
TEST(RetainedHostServiceOwner, OriginalCallerPreservesActualDefaultAffinityAndGuard){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        pthread_attr_t saved{},changed{};ASSERT_EQ(pthread_getattr_default_np(&saved),0);ASSERT_EQ(pthread_getattr_default_np(&changed),0);
        cpu_set_t allowed{},one{};CPU_ZERO(&allowed);CPU_ZERO(&one);ASSERT_EQ(pthread_getaffinity_np(pthread_self(),sizeof(allowed),&allowed),0);
        int selected=-1;for(int i=0;i<CPU_SETSIZE;++i)if(CPU_ISSET(i,&allowed)){selected=i;break;}ASSERT_GE(selected,0);CPU_SET(selected,&one);
        ASSERT_EQ(pthread_attr_setaffinity_np(&changed,sizeof(one),&one),0);
        std::size_t oldGuard=0;ASSERT_EQ(pthread_attr_getguardsize(&changed,&oldGuard),0);
        ASSERT_EQ(pthread_attr_setguardsize(&changed,oldGuard+static_cast<std::size_t>(sysconf(_SC_PAGESIZE))),0);
        ASSERT_EQ(pthread_setattr_default_np(&changed),0);
        cpu_set_t control{};std::thread original([&]{EXPECT_EQ(pthread_getaffinity_np(pthread_self(),sizeof(control),&control),0);});original.join();
        auto program=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
        EXPECT_EQ(program.guardBytes(),oldGuard+static_cast<std::size_t>(sysconf(_SC_PAGESIZE)));
        auto memory=admit(program.bom(cpu()));auto owner=PreparedRetainedHostService::materialize(RetainedHostServiceReplayAccess::claim(memory,program));
        struct NativeStack {void* base=nullptr;std::size_t bytes=0;};NativeStack nativeStack;
        const auto stackTask=owner->worker().dispatch([](void* opaque){
            auto& out=*static_cast<NativeStack*>(opaque);pthread_attr_t attributes{};
            if(pthread_getattr_np(pthread_self(),&attributes))return false;
            const auto result=pthread_attr_getstack(&attributes,&out.base,&out.bytes);pthread_attr_destroy(&attributes);return result==0;
        },&nativeStack);
        ASSERT_TRUE(owner->worker().await(stackTask));
        EXPECT_EQ(nativeStack.bytes,program.stackBytes());
        EXPECT_EQ(static_cast<std::byte*>(nativeStack.base)+nativeStack.bytes,reinterpret_cast<std::byte*>(&owner->worker()));
        NativeStack originalStack;
        std::thread originalStackControl([&]{pthread_attr_t attributes{};EXPECT_EQ(pthread_getattr_np(pthread_self(),&attributes),0);EXPECT_EQ(pthread_attr_getstack(&attributes,&originalStack.base,&originalStack.bytes),0);pthread_attr_destroy(&attributes);});originalStackControl.join();
        // Original std::thread storage cannot satisfy the exact admitted stack/object root oracle.
        EXPECT_NE(static_cast<std::byte*>(originalStack.base)+originalStack.bytes,reinterpret_cast<std::byte*>(&owner->worker()));
        cpu_set_t actual{};auto task=owner->worker().dispatch([](void* ptr){return pthread_getaffinity_np(pthread_self(),sizeof(cpu_set_t),static_cast<cpu_set_t*>(ptr))==0;},&actual);
        EXPECT_TRUE(owner->worker().await(task));EXPECT_TRUE(CPU_EQUAL(&control,&actual));EXPECT_TRUE(CPU_EQUAL(&one,&actual));
        owner.reset();ASSERT_EQ(pthread_setattr_default_np(&saved),0);pthread_attr_destroy(&changed);pthread_attr_destroy(&saved);
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Defaults changed after claim reject before any mmap/native attempt, returning exact claims. */
TEST(RetainedHostServiceOwner, DelayedTokenRejectsChangedOriginalDefaultsBeforeMapping){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        auto program=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
        auto memory=admit(program.bom(cpu()));
        auto token=RetainedHostServiceReplayAccess::claim(memory,program);
        pthread_attr_t saved{},changed{};
        ASSERT_EQ(pthread_getattr_default_np(&saved),0);
        ASSERT_EQ(pthread_getattr_default_np(&changed),0);
        std::size_t guard=0;ASSERT_EQ(pthread_attr_getguardsize(&changed,&guard),0);
        ASSERT_EQ(pthread_attr_setguardsize(&changed,guard+static_cast<std::size_t>(sysconf(_SC_PAGESIZE))),0);
        ASSERT_EQ(pthread_setattr_default_np(&changed),0);
        EXPECT_THROW(PreparedRetainedHostService::materialize(std::move(token)),std::logic_error);
        ASSERT_EQ(pthread_setattr_default_np(&saved),0);
        pthread_attr_destroy(&changed);pthread_attr_destroy(&saved);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime),0u);
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
#ifdef LLAMINAR_HOST_SERVICE_TLS_CONTROL
/** @test Same-file unload/reload changes loader generation even when final file populations match. */
TEST(RetainedHostServiceOwner, OriginalEvidenceRejectsSameModuleReloadAndNeverRehashesRoles){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        void* module=dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL);ASSERT_NE(module,nullptr);
        const auto evidence=RetainedHostServiceProgram::originalEvidence();
        const auto hashCount=RetainedHostServiceProgram::nativeEvidenceHashAttempts();
        auto first=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ExecutorParentTicket,evidence);
        auto second=RetainedHostServiceReplayAccess::withEvidence(RetainedHostServiceRole::ImmutableLookupIO,evidence);
        EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashCount);
        auto memory=admit(first.bom(cpu()));
        auto token=RetainedHostServiceReplayAccess::claim(memory,first);
        EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashCount);
        ASSERT_EQ(dlclose(module),0);module=dlopen(LLAMINAR_HOST_SERVICE_TLS_CONTROL,RTLD_NOW|RTLD_LOCAL);ASSERT_NE(module,nullptr);
        EXPECT_THROW(PreparedRetainedHostService::materialize(std::move(token)),std::logic_error);
        EXPECT_THROW(RetainedHostServiceReplayAccess::claim(memory,second),std::logic_error);
        EXPECT_EQ(RetainedHostServiceProgram::nativeEvidenceHashAttempts(),hashCount);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        ASSERT_EQ(dlclose(module),0);
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
#endif

}
