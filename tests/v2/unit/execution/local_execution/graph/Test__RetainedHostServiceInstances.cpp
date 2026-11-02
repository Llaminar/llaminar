/** @file Test__RetainedHostServiceInstances.cpp
 * @brief Actual canonical PMA instance leases reject duplicate bindings/models without another byte ledger.
 *
 * Recorded metadata selects no native certificate. Threaded worker controls run
 * in owned children; metadata-only controls create no pthread/GPU resources.
 */
#include "fixtures/execution/RetainedHostServiceReplayAccess.h"
#include "execution/local_execution/graph/RetainedParentTicketServiceWorker.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "loaders/ModelContext.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include <gtest/gtest.h>
#include <array>
#include <atomic>
#include <barrier>
#include <optional>
#include <thread>
namespace llaminar2::test {
namespace {
/** @brief One original two-role admission, keeping aggregate quota large enough to expose stealing. */
struct InstanceFixture {
    RetainedHostServiceProgram io=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO);
    RetainedHostServiceProgram ticket=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ExecutorParentTicket);
    RetainedHostServiceInstance ioKey{0,DeviceId::cuda(0),RetainedHostServiceRole::ImmutableLookupIO};
    RetainedHostServiceInstance ticketKey{0,DeviceId::cuda(0),RetainedHostServiceRole::ExecutorParentTicket};
    std::array<RetainedHostServiceInstance,2> selected{ioKey,ticketKey};
    std::shared_ptr<const RetainedHostServiceDeclarations> declarations;
    std::shared_ptr<PhysicalMemoryAuthority> memory;
    /** @brief Admit both role contributions once through the actual PMA. */
    InstanceFixture(){
        const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
        PhysicalMemoryPlanBuilder plan(io.bom(cpu));const auto second=ticket.bom(cpu);for(const auto& row:second.resources())plan.add(row);
        memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        RetainedHostServiceDeclarations::Builder builder(0);builder.add(ioKey,io).add(ticketKey,ticket);declarations=std::move(builder).freeze();
    }
    /** @return A separate metadata binding; original instance use is still centralized. */
    std::unique_ptr<RetainedHostServiceBinding> bind(){return declarations->bind(memory,selected);}
};
}
/** @test Existing stack-only byte accounting does not require service shared ownership. */
TEST(RetainedHostServiceInstances, StackAccountingAuthorityRemainsValid){
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),1u<<20,1u<<20};
    auto proof=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(PhysicalMemoryPlanBuilder().add(cpu,PhysicalMemoryOwner::RetainedHostServiceStorage,4096).build());
    PhysicalMemoryAuthority memory(proof,0);
    {auto claim=memory.claimNewAllocation(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,4096);
     EXPECT_EQ(memory.claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),4096u);}
    EXPECT_EQ(memory.claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test A second binding cannot spend the other role's aggregate budget on the same instance. */
TEST(RetainedHostServiceInstances, DuplicateBindingCannotStealDistinctRoleBudget){
    InstanceFixture f;auto first=f.bind(),second=f.bind();
    {auto token=first->take(f.ioKey);first.reset();EXPECT_THROW(second->take(f.ioKey),std::logic_error);}
    auto afterClose=f.bind();EXPECT_THROW(afterClose->take(f.ioKey),std::logic_error);
    auto distinct=f.bind();EXPECT_NO_THROW(distinct->take(f.ticketKey));
}
/** @test Separate actual ModelContexts share the same original instance lifecycle. */
TEST(RetainedHostServiceInstances, TwoActualModelContextsCannotConsumeOneInstance){
    InstanceFixture f;
    auto first=ModelContext::createForTesting("recorded-host-instance-a.gguf",nullptr,1,false);
    auto second=ModelContext::createForTesting("recorded-host-instance-b.gguf",nullptr,1,false);
    first->bindRetainedHostServices(f.declarations,f.memory,f.selected);
    second->bindRetainedHostServices(f.declarations,f.memory,f.selected);
    auto token=first->takeRetainedHostService({0,f.ioKey.consumer},f.ioKey.role,f.memory);first.reset();
    EXPECT_THROW(second->takeRetainedHostService({0,f.ioKey.consumer},f.ioKey.role,f.memory),std::logic_error);
    EXPECT_NO_THROW(second->takeRetainedHostService({0,f.ticketKey.consumer},f.ticketKey.role,f.memory));
}
/** @test Concurrent separate bindings have exactly one winner before any native resource exists. */
TEST(RetainedHostServiceInstances, ConcurrentBindingsPublishExactlyOneOriginalToken){
    InstanceFixture f;auto a=f.bind(),b=f.bind();std::barrier start(3);std::atomic<unsigned> accepted=0,rejected=0;
    std::array<std::optional<RetainedHostServiceUseToken>,2> tokens;
    const auto take=[&](RetainedHostServiceBinding& binding,std::size_t index){start.arrive_and_wait();try{tokens[index].emplace(binding.take(f.ioKey));++accepted;}catch(const std::logic_error&){++rejected;}};
    std::thread one(take,std::ref(*a),0),two(take,std::ref(*b),1);start.arrive_and_wait();one.join();two.join();
    EXPECT_EQ(accepted,1u);EXPECT_EQ(rejected,1u);
}
/** @test Failed pre-native byte claim rolls back the instance while the failed binding stays consumed. */
TEST(RetainedHostServiceInstances, FailedClaimDoesNotPoisonOrPublishInstance){
    InstanceFixture f;
    {auto all=f.memory->reserveNewAllocations(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime,8192);
     auto failed=f.bind();EXPECT_THROW(failed->take(f.ioKey),std::logic_error);
     EXPECT_THROW(failed->take(f.ioKey),std::logic_error);}
    auto retry=f.bind();EXPECT_NO_THROW(retry->take(f.ioKey));
    EXPECT_EQ(f.memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
/** @test Existing canonical make_shared ownership accepts aliases and rejects independent controlblocks first. */
TEST(RetainedHostServiceInstances, CanonicalOwnerRejectsIndependentAliasControlblocks){
    InstanceFixture f,foreign;
    std::shared_ptr<PhysicalMemoryAuthority> wrong(f.memory.get(),[](PhysicalMemoryAuthority*){});
    EXPECT_THROW(f.declarations->bind(wrong,f.selected),std::logic_error);
    std::shared_ptr<PhysicalMemoryAuthority> swapped(foreign.memory,f.memory.get());
    EXPECT_THROW(f.declarations->bind(swapped,f.selected),std::logic_error);
    auto alias=std::shared_ptr<PhysicalMemoryAuthority>(f.memory,f.memory.get());
    auto a=f.declarations->bind(alias,f.selected),b=f.bind();auto token=a->take(f.ioKey);EXPECT_THROW(b->take(f.ioKey),std::logic_error);
}
/** @test The original lease outlives a destroyed binding until the actual worker joins/unmaps. */
TEST(RetainedHostServiceInstances, WorkerOwnsInstanceAfterBindingDestruction){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        InstanceFixture f;auto binding=f.bind();auto worker=PreparedRetainedHostService::materialize(binding->take(f.ioKey));binding.reset();
        auto duplicate=f.bind();EXPECT_THROW(duplicate->take(f.ioKey),std::logic_error);
        unsigned calls=0;const auto task=worker->worker().dispatch([](void* p){++*static_cast<unsigned*>(p);return true;},&calls);
        EXPECT_TRUE(worker->worker().await(task));EXPECT_EQ(calls,1u);worker.reset();
        auto closed=f.bind();EXPECT_THROW(closed->take(f.ioKey),std::logic_error);
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test A consumed production key cannot be reopened by changing to the diagnostic policy. */
TEST(RetainedHostServiceInstances, ProductionConsumptionCannotBecomeDiagnosticReuse){
    InstanceFixture f;{auto bound=f.bind();auto token=bound->take(f.ioKey);}
    RetainedHostServiceDeclarations::Builder builder(0);
    builder.add(f.ioKey,RetainedHostServiceReplayAccess::diagnosticProgram(f.ioKey.role));
    const auto changed=std::move(builder).freeze();auto binding=changed->bind(f.memory,std::array{f.ioKey});
    EXPECT_THROW(binding->take(f.ioKey),std::logic_error);
}
/** @test Only the explicitly diagnostic arm reuses the same key after complete serial teardown. */
TEST(RetainedHostServiceInstances, DiagnosticSerialReuseNeverAllowsLiveDuplicate){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        const auto program=RetainedHostServiceReplayAccess::diagnosticProgram(RetainedHostServiceRole::ImmutableLookupIO);
        const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(program.bom(cpu)),0);
        const RetainedHostServiceInstance key{0,DeviceId::cpu(),program.role()};RetainedHostServiceDeclarations::Builder builder(0);builder.add(key,program);const auto declarations=std::move(builder).freeze();
        for(unsigned i=0;i<21;++i){
            auto first=declarations->bind(memory,std::array{key});auto worker=PreparedRetainedHostService::materialize(first->take(key));first.reset();
            auto second=declarations->bind(memory,std::array{key});EXPECT_THROW(second->take(key),std::logic_error);worker.reset();
            EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        }
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
}

namespace llaminar2::test {
/** @test Actual rank runtime projection retains the original CPU0 grouped service issuer and strict PMA identity. */
TEST(RetainedHostServiceInstances, RankProjectionRetainsOriginalGroupedSourceBinding)
{
    const auto program=RetainedHostServiceReplayAccess::program(RetainedHostServiceRole::ImmutableLookupIO);
    const RetainedHostServiceInstance key{0,DeviceId::cpu(),RetainedHostServiceRole::ImmutableLookupIO};
    RetainedHostServiceDeclarations::Builder builder(0);
    builder.add(key,program);
    auto declaration=std::move(builder).freeze();
    RuntimeConfig runtime;
    runtime.retained_host_services=declaration;
    const auto rank=RankOrchestrator::Config::fromRuntime(runtime);
    ASSERT_EQ(rank.retained_host_services,declaration);
    EXPECT_FALSE(rank.retained_host_services.owner_before(declaration));
    EXPECT_FALSE(declaration.owner_before(rank.retained_host_services));
    const PhysicalMemoryResource cpu{0,DeviceId::cpu(),128u<<20,128u<<20};
    auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(program.bom(cpu)),0);
    auto model=ModelContext::createForTesting("rank-host-source.gguf",nullptr,1,false);
    const std::array selected{key};
    model->bindRetainedHostServices(rank.retained_host_services,memory,selected);
    std::shared_ptr<PhysicalMemoryAuthority> foreign(memory.get(),[](PhysicalMemoryAuthority*){});
    EXPECT_THROW(model->takeRetainedHostService({0,DeviceId::cpu()},key.role,foreign),std::logic_error);
    auto alias=std::shared_ptr<PhysicalMemoryAuthority>(memory,memory.get());
    auto token=model->takeRetainedHostService({0,DeviceId::cpu()},key.role,alias);
    EXPECT_EQ(token.memory().get(),memory.get());
    EXPECT_FALSE(token.memory().owner_before(memory));
    EXPECT_FALSE(memory.owner_before(token.memory()));
    EXPECT_THROW(model->takeRetainedHostService({0,DeviceId::cpu()},key.role,memory),std::logic_error);
}
}
