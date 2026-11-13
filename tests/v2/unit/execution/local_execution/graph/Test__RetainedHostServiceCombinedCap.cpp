/** @file Test__RetainedHostServiceCombinedCap.cpp
 * @brief PMA funds planned mixed-role workers; an unplanned creator cannot borrow nonexistent exact bytes.
 */
#include "fixtures/execution/RetainedHostServiceFixture.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <gtest/gtest.h>
namespace llaminar2::test {
TEST(RetainedHostServiceCombinedCap, PlannedPopulationBeyondSixteenAndUnfundedExtraRejects){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    ASSERT_EXIT(([]{
        std::vector<RetainedHostServiceFixture> fixtures;
        for(int i=0;i<18;++i)fixtures.push_back(RetainedHostServiceFixture::recorded(i%2?RetainedHostServiceRole::ExecutorParentTicket:RetainedHostServiceRole::ImmutableLookupIO,{0,DeviceId::cuda(i)}));
        const PhysicalMemoryResource cpu{0,DeviceId::cpu(),512u<<20,512u<<20};
        PhysicalMemoryPlanBuilder plan;for(int i=0;i<17;++i){const auto bom=fixtures[i].bom(cpu);for(const auto& line:bom.resources())plan.add(line);}
        auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        std::vector<std::unique_ptr<PreparedRetainedHostService>> workers;
        for(int i=0;i<17;++i)workers.push_back(PreparedRetainedHostService::materialize(fixtures[i].take(memory)));
        EXPECT_EQ(workers.size(),17u);
        const auto exact=memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation);
        EXPECT_THROW(PreparedRetainedHostService::materialize(fixtures[17].take(memory)),std::logic_error);
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),exact);
        workers.clear();
        EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryMaterializationKind::NewAllocation),0u);
        EXPECT_EQ(memory->reservedBytes(DeviceId::cpu(),PhysicalMemoryOwner::RetainedHostServiceRuntime),17u*4096);
        GPUDeviceProcessLifetime::retireActive();
        std::_Exit(::testing::Test::HasFailure()?2:0);
    }()),::testing::ExitedWithCode(0),"");
}
}
