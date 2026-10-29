/**
 * @file Test__CPUGroupedMoEStorageBacking.cpp
 * @brief Device-free PMA and lifetime replay for typed grouped CPU workspace roots.
 *
 * Complete serial manifests freeze one source-authenticated maximum before
 * binding. Backing borrowers retain physical claims through manager retirement;
 * incompatible or late larger declarations never allocate around that contract.
 */
#include "memory/CPUGroupedMoEStorageBacking.h"
#include "execution/local_execution/device/WorkspaceAllocator.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "backends/BackendManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "tensors/TensorClasses.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace
{
const DeviceId cpu1{DeviceType::CPU,1};
/** @return Exact bounded existing owner authority without backend sampling. */
std::shared_ptr<PhysicalMemoryAuthority> memory(size_t bytes)
{
    PhysicalMemoryPlanBuilder builder;
    builder.add(PhysicalMemoryResource{.world_rank=0,.device=cpu1,.total_bytes=bytes,.admission_available_bytes=bytes},
        PhysicalMemoryOwner::ExecutionWorkspace,bytes);
    return std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(builder.build()),0);
}
/** @return Canonical tiny source layout, including F>D decode coverage. */
CPUGroupedMoEStorageLayout layout(size_t rows)
{return CPUGroupedMoEStorageLayout::declare(rows,5,17,4,2,cpu1);}
/** @return Actual sole-ledger materialized owner bytes. */
size_t live(const std::shared_ptr<PhysicalMemoryAuthority> &authority)
{return authority->claimedBytes(cpu1,PhysicalMemoryOwner::ExecutionWorkspace,PhysicalMemoryMaterializationKind::NewAllocation);}
/** @brief Concrete typed consumer used through the actual graph-family allocator. */
class Consumer final:public IWorkspaceConsumer
{
public:
    explicit Consumer(size_t rows):rows_(rows){}
    WorkspaceRequirements getWorkspaceRequirements(int m,int=0,int=0)const override
    {
        WorkspaceRequirements req;
        req.cpu_grouped_moe_storage=std::make_shared<const CPUGroupedMoEStorageLayout>(layout(std::max(rows_,size_t(std::max(1,m)))));
        return req;
    }
    void bindWorkspace(DeviceWorkspaceManager *manager)override
    {manager_=manager;bound=manager?manager->cpuGroupedMoEStorage():nullptr;}
    void unbindWorkspace()override {bindWorkspace(nullptr);}
    bool hasWorkspace()const override {return manager_!=nullptr;}
    DeviceWorkspaceManager *getWorkspace()const override {return manager_;}
    std::shared_ptr<CPUGroupedMoEStorageBacking> bound;
private:size_t rows_;DeviceWorkspaceManager *manager_=nullptr;
};
/** @return Explicit exact-shape request, without implicit 4096-row legacy sizing. */
WorkspaceConsumerRequest request(Consumer &consumer,int rows)
{
    return {&consumer,cpu1,rows,0,0,WorkspaceConsumerShapePolicy::FixedDeclaredShape};
}
}
TEST(CPUGroupedMoEStorageBacking, MergeFreezesOneCompleteCompatibleMaximum)
{
    const auto small=layout(1),large=layout(4);
    EXPECT_EQ(small.merge(large),large);EXPECT_TRUE(large.contains(small));EXPECT_FALSE(small.contains(large));
    EXPECT_GE(small.route_outputs.capacity(),small.route_capacity*size_t(small.expert_intermediate));
    auto incompatible=CPUGroupedMoEStorageLayout::declare(4,5,17,4,1,cpu1);
    EXPECT_THROW(large.merge(incompatible),std::invalid_argument);
    EXPECT_THROW(large.merge(CPUGroupedMoEStorageLayout::declare(4,5,17,4,2,DeviceId::cpu())),std::invalid_argument);
    EXPECT_THROW(large.merge(CPUGroupedMoEStorageLayout::declare(4,5,17,4,2,cpu1,
        CPUStoragePlacement::local(),CPUStorageInitialization::Uninitialized)),std::invalid_argument);
    auto partial=large;partial.route_capacity=1;
    EXPECT_THROW(partial.validate(),std::invalid_argument);
}
TEST(CPUGroupedMoEStorageBacking, ActualRootsRollbackAndRetainTensorAndRouterBorrowers)
{
    const auto declared=layout(4);auto authority=memory(declared.physicalBytes());
    auto backing=std::make_shared<CPUGroupedMoEStorageBacking>(declared,authority);
    EXPECT_EQ(backing->physicalBytes(),declared.physicalBytes());EXPECT_EQ(live(authority),declared.physicalBytes());
    auto tensor=backing->gate;auto view=tensor->create_view({1,17},0);ASSERT_NE(view,nullptr);
    auto router=backing->kernel_router_q8;
    EXPECT_TRUE(backing->hasExternalOwners());backing.reset();tensor.reset();
    EXPECT_EQ(live(authority),declared.gate.physicalBytes()+declared.kernel_router_q8.physicalBytes());
    view.reset();EXPECT_EQ(live(authority),declared.kernel_router_q8.physicalBytes());
    router.reset();EXPECT_EQ(live(authority),0u);
    auto shortAuthority=memory(declared.physicalBytes()-1);
    EXPECT_THROW((CPUGroupedMoEStorageBacking(declared,shortAuthority)),std::exception);
    EXPECT_EQ(live(shortAuthority),0u);
}
TEST(CPUGroupedMoEStorageBacking, SerialPlanIncludesBundleWithoutDuplicatingNamedAllocation)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    WorkspaceRequirements first,second;
    first.buffers.push_back({"projection_scratch",256,256,true});
    first.cpu_grouped_moe_storage=std::make_shared<const CPUGroupedMoEStorageLayout>(layout(1));
    second.cpu_grouped_moe_storage=std::make_shared<const CPUGroupedMoEStorageLayout>(layout(4));
    const auto plan=DeviceWorkspaceManager::planSerialFamily({first,second});ASSERT_TRUE(plan.valid());
    ASSERT_NE(plan.cpu_grouped_moe_storage,nullptr);EXPECT_EQ(*plan.cpu_grouped_moe_storage,layout(4));
    EXPECT_EQ(plan.totalPhysicalBytes(),plan.total_bytes+layout(4).physicalBytes());
    auto authority=memory(plan.totalPhysicalBytes());
    DeviceWorkspaceManager manager(cpu1,plan.totalPhysicalBytes(),authority);
    ASSERT_TRUE(manager.allocateSerialFamily(plan));EXPECT_EQ(live(authority),plan.totalPhysicalBytes());
    EXPECT_EQ(manager.used(),plan.totalPhysicalBytes());EXPECT_EQ(manager.primaryBlockSize(),plan.total_bytes);
    EXPECT_NO_THROW(manager.requireCPUGroupedMoEStorage(first.cpu_grouped_moe_storage));
    auto larger=std::make_shared<const CPUGroupedMoEStorageLayout>(layout(8));
    EXPECT_THROW(manager.requireCPUGroupedMoEStorage(larger),std::logic_error);
    auto borrower=manager.cpuGroupedMoEStorage();manager.release();
    EXPECT_EQ(live(authority),layout(4).physicalBytes());borrower.reset();EXPECT_EQ(live(authority),0u);
}
TEST(CPUGroupedMoEStorageBacking, FrozenBundleRejectsIntroductionAndRetainedReuseWithBorrowers)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    const auto declared=layout(4);auto authority=memory(declared.physicalBytes());
    DeviceWorkspaceManager manager(cpu1,declared.physicalBytes(),authority);
    WorkspaceRequirements req;req.cpu_grouped_moe_storage=std::make_shared<const CPUGroupedMoEStorageLayout>(declared);
    ASSERT_TRUE(manager.allocate(req));
    auto borrower=manager.cpuGroupedMoEStorage();std::string error;
    EXPECT_FALSE(manager.sealPrimaryBlockForReuse(&error));EXPECT_NE(error.find("retained"),std::string::npos);
    borrower.reset();ASSERT_TRUE(manager.sealPrimaryBlockForReuse(&error))<<error;
    EXPECT_EQ(manager.retainedBackingBytes(),declared.physicalBytes());
    const auto plan=DeviceWorkspaceManager::planSerialFamily({req});ASSERT_TRUE(manager.reusePrimaryBlockForSerialFamily(plan));
    EXPECT_EQ(live(authority),declared.physicalBytes());
    manager.release();EXPECT_EQ(live(authority),0u);
    DeviceWorkspaceManager empty(cpu1,declared.physicalBytes(),authority);ASSERT_TRUE(empty.allocate({}));
    EXPECT_THROW(empty.requireCPUGroupedMoEStorage(req.cpu_grouped_moe_storage),std::logic_error);
    EXPECT_EQ(live(authority),0u);
}
TEST(CPUGroupedMoEStorageBacking, AllocatorFreezesMaximumBeforeBindingAndRejectsLateGrowth)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    auto authority=memory(layout(4).physicalBytes());WorkspaceAllocator allocator(authority);ComputeGraph graph;
    Consumer small(1),large(4);WorkspaceSizingHints hints;hints.max_seq_len=4;
    hints.graph_family_policy=WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
    ASSERT_TRUE(allocator.allocateForGraph(graph,hints,{request(small,1),request(large,4)}));
    ASSERT_NE(small.bound,nullptr);EXPECT_EQ(small.bound,large.bound);EXPECT_EQ(small.bound->layout(),layout(4));
    EXPECT_EQ(allocator.totalAllocated(),layout(4).physicalBytes());
    Consumer late(8);EXPECT_THROW(allocator.allocateForGraph(graph,hints,{request(late,8)}),std::logic_error);
    EXPECT_EQ(late.bound,nullptr);EXPECT_EQ(live(authority),layout(4).physicalBytes());
    small.bound.reset();large.bound.reset();ASSERT_TRUE(allocator.sealReusablePrimaryBlocks());
    EXPECT_EQ(allocator.retainedPrimaryBytes(cpu1),layout(4).physicalBytes());
}

TEST(CPUGroupedMoEStorageBacking, NamedAllocationFailureRollsBackUnpublishedBundle)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    WorkspaceRequirements req;req.buffers.push_back({"named",256,256,true});
    req.cpu_grouped_moe_storage=std::make_shared<const CPUGroupedMoEStorageLayout>(layout(4));
    const auto bytes=req.total_bytes_with_alignment();auto authority=memory(bytes-1);
    DeviceWorkspaceManager manager(cpu1,bytes,authority);
    // Named-block PMA rejection is reported as false by the existing manager;
    // its caller-local typed bundle must still roll back before publication.
    EXPECT_FALSE(manager.allocate(req));
    EXPECT_FALSE(manager.isAllocated());EXPECT_EQ(manager.cpuGroupedMoEStorage(),nullptr);
    EXPECT_EQ(live(authority),0u);
}
TEST(CPUGroupedMoEStorageBacking, IndependentAllocatorOwnersHaveDisjointTypedBackings)
{
    if(!hasCPUBackend())initCPUBackend(-1);
    const auto bytes=layout(4).physicalBytes();auto authority=memory(2*bytes);
    WorkspaceAllocator first(authority),second(authority);ComputeGraph a,b;Consumer left(4),right(4);
    WorkspaceSizingHints hints;hints.graph_family_policy=WorkspaceGraphFamilyPolicy::ExclusiveLifetime;
    ASSERT_TRUE(first.allocateForGraph(a,hints,{request(left,4)}));
    ASSERT_TRUE(second.allocateForGraph(b,hints,{request(right,4)}));
    ASSERT_NE(left.bound,nullptr);ASSERT_NE(right.bound,nullptr);
    EXPECT_NE(left.bound,right.bound);EXPECT_NE(left.bound->gate->raw_data(),right.bound->gate->raw_data());
    EXPECT_EQ(first.totalAllocated(),bytes);EXPECT_EQ(second.totalAllocated(),bytes);EXPECT_EQ(live(authority),2*bytes);
    first.releaseAll();EXPECT_EQ(live(authority),2*bytes);
    left.bound.reset();EXPECT_EQ(live(authority),bytes);
    second.releaseAll();right.bound.reset();EXPECT_EQ(live(authority),0u);
}

/** @test Named arithmetic cannot wrap before the checked typed-root contribution. */
TEST(CPUGroupedMoEStorageBacking, CompleteManifestRejectsNamedExtentOverflowWithoutAllocation)
{
    WorkspaceRequirements requirements;
    requirements.cpu_grouped_moe_storage =
        std::make_shared<const CPUGroupedMoEStorageLayout>(layout(1));
    requirements.buffers.push_back({"huge", std::numeric_limits<size_t>::max(), 256, true});
    EXPECT_THROW(requirements.total_bytes_with_alignment(), std::overflow_error);
    requirements.buffers.front().alignment = 1;
    requirements.buffers.push_back({"additional", 1, 1, true});
    EXPECT_THROW(requirements.total_bytes(), std::overflow_error);
    EXPECT_THROW(requirements.total_bytes_with_alignment(), std::overflow_error);
}

/** @test Source width remains identity even when the physical Q8 block counts coincide. */
TEST(CPURouterQ8StorageLayout, WidthAllocatorAndCompleteCapacityAuthenticateBeforeAllocation)
{
    const auto narrow = CPURouterQ8StorageLayout::declare(3, 5, cpu1);
    const auto wider = CPURouterQ8StorageLayout::declare(3, 17, cpu1);
    EXPECT_EQ(narrow.physicalBytes(), wider.physicalBytes());
    EXPECT_THROW(narrow.merge(wider), std::invalid_argument);
    EXPECT_THROW(narrow.merge(CPURouterQ8StorageLayout::declare(3, 5, DeviceId::cpu())), std::invalid_argument);
    EXPECT_THROW(CPURouterQ8StorageLayout::declare(std::numeric_limits<size_t>::max(), 64, cpu1), std::overflow_error);
    auto incomplete = narrow;
    incomplete.d_model = 0;
    EXPECT_THROW(incomplete.validate(), std::invalid_argument);
    WorkspaceRequirements grouped, router;
    grouped.cpu_grouped_moe_storage = std::make_shared<const CPUGroupedMoEStorageLayout>(layout(1));
    router.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(narrow);
    const auto plan = DeviceWorkspaceManager::planSerialFamily({grouped, router});
    EXPECT_FALSE(plan.valid());
}

/** @test A dense-only continuation declares only the router root without phantom local experts. */
TEST(CPURouterQ8StorageLayout, ZeroLocalExpertStandInOwnsOnlyActualRouterPublication)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    class RouterOnly final : public IWorkspaceConsumer
    {
    public:
        WorkspaceRequirements getWorkspaceRequirements(int rows, int = 0, int = 0) const override
        {
            WorkspaceRequirements result;
            result.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(
                CPURouterQ8StorageLayout::declare(size_t(std::max(rows, 1)), 5, cpu1));
            return result;
        }
        void bindWorkspace(DeviceWorkspaceManager *workspace) override
        {
            manager = workspace;
            if (workspace)
            {
                workspace->requireCPURouterQ8Storage(getWorkspaceRequirements(3).cpu_router_q8_storage);
                storage = workspace->cpuRouterQ8Storage();
            }
            else storage.reset();
        }
        bool hasWorkspace() const override { return manager != nullptr; }
        DeviceWorkspaceManager *getWorkspace() const override { return manager; }
        DeviceWorkspaceManager *manager = nullptr;
        std::shared_ptr<AlignedVector<Q8_1Block>> storage;
    } consumer;
    const auto declared = CPURouterQ8StorageLayout::declare(3, 5, cpu1);
    auto authority = memory(declared.physicalBytes());
    WorkspaceAllocator allocator(authority);
    ComputeGraph graph;
    WorkspaceSizingHints hints;
    hints.graph_family_policy = WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
    const WorkspaceConsumerRequest request{&consumer, cpu1, 3, 0, 0,
        WorkspaceConsumerShapePolicy::FixedDeclaredShape};
    ASSERT_TRUE(allocator.allocateForGraph(graph, hints, {request}));
    ASSERT_NE(consumer.storage, nullptr);
    EXPECT_EQ(consumer.manager->cpuGroupedMoEStorage(), nullptr);
    EXPECT_EQ(allocator.totalAllocated(), declared.physicalBytes());
    EXPECT_EQ(live(authority), declared.physicalBytes());
    allocator.releaseAll();
    EXPECT_EQ(live(authority), declared.physicalBytes());
    consumer.storage.reset();
    EXPECT_EQ(live(authority), 0u);
}

/** @test Compatible declarations coalesce through the manager's actual grouped backing only. */
TEST(CPURouterQ8StorageLayout, GroupedAndRouterManifestShareOneBackingAndOneClaim)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    WorkspaceRequirements grouped, router;
    grouped.cpu_grouped_moe_storage = std::make_shared<const CPUGroupedMoEStorageLayout>(layout(4));
    router.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(
        CPURouterQ8StorageLayout::declare(3, 5, cpu1));
    const auto plan = DeviceWorkspaceManager::planSerialFamily({router, grouped});
    ASSERT_TRUE(plan.valid()) << plan.error;
    EXPECT_EQ(plan.totalPhysicalBytes(), layout(4).physicalBytes());
    auto authority = memory(plan.totalPhysicalBytes());
    DeviceWorkspaceManager manager(cpu1, plan.totalPhysicalBytes(), authority);
    ASSERT_TRUE(manager.allocateSerialFamily(plan));
    auto bundle = manager.cpuGroupedMoEStorage();
    auto publication = manager.cpuRouterQ8Storage();
    ASSERT_NE(bundle, nullptr);
    EXPECT_EQ(publication, bundle->kernel_router_q8);
    EXPECT_EQ(live(authority), layout(4).physicalBytes());
    EXPECT_NO_THROW(manager.requireCPURouterQ8Storage(router.cpu_router_q8_storage));
    auto wrong_width = std::make_shared<const CPURouterQ8StorageLayout>(
        CPURouterQ8StorageLayout::declare(3, 17, cpu1));
    EXPECT_THROW(manager.requireCPURouterQ8Storage(wrong_width), std::invalid_argument);
    manager.release();
    bundle.reset();
    EXPECT_EQ(live(authority), layout(4).kernel_router_q8.physicalBytes());
    publication.reset();
    EXPECT_EQ(live(authority), 0u);
}

/** @test Router-only sealing rejects borrowers, then reuses the same frozen root without growth. */
TEST(CPURouterQ8StorageLayout, StandaloneBorrowerRetirementAndReusePreserveFrozenIdentity)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    WorkspaceRequirements req;
    const auto declared = CPURouterQ8StorageLayout::declare(4, 5, cpu1);
    req.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(declared);
    auto authority = memory(declared.physicalBytes());
    DeviceWorkspaceManager manager(cpu1, declared.physicalBytes(), authority);
    ASSERT_TRUE(manager.allocate(req));
    auto borrower = manager.cpuRouterQ8Storage();
    const auto *address = borrower->data();
    std::string error;
    EXPECT_FALSE(manager.sealPrimaryBlockForReuse(&error));
    borrower.reset();
    ASSERT_TRUE(manager.sealPrimaryBlockForReuse(&error)) << error;
    const auto plan = DeviceWorkspaceManager::planSerialFamily({req});
    ASSERT_TRUE(manager.reusePrimaryBlockForSerialFamily(plan));
    EXPECT_EQ(manager.cpuRouterQ8Storage()->data(), address);
    auto larger = std::make_shared<const CPURouterQ8StorageLayout>(
        CPURouterQ8StorageLayout::declare(5, 5, cpu1));
    EXPECT_THROW(manager.requireCPURouterQ8Storage(larger), std::logic_error);
    EXPECT_EQ(live(authority), declared.physicalBytes());
    manager.release();
    EXPECT_EQ(live(authority), 0u);
}

/** @test A named-block rejection releases the unpublished standalone router claim. */
TEST(CPURouterQ8StorageLayout, NamedFailureRollsBackStandalonePublication)
{
    if (!hasCPUBackend()) initCPUBackend(-1);
    WorkspaceRequirements req;
    req.buffers.push_back({"named", 256, 256, true});
    req.cpu_router_q8_storage = std::make_shared<const CPURouterQ8StorageLayout>(
        CPURouterQ8StorageLayout::declare(4, 5, cpu1));
    const auto bytes = req.total_bytes_with_alignment();
    auto authority = memory(bytes - 1);
    DeviceWorkspaceManager manager(cpu1, bytes, authority);
    EXPECT_FALSE(manager.allocate(req));
    EXPECT_EQ(manager.cpuRouterQ8Storage(), nullptr);
    EXPECT_FALSE(manager.isAllocated());
    EXPECT_EQ(live(authority), 0u);
}
