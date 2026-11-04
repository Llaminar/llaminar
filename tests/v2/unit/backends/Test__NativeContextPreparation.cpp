/**
 * @file Test__NativeContextPreparation.cpp
 * @brief Replay actual source preparation/pool/PMA lifecycle without accelerator APIs.
 *
 * Only the opaque constructor certificate and native worker body are fixtures;
 * source ownership, stack mapping, canonical admission, generation publication,
 * foreign-authority rejection and physical-reset obligation are production code.
 */
#include "backends/NativeContextPreparation.h"
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "backends/GPUDeviceContextPool.h"
#include "mocks/MockWorkerGPUContext.h"
#include "loaders/ModelContext.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::test {
/** @brief Production callers cannot register arbitrary source constructors. */
template<class Pool> concept PublicPreparedRegistration=requires(Pool& pool){
    pool.registerPreparedFactory(DeviceType::CUDA,std::function<std::shared_ptr<IWorkerGPUContext>(int,std::shared_ptr<PreparedGPUWorkerStack>)>{});
};
static_assert(!PublicPreparedRegistration<GPUDeviceContextPool>);


namespace {
/** @brief Actual mapped-stack thread within a hardware-free recorded worker body. */
class StackWorker final:public testing::MockWorkerGPUContext {
public:
    StackWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack):MockWorkerGPUContext(ordinal){
        worker_.start(std::move(stack),[](void*){},nullptr);
    }
    ~StackWorker()override{worker_.join();}
private:PreparedGPUWorkerThread worker_;
};
/** @brief Register the recorded constructor body once; no test mode exists in production lookup. */
void creator(DeviceId device){NativeContextPreparationReplayAccess::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack){
    auto worker=NativeContextPreparationReplayAccess::construct<StackWorker>(std::move(stack),device.gpu_ordinal());
    if(device.gpu_ordinal()==98)throw std::runtime_error("recorded failure after initialized worker prefix");
    return worker;
});}
/** @return Sum actual canonical opaque commitment observations, not duplicated admission state. */
std::size_t opaque(const NativeContextPreparation& source){std::size_t total=0;
    for(const auto& value:source.memory()->rankAttestation())total+=value.opaque_runtime_committed_bytes;return total;}
}
TEST(NativeContextPreparation, ActualOwnerSamePMAAdmissionAndRepeatedBorrow){
    const auto device=DeviceId::cuda(97);creator(device);auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    EXPECT_EQ(opaque(*source),12288u);
    const auto program=GPUWorkerStackProgram::originalDefaults();std::size_t exact=0;
    for(const auto& line:source->memory()->rankAttestation())exact+=line.preparation_materialized_bytes;
    EXPECT_EQ(exact,program.stackBytes()+((sizeof(StackWorker)+program.pageBytes()-1)/program.pageBytes())*program.pageBytes());
    EXPECT_THROW(GPUDeviceContextPool::instance().retainExistingPreparedContext(device,source->memory()),std::logic_error);
    {
        auto first=source->acquire(device);auto second=source->acquire(device);
        EXPECT_THROW(NativeContextPreparationReplayAccess::repeatRegistration(DeviceType::CUDA),std::logic_error);
        auto existing=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,source->memory());
        EXPECT_EQ(existing.worker(),first.worker());EXPECT_EQ(existing.generation(),first.generation());
        EXPECT_NO_THROW(existing.requireWorker(first.worker()));
        EXPECT_THROW(GPUDeviceContextPool::instance().retainExistingPreparedContext(DeviceId::rocm(97),source->memory()),std::logic_error);
        auto aliasPMA=std::shared_ptr<PhysicalMemoryAuthority>(source->memory().get(),[](auto*){});
        EXPECT_THROW(GPUDeviceContextPool::instance().retainExistingPreparedContext(device,aliasPMA),std::logic_error);
        auto aliasWorker=std::shared_ptr<IWorkerGPUContext>(first.worker().get(),[](auto*){});
        EXPECT_THROW(existing.requireWorker(aliasWorker),std::logic_error);
        EXPECT_THROW(existing.requireWorker(std::make_shared<testing::MockWorkerGPUContext>(97)),std::logic_error);
        EXPECT_THROW((void)GPUDeviceContextPool::instance().retireExclusiveGeneration(device),std::logic_error);
        EXPECT_EQ(first.worker(),second.worker());EXPECT_EQ(first.memory(),source->memory());EXPECT_NE(first.generation(),0u);
        EXPECT_THROW(GPUDeviceContextPool::instance().requireOrdinaryContextStackGrowth(device,source->memory()),std::runtime_error);
        auto snapshot=source->closeForPlanning();auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(snapshot.retainedPreparationBOM());
        auto final=source->admit(std::move(snapshot),certificate);EXPECT_EQ(first.memory(),source->memory());
        source->memory()->requireRankCommitted();
        // Upper limits remain commitments; reporting them as actual physical bytes would lie.
        EXPECT_FALSE(source->memory()->rankComplete());
        final->sealForExecution();
    }
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
    EXPECT_EQ(opaque(*source),12288u);NativeContextPreparationReplayAccess::physicalReset(scope);EXPECT_EQ(opaque(*source),0u);
}
TEST(NativeContextPreparation, ForeignAuthorityCannotAdoptExistingOriginalWorker){
    const auto device=DeviceId::cuda(96);creator(device);auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    {auto permit=source->acquire(device);auto foreign=NativeContextPreparationReplayAccess::source<StackWorker>(device);
     EXPECT_THROW(foreign->acquire(device),std::logic_error);}
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
    NativeContextPreparationReplayAccess::physicalReset(scope);
}
TEST(NativeContextPreparation, FailedBirthRetainsOriginalObligationUntilReset){
    const auto device=DeviceId::cuda(98);creator(device);auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    EXPECT_THROW(source->acquire(device),std::runtime_error);EXPECT_EQ(opaque(*source),12288u);
    EXPECT_THROW(source->acquire(device),std::logic_error);
    EXPECT_THROW((void)GPUDeviceContextPool::instance().retainNvidiaContext(98),std::logic_error);
    EXPECT_THROW(source->closeForPlanning(),std::logic_error);
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
    EXPECT_NE(scope.receipt().retired_generation,0u);NativeContextPreparationReplayAccess::physicalReset(scope);EXPECT_EQ(opaque(*source),0u);
}
TEST(NativeContextPreparation, OriginalCapacityRejectsBeforeAnyNativeCreator){
    EXPECT_THROW(NativeContextPreparationReplayAccess::source<StackWorker>(DeviceId::cuda(95),8191),PhysicalMemoryCapacityExhausted);
}
TEST(NativeContextPreparation, ResetExclusionAndFreshOriginalGeneration){
    const auto device=DeviceId::cuda(94);creator(device);auto first=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    std::uint64_t original=0;
    {auto permit=first->acquire(device);original=permit.generation();}
    {
        auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
        auto future=NativeContextPreparationReplayAccess::source<StackWorker>(device);
        EXPECT_THROW(future->acquire(device),std::logic_error);
        NativeContextPreparationReplayAccess::physicalReset(scope);
    }
    EXPECT_THROW(first->acquire(device),std::logic_error);
    creator(device);auto second=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    {auto permit=second->acquire(device);EXPECT_GT(permit.generation(),original);}
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
    NativeContextPreparationReplayAccess::physicalReset(scope);
}

TEST(NativeContextPreparation, OrdinaryExistingProvenanceHasNoSourceOrAdmissionRequirement){
    auto& pool=GPUDeviceContextPool::instance();const auto device=DeviceId::cuda(93);
    pool.registerNvidiaFactory([](int ordinal){return std::unique_ptr<IWorkerGPUContext>(new testing::MockWorkerGPUContext(ordinal));},100);
    {
        auto worker=pool.retainNvidiaContext(93);auto proof=pool.retainExistingContextProvenance(device);
        EXPECT_EQ(proof.device(),device);EXPECT_GT(proof.generation(),0u);EXPECT_NO_THROW(proof.requireWorker(worker));
        EXPECT_THROW(pool.retainExistingContextProvenance(DeviceId::rocm(93)),std::logic_error);
        auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device);
        EXPECT_THROW(source->acquire(device),std::logic_error);
        EXPECT_THROW(pool.retainExistingPreparedContext(device,source->memory()),std::logic_error);
    }
    auto scope=pool.beginExclusiveGenerationRetirement(device);
    EXPECT_NE(scope.receipt().retired_generation,0u);
    EXPECT_THROW(pool.retainExistingContextProvenance(device),std::logic_error);
}

TEST(NativeContextPreparation, ActualModelContextRetainsOneOriginalSourceHandoff){
    auto context=ModelContext::createForTesting("recorded-source.gguf",nullptr,48,false);
    auto source=NativeContextPreparationReplayAccess::source<StackWorker>(DeviceId::cuda(92));
    EXPECT_THROW(context->bindNativeContextPreparation(nullptr),std::invalid_argument);
    context->bindNativeContextPreparation(source);
    EXPECT_EQ(context->nativeContextPreparation(),source);
    EXPECT_NO_THROW(context->bindNativeContextPreparation(source));
    auto foreign=NativeContextPreparationReplayAccess::source<StackWorker>(DeviceId::cuda(92));
    EXPECT_THROW(context->bindNativeContextPreparation(foreign),std::logic_error);
    EXPECT_EQ(context->nativeContextPreparation()->memory(),source->memory());
}

TEST(NativeContextPreparation, WorkerOutsideExactOriginalObjectExtentCannotPublish){
    const auto device=DeviceId::cuda(91);
    NativeContextPreparationReplayAccess::seed(device,[](std::shared_ptr<PreparedGPUWorkerStack>){
        return std::make_shared<testing::MockWorkerGPUContext>(91);
    });
    auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device);
    EXPECT_THROW(source->acquire(device),std::logic_error);
    EXPECT_THROW(GPUDeviceContextPool::instance().retainExistingContextProvenance(device),std::logic_error);
    EXPECT_EQ(opaque(*source),12288u);
    auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
    NativeContextPreparationReplayAccess::physicalReset(scope);
}

}

namespace llaminar2::test {
namespace {
/** @brief Original cold world2 topology; CPU NUMA coordinates are not CPU ordinals. */
struct GlobalPreparationFixture {
    OrchestrationConfig config;
    ClusterInventory inventory;
    MoEExpertOverlayExecutionPlan execution;
    MoEOverlayCapacityAdmissionPolicy policy;
    /**
     * @brief Resolve original declared roles against recorded per-rank inventory.
     * @param device Backend/ordinal of the sole locally created worker.
     * @param joint Whether both original sockets own expert tensor fragments.
     * @param include_remote_gpu Whether rank one declares a separate same-ordinal GPU.
     */
    explicit GlobalPreparationFixture(DeviceId device,bool joint=true,bool include_remote_gpu=true) {
        auto plan=std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled=true;plan->topology=RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain=plan->base_model_domain=plan->shared_expert_domain="continuation";
        RoutedExpertDomain gpu;gpu.name="continuation";gpu.scope=ExecutionDomainScope::RANK_LOCAL;
        gpu.owner_rank=0;gpu.backend=device.is_cuda()?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        gpu.participants={GlobalDeviceAddress::fromLocalDeviceId(device)};
        gpu.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        RoutedExpertDomain cpu;cpu.name="cpu";cpu.scope=ExecutionDomainScope::NODE_LOCAL;
        cpu.owner_rank=-1;cpu.backend=CollectiveBackendType::UPI;
        cpu.participants=joint?std::vector{GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1)}:
            std::vector{GlobalDeviceAddress::cpu(1)};
        cpu.world_ranks=joint?std::vector{0,1}:std::vector{1};
        cpu.routed_compute_policy=joint?RoutedExpertComputePolicy::TensorSharded:RoutedExpertComputePolicy::Apportioned;
        plan->domains={gpu,cpu};
        plan->routed_tiers={{.name="gpu",.domain="continuation",.priority=0,.max_experts_per_layer=8},
            {.name="cpu",.domain="cpu",.priority=2,.max_experts_per_layer=8,.fallback=true}};
        if(joint && include_remote_gpu) {
            auto remote=gpu;remote.name="remote_gpu";remote.owner_rank=1;plan->domains.push_back(remote);
            plan->routed_tiers.insert(plan->routed_tiers.begin()+1,
                {.name="remote_gpu",.domain="remote_gpu",.priority=1,.max_experts_per_layer=8});
        }
        config.moe_routed_expert_plan=plan;config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        inventory.world_size=joint?2:3;inventory.ranks.resize(inventory.world_size);
        for(int rank=0;rank<inventory.world_size;++rank) {
            auto& observed=inventory.ranks[rank];observed.rank=rank;observed.node_id=0;
            observed.cpu.memory_bytes=128u<<20;observed.cpu.free_memory_bytes=(120u-rank)<<20;
            observed.cpu.numa_node=rank;
            DeviceInfo gpuInfo;gpuInfo.type=device.type;gpuInfo.local_device_id=device.ordinal;
            gpuInfo.memory_bytes=64u<<20;gpuInfo.free_memory_bytes=(60u-rank)<<20;
            observed.gpus.push_back(gpuInfo); // Relay visibility never creates ownership.
        }
        execution=resolveMoEExpertOverlayExecutionPlan(plan,{.current_world_rank=0,.world_size=inventory.world_size});
    }
    /** @return Actual production projection from these immutable declarations. */
    std::vector<PhysicalMemoryResource> resources()const {
        return originalMoEOverlayNativePreparationResources(execution,inventory,config,policy,{});
    }
};
}
/** @test Global original registry survives local-only worker birth and exact final admission. */
TEST(NativeContextPreparation, TopologyRegistryPreservesRemoteSameOrdinalAndExactCertificate) {
    for(const auto device:{DeviceId::cuda(89),DeviceId::rocm(89)})
    for(bool remote:{false,true}) {
        GlobalPreparationFixture fixture(device,true,remote);const auto resources=fixture.resources();
        ASSERT_EQ(resources.size(),remote?4u:3u);
        int created=0;
        NativeContextPreparationReplayAccess::seed(device,[&created,device](std::shared_ptr<PreparedGPUWorkerStack> stack) {
            ++created;
            return NativeContextPreparationReplayAccess::construct<StackWorker>(std::move(stack),device.gpu_ordinal());
        });
        auto source=NativeContextPreparationReplayAccess::source<StackWorker>(device,resources);
        {
            auto local=source->acquire(device);EXPECT_EQ(created,1);
            auto snapshot=source->closeForPlanning();ASSERT_EQ(snapshot.initialObservations().size(),resources.size());
            const auto& retained=snapshot.retainedPreparationBOM();
            const auto* localGPU=retained.find({0,device});const auto* remoteGPU=retained.find({1,device});
            ASSERT_NE(localGPU,nullptr);
            EXPECT_GT(localGPU->bytes(PhysicalMemoryOwner::NativeExecutionContext),0u);
            if(remote) {
                ASSERT_NE(remoteGPU,nullptr);EXPECT_EQ(remoteGPU->totalBytes(),0u);
                EXPECT_NE(localGPU->resource().admission_available_bytes,remoteGPU->resource().admission_available_bytes);
            } else EXPECT_EQ(remoteGPU,nullptr);
            PhysicalMemoryPlanBuilder dropped,changed,extra(retained);
            for(const auto& bom:retained.resources()) {
                if(!(bom.resource().world_rank==1 && bom.resource().device.is_cpu()))dropped.add(bom);
                if(bom.resource().world_rank==1 && bom.resource().device.is_cpu()) {
                    auto different=bom.resource();--different.admission_available_bytes;
                    changed.add(PhysicalMemoryBOMBuilder(different).build());
                } else changed.add(bom);
            }
            extra.add(PhysicalMemoryBOMBuilder({2,DeviceId::cpu(),128u<<20,120u<<20}).build());
            for(const auto& bad:{dropped.build(),changed.build(),extra.build()}) {
                EXPECT_THROW(source->admit(std::move(snapshot),std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(bad)),std::invalid_argument);
                EXPECT_TRUE(snapshot.valid());
                EXPECT_EQ(source->memory()->phase(),PhysicalMemoryAuthorityPhase::Preparing);
            }
            auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(retained);
            auto final=source->admit(std::move(snapshot),certificate);
            EXPECT_EQ(source->memory()->admission(),certificate);
            EXPECT_EQ(local.memory(),source->memory());
            final->sealForExecution();
        }
        auto retirement=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
        NativeContextPreparationReplayAccess::physicalReset(retirement);
        EXPECT_EQ(opaque(*source),0u);
    }
}
/** @test Unowned relay inventory contributes no resource; malformed ownership cannot extend the registry. */
TEST(NativeContextPreparation, OriginalRegistryExcludesInactiveRelayAndRejectsForeignOwnership) {
    GlobalPreparationFixture fixture(DeviceId::cuda(89),false);
    const auto resources=fixture.resources();ASSERT_EQ(resources.size(),3u);
    for(const auto& resource:resources)EXPECT_NE(resource.world_rank,2);
    EXPECT_EQ(std::count_if(resources.begin(),resources.end(),[](const auto& r){return r.world_rank==1 && r.device.is_gpu();}),0);
    auto missing=fixture;missing.inventory.ranks[0].gpus.clear();
    EXPECT_THROW(missing.resources(),std::invalid_argument);
    auto duplicate=fixture;duplicate.execution.rank_plans[2].world_rank=1;
    EXPECT_THROW(duplicate.resources(),std::invalid_argument);
}
}
