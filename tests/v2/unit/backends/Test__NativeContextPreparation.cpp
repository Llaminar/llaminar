/**
 * @file Test__NativeContextPreparation.cpp
 * @brief Replay actual source preparation/pool/PMA lifecycle without accelerator APIs.
 *
 * Only the opaque constructor certificate and native worker body are fixtures;
 * source ownership, stack mapping, canonical admission, generation publication,
 * foreign-authority rejection and physical-reset obligation are production code.
 */
#include "backends/NativeContextPreparation.h"
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
