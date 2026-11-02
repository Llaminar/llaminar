/**
 * @file Test__OriginalPreparedBirthKey.cpp
 * @brief Actual prepared-source/pool key aliases retain original immutable metadata without a worker cache.
 *
 * Recorded creator/certificate inputs exercise the production source birth,
 * canonical generation checks and physical-reset hook. No accelerator API,
 * provider identity hashing, production catalog override or new key cache runs.
 */
#include "backends/NativeContextPreparation.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include <gtest/gtest.h>
#include <atomic>
#include <optional>
#include <latch>
#include <thread>
namespace llaminar2::test { namespace {
/** @brief Recorded worker constructed in its original exact stack/object extent. */
class KeyWorker final:public testing::MockWorkerGPUContext {
public:
    KeyWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> stack):MockWorkerGPUContext(ordinal){worker_.start(std::move(stack),[](void*){},nullptr);}
    ~KeyWorker()override{worker_.join();++destroyed;}
    static inline std::atomic<unsigned> destroyed=0;
private:PreparedGPUWorkerThread worker_;
};
/** @test Repeated aliases are the original key/control block and survive actual original reset without retaining a worker. */
TEST(OriginalPreparedBirthKey, RepeatedOriginalAliasSurvivesResetWithoutAuthorizingUse){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(54),DeviceId::rocm(54)}){
        ASSERT_EXIT(([device]{
            using Access=NativeContextPreparationReplayAccess;
            testing::installHardwareFreeGPUContextFactories();
            Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack){return Access::construct<KeyWorker>(std::move(stack),device.gpu_ordinal());});
            auto source=Access::source<KeyWorker>(device);auto& pool=GPUDeviceContextPool::instance();
            std::shared_ptr<const NativeContextBirthKey> retained;
            std::weak_ptr<IWorkerGPUContext> retiredWorker;
            std::optional<GPUContextProvenance> movedFrom;
            const auto original=Access::birthKey(source,device);
            const auto expected=*original;
            {
                auto permit=source->acquire(device);
                retiredWorker=permit.worker();
                auto provenance=pool.retainExistingContextProvenance(device);
                EXPECT_EQ(provenance.generation(),permit.generation());
                for(int replay=0;replay<20;++replay){
                    const auto key=provenance.originalBirthKey();
                    EXPECT_EQ(key.get(),original.get());EXPECT_EQ(*key,expected);
                    EXPECT_FALSE(key.owner_before(original));EXPECT_FALSE(original.owner_before(key));
                    retained=key;
                }
                auto independent=std::shared_ptr<IWorkerGPUContext>(permit.worker().get(),[](auto*){});
                EXPECT_THROW(provenance.requireWorker(independent),std::logic_error);
                EXPECT_THROW(Access::birthKeyWithSubstitutedWorker(provenance,independent),std::logic_error);
                EXPECT_THROW(Access::birthKeyFromForeignPublishedWorker(device,
                    std::make_shared<testing::MockWorkerGPUContext>(device.gpu_ordinal())),std::logic_error);
                EXPECT_EQ(provenance.originalBirthKey().get(),original.get());
                EXPECT_THROW((void)pool.beginExclusiveGenerationRetirement(device),std::logic_error);
                movedFrom.emplace(pool.retainExistingContextProvenance(device));
                auto moved=std::move(*movedFrom);
                EXPECT_THROW(movedFrom->originalBirthKey(),std::logic_error);
                EXPECT_EQ(moved.originalBirthKey().get(),original.get());
            }
            Access::admit(source);
            {auto retirement=pool.beginExclusiveGenerationRetirement(device);Access::physicalReset(retirement);}
            EXPECT_TRUE(retiredWorker.expired());EXPECT_EQ(KeyWorker::destroyed.load(),1u);
            EXPECT_EQ(retained.get(),original.get());EXPECT_EQ(*retained,expected);
            EXPECT_FALSE(retained.owner_before(original));EXPECT_FALSE(original.owner_before(retained));
            EXPECT_THROW(movedFrom->originalBirthKey(),std::logic_error);
            EXPECT_THROW(pool.retainExistingContextProvenance(device),std::logic_error);
            EXPECT_THROW(source->acquire(device),std::logic_error);
            // A successor ordinary worker is a new generation, never evidence
            // that this retained metadata can authorize another native family.
            pool.getContext(device);
            {const auto ordinary=pool.retainExistingContextProvenance(device);
             EXPECT_THROW(ordinary.originalBirthKey(),std::logic_error);}
            pool.shutdown();
            EXPECT_EQ(*retained,expected);
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
/** @test An ordinary worker has no source-born key on either backend, even with a valid exact provenance capability. */
TEST(OriginalPreparedBirthKey, OrdinaryOriginalWorkerCannotManufacturePreparedSource){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(55),DeviceId::rocm(55)}){
        ASSERT_EXIT(([device]{
            testing::installHardwareFreeGPUContextFactories();auto& pool=GPUDeviceContextPool::instance();
            pool.getContext(device);
            {const auto ordinary=pool.retainExistingContextProvenance(device);
             EXPECT_EQ(ordinary.device(),device);EXPECT_NE(ordinary.generation(),0u);
             for(int replay=0;replay<20;++replay)EXPECT_THROW(ordinary.originalBirthKey(),std::logic_error);}
            pool.shutdown();
            EXPECT_THROW(pool.retainExistingContextProvenance(device),std::logic_error);
            std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
/** @test A failed prepared birth winning the real factory gap blocks the second ordinary creator phase. */
TEST(OriginalPreparedBirthKey, PreparedFailureWinsOrdinaryFactoryGap){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto device:{DeviceId::cuda(56),DeviceId::rocm(56)}){
        SCOPED_TRACE(device.toString());
        EXPECT_EXIT(([device]{
            using Access=NativeContextPreparationReplayAccess;auto& pool=GPUDeviceContextPool::instance();
            std::atomic<unsigned> ordinaryCreators=0;
            const auto creator=[&](int ordinal){++ordinaryCreators;return std::make_unique<testing::MockWorkerGPUContext>(ordinal);};
            if(device.is_cuda())pool.registerNvidiaFactory(creator,64);else pool.registerAMDFactory(creator,64);
            Access::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack)->std::shared_ptr<IWorkerGPUContext>{
                auto prefix=Access::construct<KeyWorker>(std::move(stack),device.gpu_ordinal());
                throw std::runtime_error("recorded initialized prepared creator failure in ordinary lookup gap");
            });
            auto source=Access::source<KeyWorker>(device);
            std::latch checked{1},resume{1};std::exception_ptr precheckFailure;
            bool rejected=false,unexpectedFailure=false;
            std::thread ordinary([&]{
                try{Access::ordinaryBirthPrecheck(device);}catch(...){precheckFailure=std::current_exception();}
                checked.count_down();resume.wait();
                if(!precheckFailure){
                    try{auto result=Access::ordinaryBirthAfterFactoryAdmission(device);}
                    catch(const std::logic_error&){rejected=true;}
                    catch(...){unexpectedFailure=true;}
                }
            });
            checked.wait();EXPECT_FALSE(precheckFailure);
            EXPECT_THROW(source->acquire(device),std::runtime_error);
            resume.count_down();ordinary.join();
            EXPECT_TRUE(rejected);EXPECT_FALSE(unexpectedFailure);EXPECT_EQ(ordinaryCreators.load(),0u);
            EXPECT_THROW(pool.retainExistingContextProvenance(device),std::logic_error);
            {auto retirement=pool.beginExclusiveGenerationRetirement(device);Access::physicalReset(retirement);}
            // Validated reset removes the failed original birth; a subsequent
            // ordinary creator remains first class and still has no source key.
            pool.getContext(device);EXPECT_EQ(ordinaryCreators.load(),1u);
            {auto successor=pool.retainExistingContextProvenance(device);EXPECT_THROW(successor.originalBirthKey(),std::logic_error);}
            pool.shutdown();std::_Exit(::testing::Test::HasFailure()?2:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}

}}
