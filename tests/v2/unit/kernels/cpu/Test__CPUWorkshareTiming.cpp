/**
 * @file Test__CPUWorkshareTiming.cpp
 * @brief Model-free retained-team timing authority, publication and stack-reuse regressions.
 *
 * The production scope is tested without a substitute profiler or MPI session.
 * One sink observes exact private-counter totals before lexical storage reuse;
 * delayed contributors expose early snapshots independently of an enclosing join.
 */
#include "kernels/cpu/CPUWorkshareTiming.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace llaminar2::test
{
namespace
{
/** @brief Restore original caller OpenMP policy on every test exit. */
class Policy final
{
public:
    /** @brief Retain caller geometry and require real requested teams. */
    Policy():workers_(omp_get_max_threads()),dynamic_(omp_get_dynamic()){omp_set_dynamic(0);}
    /** @brief Restore caller worker geometry and dynamic policy. */
    ~Policy(){omp_set_num_threads(workers_);omp_set_dynamic(dynamic_);}
private:int workers_,dynamic_;
};
/** @brief Invoke the real scope under standalone or already-retained ownership.
 * @param enabled Original immutable profiling policy; @param calls Shared sink-call count.
 * @param qk Shared exact emitted QK time; @param value Shared exact emitted V time.
 * @param workers Shared observed native team size.
 */
void invoke(bool enabled,std::atomic<int>&calls,std::atomic<std::uint64_t>&qk,
            std::atomic<std::uint64_t>&value,std::atomic<int>&workers)
{
    cpu::CPUWorkshareTiming timing(enabled);
    auto work=timing.worker([&](auto&local){
        const int rank=omp_get_thread_num();
        if(rank==omp_get_num_threads()-1)std::this_thread::sleep_for(std::chrono::milliseconds(2));
        local.qk_ns=std::uint64_t(rank+1);local.value_ns=2*local.qk_ns;
    });
    OMP_WORKSHARE_REGION(work);
    timing.publish([&](std::uint64_t a,std::uint64_t b,int team){
        qk.fetch_add(a,std::memory_order_relaxed);value.fetch_add(b,std::memory_order_relaxed);
        workers.store(team,std::memory_order_relaxed);calls.fetch_add(1,std::memory_order_release);
    });
}
/** @test A standalone owner joins contributions and publishes once for the actual team. */
TEST(CPUWorkshareTiming, StandalonePublishesOneCompleteSnapshot)
{
    const Policy policy;
    for(int team:{1,3,10})for(bool enabled:{false,true})
    {
        omp_set_num_threads(team);std::atomic<int>calls{0},workers{0};std::atomic<std::uint64_t>qk{0},value{0};
        invoke(enabled,calls,qk,value,workers);
        EXPECT_EQ(calls.load(),enabled?1:0);
        EXPECT_EQ(qk.load(),enabled?std::uint64_t(team*(team+1)/2):0);
        EXPECT_EQ(value.load(),2*qk.load());EXPECT_EQ(workers.load(),enabled?team:0);
    }
}
/** @test Retained callers publish after every delayed contribution, before any chosen stack owner retires. */
TEST(CPUWorkshareTiming, RetainedPublicationCompletesBeforeStackReuse)
{
    const Policy policy;
    for(int team:{1,3,10})for(bool enabled:{false,true})
    {
        omp_set_num_threads(team);
        std::atomic<int>calls{0},workers{0};std::atomic<std::uint64_t>qk{0},value{0};
#pragma omp parallel num_threads(team)
        {
            for(int iteration=0;iteration<20;++iteration)
            {
                invoke(enabled,calls,qk,value,workers);
                if(enabled)
                {
                    const auto expected=std::uint64_t(iteration+1)*team*(team+1)/2;
                    EXPECT_EQ(qk.load(std::memory_order_acquire),expected);
                    EXPECT_EQ(calls.load(std::memory_order_acquire),iteration+1);
                }
                // There is deliberately no enclosing join/barrier here. The
                // next invocation may reuse the elected owner's stack before
                // a late peer has finished its old private-state cleanup.
            }
        }
        EXPECT_EQ(calls.load(),enabled?20:0);EXPECT_EQ(value.load(),2*qk.load());
        EXPECT_EQ(qk.load(),enabled?std::uint64_t(20*team*(team+1)/2):0);
        EXPECT_EQ(workers.load(),enabled?team:0);
    }
}
/** @test One-way lifecycle rejects samples or publication after finalization. */
TEST(CPUWorkshareTiming, PublicationStateRejectsRepeatedOperations)
{
    cpu::CPUWorkshareTiming timing(true);int calls=0;
    timing.withThreadSample([](auto&local){local.qk_ns=7;local.value_ns=11;});
    timing.publish([&](auto qk,auto value,int workers){EXPECT_EQ(qk,7);EXPECT_EQ(value,11);EXPECT_EQ(workers,1);++calls;});
    EXPECT_EQ(calls,1);
    EXPECT_THROW(timing.withThreadSample([](auto&){}),std::logic_error);
    EXPECT_THROW(timing.worker([](auto&){}),std::logic_error);
    EXPECT_THROW(timing.publish([](auto,auto,int){}),std::logic_error);
}
} // namespace
} // namespace llaminar2::test
