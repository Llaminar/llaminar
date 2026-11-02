/** @file Test__HostServicePopulationRelease.cpp
 * @brief Actual diagnostic live-observer helper releases real waiting threads before exception unwind.
 *
 * The failure control cleans up only after recording the invariant failure, so
 * the old missing-release body fails deterministically without hanging a gate.
 */
#include "integration/execution/NativeHostServicePopulationRelease.h"
#include <gtest/gtest.h>
#include <atomic>
#include <stdexcept>
#include <thread>
namespace llaminar2::test {
namespace {
/** @brief Exercise the same actual helper with either successful or throwing observation. */
void exercisePopulationObserver(bool fail){
    std::barrier start(3);std::latch touched(2),release(1);std::atomic<unsigned> retired=0;
    const auto work=[&]{start.arrive_and_wait();touched.count_down();release.wait();++retired;};
    std::thread first(work),second(work);
    const auto observe=[&]{EXPECT_FALSE(release.try_wait());if(fail)throw std::runtime_error("Injected allocating-checkpoint failure");};
    if(fail)EXPECT_THROW(observeLiveHostServicePopulation(start,touched,release,observe),std::runtime_error);
    else EXPECT_NO_THROW(observeLiveHostServicePopulation(start,touched,release,observe));
    EXPECT_TRUE(release.try_wait())<<"Observer unwind must publish release before caller joins owners";
    if(!release.try_wait())release.count_down(); // Negative-control cleanup; the assertion above has already failed.
    first.join();second.join();EXPECT_EQ(retired,2u);
}
}
TEST(HostServicePopulationRelease, SuccessfulObservationPublishesOnce){exercisePopulationObserver(false);}
TEST(HostServicePopulationRelease, ThrowingObservationReleasesBeforeOwnerJoin){exercisePopulationObserver(true);}
}
