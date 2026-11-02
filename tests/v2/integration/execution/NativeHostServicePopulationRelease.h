/** @file NativeHostServicePopulationRelease.h
 * @brief Release diagnostic workers before exception unwinding can join them.
 *
 * This owns one latch publication, not native resources or allocation state.
 * The observer may allocate/throw only while this guard owns release ordering.
 */
#pragma once
#include <barrier>
#include <latch>
#include <utility>
namespace llaminar2::test {
/** @brief The single release authority; all task consumers receive only a const wait view. */
class HostServicePopulationRelease final {
public:
    explicit HostServicePopulationRelease(std::latch& release)noexcept:release_(release){}
    ~HostServicePopulationRelease()noexcept{publish();}
    HostServicePopulationRelease(const HostServicePopulationRelease&)=delete;
    HostServicePopulationRelease& operator=(const HostServicePopulationRelease&)=delete;
    /** @brief Publish once on successful observation or automatically during unwind. */
    void publish()noexcept{if(!published_){release_.count_down();published_=true;}}
private:
    std::latch& release_;
    bool published_=false;
};
/** @brief Hold all actual TLS tasks live for observation while guaranteeing release on observer failure. */
template<class Observer>
void observeLiveHostServicePopulation(std::barrier<>& start,std::latch& touched,std::latch& release,Observer&& observer){
    HostServicePopulationRelease guard(release);
    start.arrive_and_wait();touched.wait();
    std::forward<Observer>(observer)();
    guard.publish();
}
}
