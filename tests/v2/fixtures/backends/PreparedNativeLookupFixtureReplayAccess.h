/** @file PreparedNativeLookupFixtureReplayAccess.h
 * @brief Recorded source inputs to the actual fixture lifecycle, never production class installation.
 */
#pragma once
#include "PreparedNativeLookupFixture.h"
#include "NativeContextPreparationReplayAccess.h"
namespace llaminar2::test {
/** @brief Only recorded source construction and validated reset substitute native SDK operations. */
struct PreparedNativeLookupFixtureReplayAccess final {
    /** @return Same source acquisition, freeze and admission implementation consumed by native fixtures. */
    static std::unique_ptr<PreparedNativeLookupFixture> prepare(std::shared_ptr<NativeContextPreparation> source,std::span<const DeviceId> devices){
        auto result=std::unique_ptr<PreparedNativeLookupFixture>(new PreparedNativeLookupFixture({devices.begin(),devices.end()}));
        result->bindSource(std::move(source));return result;
    }
    /** @brief Apply the canonical recorded reset hook after original borrows and payloads retire. */
    static void retire(PreparedNativeLookupFixture& fixture){
        fixture.permits_.clear();
        for(const auto device:fixture.devices_){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);NativeContextPreparationReplayAccess::physicalReset(scope);}
        fixture.retired_=true;fixture.requireExactRetired();
    }
};
}
