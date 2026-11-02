/** @file RetainedHostServiceFixture.h
 * @brief Explicit original or recorded fixture declaration, priced before the fixture's final PMA.
 */
#pragma once
#include "RetainedHostServiceReplayAccess.h"
namespace llaminar2::test {
/** @brief One cold fixture instance; no optional authority or legacy thread constructor exists. */
class RetainedHostServiceFixture final {
public:
    /** @return Actual installed-class declaration; a missing class rejects when its BOM is requested. */
    static RetainedHostServiceFixture original(RetainedHostServiceRole role,PhysicalMemoryAllocatorIdentity consumer,std::uint64_t ordinal=0){return {RetainedHostServiceProgram::original(role),{consumer.world_rank,consumer.device,role,ordinal}};}
    /** @return Sealed recorded metadata for hardware-free actual-thread replays only. */
    static RetainedHostServiceFixture recorded(RetainedHostServiceRole role,PhysicalMemoryAllocatorIdentity consumer,std::uint64_t ordinal=0){return {RetainedHostServiceReplayAccess::program(role),{consumer.world_rank,consumer.device,role,ordinal}};}
    /** @return Original typed exact/opaque contribution, before admission. */
    PhysicalMemoryPlan bom(PhysicalMemoryResource cpu)const{return program_.bom(cpu);}
    /** @brief Transfer one declared instance only after original admission. */
    RetainedHostServiceUseToken take(std::shared_ptr<PhysicalMemoryAuthority> memory){
        if(taken_)throw std::logic_error("Fixture CPU service instance was already consumed");taken_=true;
        RetainedHostServiceDeclarations::Builder builder(key_.worldRank);builder.add(key_,program_);auto declared=std::move(builder).freeze();
        auto bound=declared->bind(std::move(memory),std::span{&key_,std::size_t{1}});return bound->take(key_);
    }
private:
    RetainedHostServiceFixture(RetainedHostServiceProgram program,RetainedHostServiceInstance key):program_(std::move(program)),key_(key){}
    RetainedHostServiceProgram program_;
    RetainedHostServiceInstance key_;
    bool taken_=false;
};
}
