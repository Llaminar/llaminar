/**
 * @file NativeOrderingEventProgram.h
 * @brief Closed existing ordering-event programs derived from actual pooled root presence.
 */
#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace llaminar2 {
/** @brief Fixed program identity; callers cannot author an event count. */
class NativeOrderingEventProgram final {
public:
    enum class Kind {PooledOwnerInitialization,PooledCoordinatorInitialization,PooledSourceInitialization,LookupSubmissionRetirement,ReadAheadParentTerminal};
    /** @return Original setup plus frame/native and optional cache tensor publications. */
    static constexpr NativeOrderingEventProgram pooledOwner(bool hasCache){return {Kind::PooledOwnerInitialization,std::uint8_t(hasCache?7:6)};}
    /** @return Original setup plus frame/native and optional policy tensor publications. */
    static constexpr NativeOrderingEventProgram pooledCoordinator(bool hasPolicy){return {Kind::PooledCoordinatorInitialization,std::uint8_t(hasPolicy?7:6)};}
    /** @return Original setup plus the two actual source tensors. */
    static constexpr NativeOrderingEventProgram pooledSource(){return {Kind::PooledSourceInitialization,6};}
    /** @return Original one-event boundary program, including failed admission/rerecord use. */
    static constexpr NativeOrderingEventProgram lookupSubmissionRetirement(){return {Kind::LookupSubmissionRetirement,0};}
    /** @return Independently named one-event read-ahead parent program. */
    static constexpr NativeOrderingEventProgram readAheadParentTerminal(){return {Kind::ReadAheadParentTerminal,0};}
    Kind kind()const noexcept{return kind_;}
    std::uint8_t nonemptyRootMask()const noexcept{return mask_;}
    std::size_t orderingEvents()const noexcept{return 1u+std::popcount(mask_);}
    /** @brief Validate the actual optional/frame/native root extents before any claims or creators. */
    void requireRootBytes(const std::array<std::size_t,3>& bytes)const{
        std::uint8_t actual=0;for(std::size_t i=0;i<bytes.size();++i)if(bytes[i])actual|=std::uint8_t(1u<<i);
        if(!mask_ || actual!=mask_)throw std::invalid_argument("Native ordering program differs from actual nonempty pooled roots");
    }
    bool operator==(const NativeOrderingEventProgram&)const=default;
private:
    constexpr NativeOrderingEventProgram(Kind kind,std::uint8_t mask):kind_(kind),mask_(mask){}
    Kind kind_;std::uint8_t mask_;
};
}
