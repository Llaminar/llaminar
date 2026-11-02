/**
 * @file ReadAheadNativeResourceObligation.h
 * @brief Closed move-only ownership of the original read-ahead PMA reservations.
 *
 * The original generation retains this leaf without a worker or byte balance.
 * Token destruction returns exactly the existing canonical reservations only
 * after their certified physical lifetime. Process-runtime tokens remain with
 * the original process owner through OS exit; no second authority is created.
 */
#pragma once
#include "NativeContextRetirement.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <optional>

namespace llaminar2 {
class ReadAheadNativeResourceFamily;
class ReadAheadNativeParentTerminal;
/** @brief Source-certificate retirement policy, never selected as a runtime fallback. */
enum class ReadAheadNativeRetirementPolicy {HandleDestruction,OriginalContextReset,OriginalProcessRuntime};
/** @brief Length-delimited immutable process class; execution generation is independently authenticated. */
struct ReadAheadNativeResourceClosure;
std::string readAheadProcessPopulationIdentity(const ReadAheadNativeResourceClosure&,const std::string&,std::size_t,std::size_t);
/** @brief Closed leaf consumed by the original canonical generation owner. */
class ReadAheadNativeResourceObligation final:public IContextNativeRetirementObligation {
public:
    /** @return Exact original PMA retained by the existing tokens. */
    const std::shared_ptr<PhysicalMemoryAuthority>& originalAuthority()const noexcept override{return memory_;}
    /** @return Exact physical resource device. */
    DeviceId device()const noexcept override{return device_;}
    /** @return Exact original pool generation. */
    std::uint64_t poolGeneration()const noexcept override{return generation_;}
    /** @return Actual source/runtime/population certificate identity. */
    const std::string& certificateIdentity()const noexcept override{return certificate_;}
    /** @return Closed certificate physical lifetime, independent from handle teardown. */
    NativePhysicalLifetime physicalLifetime()const noexcept override{return lifetime_;}
private:
    friend struct test::ReadAheadNativeResourceReplayAccess;
    friend class ReadAheadNativeResourceFamily;
    friend class ReadAheadNativeParentTerminal;
    friend class NativeOrderingEventGroup;
    /** @brief Move only original tokens; the independently scoped zero GPU allowance has no token. */
    ReadAheadNativeResourceObligation(std::shared_ptr<PhysicalMemoryAuthority> memory,DeviceId device,
        std::uint64_t generation,std::string certificate,PhysicalMemoryOwnerReservation host,
        std::optional<PhysicalMemoryOwnerReservation> gpu,NativePhysicalLifetime lifetime=NativePhysicalLifetime::ContextGeneration)noexcept
        :memory_(std::move(memory)),device_(device),generation_(generation),certificate_(std::move(certificate)),
         lifetime_(lifetime),host_(std::move(host)),gpu_(std::move(gpu)){}
    std::shared_ptr<PhysicalMemoryAuthority> memory_;
    DeviceId device_;
    std::uint64_t generation_;
    std::string certificate_;
    NativePhysicalLifetime lifetime_;
    PhysicalMemoryOwnerReservation host_;
    std::optional<PhysicalMemoryOwnerReservation> gpu_;
};
}
