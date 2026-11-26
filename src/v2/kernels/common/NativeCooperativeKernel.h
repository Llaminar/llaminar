/**
 * @file NativeCooperativeKernel.h
 * @brief Cold, exact-program admission for a native cooperative kernel node.
 *
 * The existing immutable backend inventory owns physical capabilities. This
 * value owns only one loaded function's generation/resource observation; it
 * contains no allocation, graph, stream, mutable device state or hardware cache.
 */
#pragma once
#include "backends/DeviceId.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2
{
/** @brief Select whether admission requires native cooperative launch capability. */
enum class NativeResidentLaunchPolicy { Cooperative, Ordinary };
/** @brief Native compiler/runtime boundary, never an inferred device fallback. */
enum class NativeCooperativeBackend { CUDA, ROCm };
template<NativeCooperativeBackend> class NativeCooperativeKernelBoundary;

/** @brief Declarative processor reservation for co-resident background services. */
struct NativeCoResidencyReservation
{
    unsigned reserved_processors=0; ///< Physical CUs (ROCm) or SMs (CUDA) held back for co-resident services.

    /** @return Unreserved policy permitting full-device resident grids. */
    static constexpr NativeCoResidencyReservation none() noexcept {return {};}
    /** @return Reserve @p processors CUs/SMs for co-resident tier maintenance or transfers. */
    static constexpr NativeCoResidencyReservation reserveProcessors(unsigned processors) noexcept
    {
        return {processors};
    }
    constexpr bool operator==(const NativeCoResidencyReservation&) const noexcept = default;
};

/** @brief Exact loaded-program resources observed outside capture. */
struct NativeCooperativeKernelResources
{
    unsigned registers_per_thread=0,maximum_threads_per_block=0,blocks_per_processor=0;
    std::size_t private_bytes=0,static_shared_bytes=0;
};

/** @brief A successful sealed resident-grid admission for one program generation. */
class NativeCooperativeKernelAdmission final
{
public:
    /** @return Original participant whose worker admitted the loaded program. */
    DeviceId device() const noexcept {return device_;}
    /** @return Actual native function embedded in every launch using this value. */
    const void* function() const noexcept {return function_;}
    /** @return Existing caller-owned program/workspace lifecycle generation. */
    std::uint64_t generation() const noexcept {return generation_;}
    /** @return Exact physical CTA width used for the occupancy observation. */
    unsigned threads() const noexcept {return threads_;}
    /** @return Exact dynamic LDS/smem bytes used for the occupancy observation. */
    std::size_t dynamicSharedBytes() const noexcept {return dynamic_shared_;}
    /** @return Declarative co-residency reservation honored by this admission. */
    NativeCoResidencyReservation coResidency() const noexcept {return co_residency_;}
    /** @return Physical processors available to this resident kernel after co-residency reservation. */
    unsigned availableProcessors() const noexcept {return available_processors_;}
    /** @return Maximum resident grid from cached hardware, co-residency reservation and actual occupancy. */
    unsigned maximumGrid() const noexcept {return maximum_grid_;}
    /** @return The explicit submission policy whose capability was admitted. */
    NativeResidentLaunchPolicy policy() const noexcept {return policy_;}
    /** @return Loaded compiler resource evidence, independent of physical facts. */
    const NativeCooperativeKernelResources& resources() const noexcept {return resources_;}
private:
    template<NativeCooperativeBackend> friend class NativeCooperativeKernelBoundary;
    /** @brief Seal validated native observations; ordinary callers cannot forge residency. */
    NativeCooperativeKernelAdmission(DeviceId device,const void* function,std::uint64_t generation,
        unsigned threads,std::size_t shared,unsigned available_processors,unsigned maximum_grid,
        NativeCooperativeKernelResources resources,NativeResidentLaunchPolicy policy,
        NativeCoResidencyReservation co_residency=NativeCoResidencyReservation::none())
        :device_(device),function_(function),generation_(generation),threads_(threads),dynamic_shared_(shared),
         available_processors_(available_processors),maximum_grid_(maximum_grid),resources_(resources),
         policy_(policy),co_residency_(co_residency) {}
    DeviceId device_;
    const void* function_;
    std::uint64_t generation_;
    unsigned threads_;
    std::size_t dynamic_shared_;
    unsigned available_processors_;
    unsigned maximum_grid_;
    NativeCooperativeKernelResources resources_;
    NativeResidentLaunchPolicy policy_;
    NativeCoResidencyReservation co_residency_;
};
}
