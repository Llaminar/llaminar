/**
 * @file ROCmRetainedM1HCFFNDevice.h
 * @brief Exact native leaf and shared storage geometry for the retained M1 body.
 */
#pragma once
#include "execution/moe/ExpertHistogramSource.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"

namespace llaminar2 {
namespace rocm {
/** @brief Number of routed slots admitted by this retained QFN decode specialization. */
inline constexpr unsigned retainedM1SelectedExperts = 10;

/** @return Compiler-owned retained scratch payload before slot-stride padding. */
std::size_t retainedM1HCFFNScratchBytes() noexcept;

/** @return Existing barrier slot plus its aligned private retained payload. */
inline std::size_t fullNativeHCFFNSlotBytes(FullNativeHCFFNImplementation implementation) noexcept
{
    constexpr auto prefix=(FullNativeHCFFNBarrierLayout::bytes+15)&~std::size_t(15);
    return implementation==FullNativeHCFFNImplementation::RetainedGFX906M1
        ? (prefix+retainedM1HCFFNScratchBytes()+15)&~std::size_t(15)
        : FullNativeHCFFNBarrierLayout::bytes;
}

/** @return Address of the retained payload within the existing exclusive slot. */
inline void* retainedM1HCFFNScratch(void* slot) noexcept
{
    constexpr auto prefix=(FullNativeHCFFNBarrierLayout::bytes+15)&~std::size_t(15);
    return static_cast<std::byte*>(slot)+prefix;
}

/** @brief Full-width GCC-to-HIP host ABI encoding of the original producer role. */
enum class RetainedM1NativeHistogramPolicy : std::uint32_t { RouteOnly, ObserveDecode };

/** @brief Preserve the original cold producer authority at the mixed host ABI boundary. */
inline RetainedM1NativeHistogramPolicy retainedM1NativeHistogramPolicy(MoERuntimeHistogramProducerRole role)
{
    switch(role)
    {
    case MoERuntimeHistogramProducerRole::NotProducer:return RetainedM1NativeHistogramPolicy::RouteOnly;
    case MoERuntimeHistogramProducerRole::ProductionDecode:return RetainedM1NativeHistogramPolicy::ObserveDecode;
    }
    throw std::invalid_argument("Retained M1 HC/FFN received a foreign histogram authority");
}

/** @return Exact resident native function for the admitted fixed geometry. */
const void* retainedM1HCFFNKernelFunction() noexcept;

/** @return Native function for the original sealed public output demand. */
const void* retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand demand,
    MoESelectedDescriptorDemand descriptors=MoESelectedDescriptorDemand::ExternalTables) noexcept;

/** @brief Concrete routed native rows authorized by the original source envelope. */
enum class RetainedM1RoutedFormat : std::uint8_t { Q4K_Q51, Q4K_Q80, Q5K_Q80 };

/**
 * @brief Resolve a single reachable routed encoding per role from the original producer.
 * @param execution Original retained owner's physical decoder masks.
 * @return The exact routed program, or no entry for a foreign or widened envelope.
 */
inline std::optional<RetainedM1RoutedFormat> retainedM1RoutedFormat(ResidentMoEFFNExecutionMasks execution) noexcept
{
    constexpr std::uint32_t q4=1u<<5, q5=1u<<7, q8=1u<<19;
    if(execution.routed_gate==q4&&execution.routed_down==q5)return RetainedM1RoutedFormat::Q4K_Q51;
    if(execution.routed_gate==q4&&execution.routed_down==q8)return RetainedM1RoutedFormat::Q4K_Q80;
    if(execution.routed_gate==q5&&execution.routed_down==q8)return RetainedM1RoutedFormat::Q5K_Q80;
    return std::nullopt;
}

/** @return The exact admitted function for the original retained format and publication policy. */
const void* retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand demand,
    MoESelectedDescriptorDemand descriptors, ResidentMoEFFNExecutionMasks execution,
    RetainedM1NativeHistogramPolicy histogram=RetainedM1NativeHistogramPolicy::ObserveDecode) noexcept;

/**
 * @brief Admit the exact ordinary resident function on its original worker.
 * @param device Original ROCm worker participant.
 * @param full Exact retained operands checked by their original owners.
 * @param masks Authenticated original complete source and execution envelope.
 * @param generation Original workspace generation.
 * @return Native resource evidence for the only submitted full entry.
 */
PreparedMoEFFNProgram prepareRetainedM1HCFFN(DeviceId device,FullNativeHCFFNArgs& full,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation,
    RetainedM1NativeHistogramPolicy histogram=RetainedM1NativeHistogramPolicy::ObserveDecode);

/**
 * @brief Launch the exact admitted retained body on the existing stream.
 * @param full Original sealed complete participant-local operation.
 * @param shared_gate Original prepared shared gate projection.
 * @param shared_up Original prepared shared up projection.
 * @param shared_down Original prepared shared down projection.
 * @param scratch Beginning of its exclusive retained workspace slot payload.
 * @param generation Original workspace generation.
 * @param stream Exact original nondefault graph stream.
 */
void launchRetainedM1HCFFNDevice(const FullNativeHCFFNArgs& full,
    const DeviceNativeQuantisedMatrixDesc& shared_gate,const DeviceNativeQuantisedMatrixDesc& shared_up,
    const DeviceNativeQuantisedMatrixDesc& shared_down,void* scratch,std::uint64_t generation,void* stream,
    RetainedM1NativeHistogramPolicy histogram=RetainedM1NativeHistogramPolicy::ObserveDecode);
}
}
