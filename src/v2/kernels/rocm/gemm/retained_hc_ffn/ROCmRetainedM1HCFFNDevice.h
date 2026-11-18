/**
 * @file ROCmRetainedM1HCFFNDevice.h
 * @brief Exact native leaf and shared storage geometry for the retained M1 body.
 */
#pragma once
#include "kernels/rocm/gemm/ROCmFullNativeHCFFN.h"

namespace llaminar2 {
namespace rocm {
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

/** @return Exact resident native function for the admitted fixed geometry. */
const void* retainedM1HCFFNKernelFunction() noexcept;

/** @return Native function for the original sealed public output demand. */
const void* retainedM1HCFFNKernelFunction(FullNativeHCFFNOutputDemand demand,
    MoESelectedDescriptorDemand descriptors=MoESelectedDescriptorDemand::ExternalTables) noexcept;

/**
 * @brief Admit the exact ordinary resident function on its original worker.
 * @param device Original ROCm worker participant.
 * @param full Exact retained operands checked by their original owners.
 * @param masks Authenticated original complete source and execution envelope.
 * @param generation Original workspace generation.
 * @return Native resource evidence for the only submitted full entry.
 */
PreparedMoEFFNProgram prepareRetainedM1HCFFN(DeviceId device,FullNativeHCFFNArgs& full,
    ResidentMoEFFNPolicyMasks masks,std::uint64_t generation);

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
    const DeviceNativeQuantisedMatrixDesc& shared_down,void* scratch,std::uint64_t generation,void* stream);
}
}
