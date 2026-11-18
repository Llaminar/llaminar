/**
 * @file ROCmRetainedM1DescriptorDemandHostABI.cpp
 * @brief Exercise the actual CXX-to-HIP host function selector without starting a device.
 *
 * GCC and HIP Clang previously disagreed on which bits carried the byte-sized
 * selector in an optimized call. Runtime-derived arguments guard against a
 * constant source-level call silently bypassing that boundary.
 */
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNDevice.h"
#include <array>
#include <cstdio>
#include <cstdlib>

namespace {
using llaminar2::FullNativeHCFFNOutputDemand;
using llaminar2::MoESelectedDescriptorDemand;
using llaminar2::rocm::retainedM1HCFFNKernelFunction;

/**
 * @brief Cross the library ABI with a demand read from runtime state.
 * @param output Publication policy of the original optional HC/FFN outputs.
 * @param descriptor_mode Runtime-encoded selected descriptor publication.
 * @return Exact native function pointer, or null for an invalid mode.
 */
[[gnu::noinline]] const void* nativeSelection(
    FullNativeHCFFNOutputDemand output,const volatile std::uint32_t& descriptor_mode)
{
    return retainedM1HCFFNKernelFunction(output,static_cast<MoESelectedDescriptorDemand>(descriptor_mode));
}

/**
 * @brief Cross the mixed host ABI after deriving the valid scoped enum from a branch.
 * @param output Original optional-output policy.
 * @param descriptor_mode Runtime-owned zero or nonzero reader state.
 * @return Exact original entry for the corresponding declared role-table mode.
 */
[[gnu::noinline]] const void* nativeValidSelection(
    FullNativeHCFFNOutputDemand output,const volatile std::uint32_t& descriptor_mode)
{
    return retainedM1HCFFNKernelFunction(output,descriptor_mode!=0?
        MoESelectedDescriptorDemand::AcquiredBankOnly:MoESelectedDescriptorDemand::ExternalTables);
}
}

/** @return Success when both policies select distinct native entries and invalid modes reject. */
int main()
{
    constexpr auto external=MoESelectedDescriptorDemand::ExternalTables;
    constexpr auto closed=MoESelectedDescriptorDemand::AcquiredBankOnly;
    constexpr auto lean=FullNativeHCFFNOutputDemand::none();
    constexpr auto observed=FullNativeHCFFNOutputDemand::all();
    const std::array expected{
        retainedM1HCFFNKernelFunction(lean,external),retainedM1HCFFNKernelFunction(lean,closed),
        retainedM1HCFFNKernelFunction(observed,external),retainedM1HCFFNKernelFunction(observed,closed)};
    if(expected[2]!=retainedM1HCFFNKernelFunction())return EXIT_FAILURE;
    volatile std::uint32_t role=0;
    for(std::size_t i=0;i<expected.size();++i) {
        role=static_cast<std::uint32_t>(i%2);
        const auto actual=nativeValidSelection(i<2?lean:observed,role);
        if(!actual||actual!=expected[i]) {
            std::fprintf(stderr,"Retained M1 descriptor selector returned an invalid native entry at %zu\n",i);
            return EXIT_FAILURE;
        }
        for(std::size_t earlier=0;earlier<i;++earlier)
            if(actual==expected[earlier])return EXIT_FAILURE;
    }
    for(const auto invalid:{2u,256u,UINT32_MAX}) {
        role=invalid;
        if(nativeSelection(lean,role)||nativeSelection(observed,role)) {
            std::fprintf(stderr,"Retained M1 descriptor selector accepted invalid mode %u\n",invalid);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
