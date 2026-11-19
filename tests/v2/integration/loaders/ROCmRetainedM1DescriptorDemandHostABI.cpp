/**
 * @file ROCmRetainedM1DescriptorDemandHostABI.cpp
 * @brief Exercise the actual CXX-to-HIP host function selector without starting a device.
 *
 * GCC and HIP Clang previously disagreed on which bits carried the byte-sized
 * selector in an optimized call. Runtime-derived arguments guard against a
 * constant source-level call silently bypassing that boundary.
 */
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNDevice.h"
#include "kernels/rocm/gemm/retained_hc_ffn/ROCmRetainedM1HCFFNBinding.h"
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
/** @brief Authenticate counted and fully-active ordinary rows without reading device-owned data. */
bool originalRows()
{
    using namespace llaminar2;
    const auto fully=DeviceRowRange::fullyActive(1);
    rocm::PersistentHCReadArgs hc{.publication={.rows=fully},.up={.rows=fully}};
    hc.norm.extent.rows=1;
    using Binding=rocm::ROCmRetainedM1HCFFNBinding;
    if(Binding::validateRowAuthority(fully,hc)!=nullptr)return false;
    std::array<std::int32_t,2> originalCount{0,1}; // cold opaque owners with distinct addresses
    for(std::size_t owner=0;owner<originalCount.size();++owner)
    {
        const auto& count=originalCount[owner];
        const auto row=DeviceRowRange::deviceCounted(1,&count);
        hc.norm.extent.liveRows=&count;
        hc.up.rows=hc.publication.rows=row;
        if(Binding::validateRowAuthority(row,hc)!=&count)return false;
        for(const auto wrong:{fully,DeviceRowRange::deviceCounted(1,&originalCount[1-owner]),
                               row.repeatEachRow(2),DeviceRowRange::deviceCounted(2,&count).slice(1,1)})
        {
            try { (void)Binding::validateRowAuthority(wrong,hc); return false; }
            catch(const std::invalid_argument&) {}
        }
    }
    return true;
}

/**
 * @brief Exercise the real mixed GCC/HIP selector with a runtime-derived source and role.
 * @return The selected native program, or null for a foreign policy or format.
 */
[[gnu::noinline]] const void* nativeHistogramSelection(FullNativeHCFFNOutputDemand output,
    MoESelectedDescriptorDemand descriptors, llaminar2::rocm::ResidentMoEFFNExecutionMasks masks,
    const volatile std::uint32_t& role)
{
    using Policy=llaminar2::rocm::RetainedM1NativeHistogramPolicy;
    return retainedM1HCFFNKernelFunction(output,descriptors,masks,static_cast<Policy>(role));
}
}

/** @return Success when complete native entry and graph row identities are preserved. */
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
    using namespace llaminar2;
    constexpr std::uint32_t q4=1u<<5, q5=1u<<7, q8=1u<<19;
    for(const auto sources:{rocm::ResidentMoEFFNExecutionMasks{q4,q5,q8,q8},
        rocm::ResidentMoEFFNExecutionMasks{q4,q8,q8,q8},rocm::ResidentMoEFFNExecutionMasks{q5,q8,q8,q8}})
        for(const auto output:{lean,observed})for(const auto demand:{closed,external})
        {
            role=0;
            const auto routeOnly=nativeHistogramSelection(output,demand,sources,role);
            role=1;
            const auto observedHistogram=nativeHistogramSelection(output,demand,sources,role);
            if(!routeOnly||!observedHistogram||routeOnly==observedHistogram)return EXIT_FAILURE;
            for(const auto invalid:{2u,256u,UINT32_MAX})
            {
                role=invalid;
                if(nativeHistogramSelection(output,demand,sources,role))return EXIT_FAILURE;
            }
        }
    if(!originalRows())
    {
        std::fprintf(stderr,"Retained M1 row authority matrix failed\n");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
