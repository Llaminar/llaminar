/**
 * @file CPUNativeQuantisedDecodePolicyLookup.cpp
 * @brief Single compiled translation unit owning generated CPU NativeQuantised M=1 decode policy table.
 *
 * Isolates the 133,353-line generated decode schedule table into its own TU so it
 * compiles once in parallel with the verifier-row policy lookup TU.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedPolicyLookup.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2::cpu::native_quantised::decode_policy_detail
{
#include "kernels/cpu/gemm/CPUNativeQuantisedDecodePolicyGenerated.inc"
} // namespace llaminar2::cpu::native_quantised::decode_policy_detail

namespace llaminar2::cpu::native_quantised::generated
{

bool selectCPUNativeQuantisedDecodeGeneratedPolicy(
    CPUNativeQuantisedDecodeBuildISA build_isa,
    CPUNativeQuantisedDecodeRuntimeISA runtime_isa, int threads,
    uint8_t codebook, int n, int k, bool serial_kpart, int k_tiles,
    CPUNativeQuantisedDecodePolicy &policy)
{
    namespace gen = decode_policy_detail::llaminar2::cpu::native_quantised::generated;
    gen::CPUNativeQuantisedDecodePolicy raw{};
    if (!gen::selectCPUNativeQuantisedDecodeGeneratedPolicy(
            static_cast<gen::CPUNativeQuantisedDecodeBuildISA>(build_isa),
            static_cast<gen::CPUNativeQuantisedDecodeRuntimeISA>(runtime_isa),
            threads, codebook, n, k, serial_kpart, k_tiles, raw))
        return false;
    policy = static_cast<CPUNativeQuantisedDecodePolicy>(raw);
    return true;
}

} // namespace llaminar2::cpu::native_quantised::generated
