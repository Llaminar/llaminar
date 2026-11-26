/**
 * @file CPUNativeQuantisedVerifierPolicyLookup.cpp
 * @brief Single compiled translation unit owning generated CPU NativeQuantised grouped verifier policy table.
 *
 * Isolates the 118,510-line generated verifier-row schedule table into its own TU so it
 * compiles once in parallel with the M=1 decode policy lookup TU.
 */
#include "kernels/cpu/gemm/CPUNativeQuantisedPolicyLookup.h"
#include <cstddef>
#include <cstdint>

namespace llaminar2::cpu::native_quantised::verifier_policy_detail
{
#include "kernels/cpu/gemm/CPUNativeQuantisedVerifierRowsPolicyGenerated.inc"
} // namespace llaminar2::cpu::native_quantised::verifier_policy_detail

namespace llaminar2::cpu::native_quantised::generated
{

bool selectCPUNativeQuantisedVerifierRowsGeneratedPolicy(
    CPUNativeQuantisedBuildISA build_isa,
    CPUNativeQuantisedRuntimeISA runtime_isa, int threads,
    uint8_t codebook, int m, int n, int k, int k_tiles,
    CPUNativeQuantisedVerifierRowsPolicy &policy, float *measured_speedup)
{
    namespace gen = verifier_policy_detail::llaminar2::cpu::native_quantised::generated;
    gen::CPUNativeQuantisedVerifierRowsPolicy raw{};
    if (!gen::selectCPUNativeQuantisedVerifierRowsGeneratedPolicy(
            static_cast<gen::CPUNativeQuantisedBuildISA>(build_isa),
            static_cast<gen::CPUNativeQuantisedRuntimeISA>(runtime_isa),
            threads, codebook, m, n, k, k_tiles, raw, measured_speedup))
        return false;
    policy = static_cast<CPUNativeQuantisedVerifierRowsPolicy>(raw);
    return true;
}

} // namespace llaminar2::cpu::native_quantised::generated
